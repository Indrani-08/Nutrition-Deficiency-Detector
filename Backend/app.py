import random
from datetime import datetime, timedelta

from flask_mail import Mail, Message
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np

from utils.model_loader import model, CLASS_NAMES
from utils.preprocess import preprocess_image
from utils.recommendations import RECOMMENDATIONS
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
CORS(app)

# ---------------- MAIL CONFIG ---------------- #

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True

app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_USERNAME")

mail = Mail(app)
# Stores OTP temporarily
otp_storage = {}

# ---------------- DATABASE CONFIG ---------------- #

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ---------------- USER MODEL ---------------- #

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    prediction = db.Column(db.String(100), nullable=False)

    confidence = db.Column(db.Float, nullable=False)

    risk_level = db.Column(db.String(50), nullable=False)

    description = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

# ---------------- UPLOAD FOLDER ---------------- #

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- HOME ---------------- #

@app.route("/")
def home():
    return "Backend is running successfully!"

# ---------------- REGISTER ---------------- #

@app.route("/register", methods=["POST"])
def register():

    data = request.get_json()

    fullname = data.get("fullname")
    email = data.get("email")
    password = data.get("password")

    if not fullname or not email or not password:
        return jsonify({
            "success": False,
            "message": "Please fill all the fields."
        }), 400

    existing_user = User.query.filter_by(email=email).first()

    if existing_user:
        return jsonify({
            "success": False,
            "message": "Email already registered."
        }), 409

    hashed_password = generate_password_hash(password)

    new_user = User(
        fullname=fullname,
        email=email,
        password=hashed_password
    )

    db.session.add(new_user)
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Account created successfully."
    }), 201

# ---------------- LOGIN ---------------- #

@app.route("/login", methods=["POST"])
def login():

    data = request.get_json()

    email = data.get("email")
    password = data.get("password")

    user = User.query.filter_by(email=email).first()

    if user is None:
        return jsonify({
            "success": False,
            "message": "Email not found."
        }), 404

    if not check_password_hash(user.password, password):
        return jsonify({
            "success": False,
            "message": "Incorrect password."
        }), 401

    return jsonify({
        "success": True,
        "message": "Login successful.",
        "user": {
            "id": user.id,
            "fullname": user.fullname,
            "email": user.email
        }
    }), 200

@app.route("/forgot-password", methods=["POST"])
def forgot_password():

    data = request.get_json()
    email = data.get("email")

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({
            "success": False,
            "message": "Email not registered."
        }), 404

    otp = str(random.randint(100000, 999999))

    otp_storage[email] = {
        "otp": otp,
        "expiry": datetime.now() + timedelta(minutes=5)
    }

    msg = Message(
        subject="Nail Nutrition Password Reset OTP",
        recipients=[email]
    )

    msg.body = f"""
Hello {user.fullname},

Your OTP for password reset is:

{otp}

This OTP is valid for 5 minutes.

If you did not request this, please ignore this email.

Thank you,
Nail Nutrition Team
"""

    mail.send(msg)

    return jsonify({
        "success": True,
        "message": "OTP sent successfully."
    })

@app.route("/verify-otp", methods=["POST"])
def verify_otp():

    data = request.get_json()

    email = data.get("email")
    otp = data.get("otp")

    if email not in otp_storage:
        return jsonify({
            "success": False,
            "message": "OTP not found."
        }), 404

    saved_data = otp_storage[email]

    if datetime.now() > saved_data["expiry"]:
        del otp_storage[email]
        return jsonify({
            "success": False,
            "message": "OTP has expired."
        }), 400

    if otp != saved_data["otp"]:
        return jsonify({
            "success": False,
            "message": "Invalid OTP."
        }), 400

    return jsonify({
        "success": True,
        "message": "OTP verified successfully."
    })

@app.route("/reset-password", methods=["POST"])
def reset_password():

    data = request.get_json()

    email = data.get("email")
    new_password = data.get("new_password")

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({
            "success": False,
            "message": "User not found."
        }), 404

    user.password = generate_password_hash(new_password)

    db.session.commit()

    if email in otp_storage:
        del otp_storage[email]

    return jsonify({
        "success": True,
        "message": "Password updated successfully."
    })

# ---------------- TEST EMAIL ---------------- #

@app.route("/test-email")
def test_email():
    try:
        msg = Message(
            subject="Nail Nutrition Test Email",
            recipients=["harshithaguru10@gmail.com"]
        )

        msg.body = """
Hello!

This is a test email from the Nail Nutrition project.

If you received this email, your email integration is working successfully.

Thank you!
"""

        mail.send(msg)

        return jsonify({
            "success": True,
            "message": "Test email sent successfully."
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })

# ================= SAVE REPORT =================

@app.route("/save-report", methods=["POST"])
def save_report():

    data = request.get_json()

    user_id = data.get("user_id")
    prediction = data.get("prediction")
    confidence = data.get("confidence")
    risk_level = data.get("risk_level")
    description = data.get("description")

    report = Report(
        user_id=user_id,
        prediction=prediction,
        confidence=confidence,
        risk_level=risk_level,
        description=description
    )

    db.session.add(report)
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Report saved successfully."
    })

@app.route("/reports/<int:user_id>", methods=["GET"])
def get_reports(user_id):

    reports = Report.query.filter_by(user_id=user_id).order_by(
        Report.created_at.desc()
    ).all()

    report_list = []

    for report in reports:
        report_list.append({
            "id": report.id,
            "prediction": report.prediction,
            "confidence": report.confidence,
            "risk_level": report.risk_level,
            "description": report.description,
            "created_at": report.created_at.strftime("%d-%m-%Y %H:%M")
        })

    return jsonify(report_list)

@app.route("/report/<int:report_id>", methods=["DELETE"])
def delete_report(report_id):

    report = Report.query.get(report_id)

    if not report:
        return jsonify({
            "success": False,
            "message": "Report not found."
        }), 404


    db.session.delete(report)
    db.session.commit()

    return jsonify({
        "success": True,
        "message": "Report deleted successfully."
    })


# ---------------- PREDICT ---------------- #

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = preprocess_image(filepath)

    predictions = model.predict(image, verbose=0)

    predicted_index = np.argmax(predictions[0])

    prediction = CLASS_NAMES[predicted_index]

    confidence = float(predictions[0][predicted_index] * 100)

    recommendation = RECOMMENDATIONS[prediction]

    return jsonify({
        "prediction": recommendation["title"],
        "confidence": round(confidence, 2),
        "risk_level": recommendation["risk_level"],
        "description": recommendation["description"],
        "possible_indication": recommendation["possible_indication"],
        "symptoms": recommendation["symptoms"],
        "foods": recommendation["foods"],
        "home_remedies": recommendation["home_remedies"],
        "lifestyle": recommendation["lifestyle"],
        "doctor_advice": recommendation["doctor_advice"],
        "disclaimer": "This application provides an AI-assisted screening result based on fingernail images. It is intended for educational purposes only and should not be considered a medical diagnosis. Always consult a qualified healthcare professional for proper evaluation and treatment."
    })



# ---------------- RUN APP ---------------- #

if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)