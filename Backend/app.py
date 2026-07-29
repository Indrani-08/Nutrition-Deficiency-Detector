import os
import random
from datetime import datetime, timedelta

import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mail import Mail, Message
from flask_sqlalchemy import SQLAlchemy

from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv

from utils.model_loader import get_model, CLASS_NAMES
from utils.preprocess import preprocess_image
from utils.recommendations import RECOMMENDATIONS


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

app = Flask(__name__)

CORS(app)


# ============================================================
# EMAIL CONFIGURATION
# ============================================================

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True

app.config["MAIL_USERNAME"] = os.environ.get("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.environ.get("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.environ.get("MAIL_USERNAME")

mail = Mail(app)


# Temporary OTP storage
otp_storage = {}


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# ============================================================
# DATABASE MODELS
# ============================================================

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    fullname = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    password = db.Column(
        db.String(255),
        nullable=False
    )


class Report(db.Model):
    id = db.Column(
        db.Integer,
        primary_key=True
    )

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("user.id"),
        nullable=False
    )

    prediction = db.Column(
        db.String(100),
        nullable=False
    )

    confidence = db.Column(
        db.Float,
        nullable=False
    )

    risk_level = db.Column(
        db.String(50),
        nullable=False
    )

    description = db.Column(db.Text)

    created_at = db.Column(
        db.DateTime,
        default=db.func.current_timestamp()
    )


# Create database tables when Gunicorn imports the application
with app.app_context():
    db.create_all()


# ============================================================
# UPLOAD DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "success": True,
        "message": "Nail Nutrition backend is running successfully."
    })


# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["POST"])
def register():

    data = request.get_json(silent=True) or {}

    fullname = data.get("fullname")
    email = data.get("email")
    password = data.get("password")

    if not fullname or not email or not password:

        return jsonify({
            "success": False,
            "message": "Please fill all the fields."
        }), 400


    existing_user = User.query.filter_by(
        email=email
    ).first()


    if existing_user:

        return jsonify({
            "success": False,
            "message": "Email already registered."
        }), 409


    hashed_password = generate_password_hash(
        password
    )


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


# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["POST"])
def login():

    data = request.get_json(silent=True) or {}

    email = data.get("email")
    password = data.get("password")


    if not email or not password:

        return jsonify({
            "success": False,
            "message": "Email and password are required."
        }), 400


    user = User.query.filter_by(
        email=email
    ).first()


    if user is None:

        return jsonify({
            "success": False,
            "message": "Email not found."
        }), 404


    if not check_password_hash(
        user.password,
        password
    ):

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


# ============================================================
# FORGOT PASSWORD
# ============================================================

@app.route("/forgot-password", methods=["POST"])
def forgot_password():

    data = request.get_json(silent=True) or {}

    email = data.get("email")


    if not email:

        return jsonify({
            "success": False,
            "message": "Email is required."
        }), 400


    user = User.query.filter_by(
        email=email
    ).first()


    if not user:

        return jsonify({
            "success": False,
            "message": "Email not registered."
        }), 404


    otp = str(
        random.randint(
            100000,
            999999
        )
    )


    otp_storage[email] = {

        "otp": otp,

        "expiry":
            datetime.now()
            + timedelta(minutes=5)

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


    try:

        mail.send(msg)

    except Exception as e:

        print(
            f"Email sending failed: {e}"
        )

        return jsonify({
            "success": False,
            "message": "Unable to send OTP email."
        }), 500


    return jsonify({
        "success": True,
        "message": "OTP sent successfully."
    })


# ============================================================
# VERIFY OTP
# ============================================================

@app.route("/verify-otp", methods=["POST"])
def verify_otp():

    data = request.get_json(silent=True) or {}

    email = data.get("email")
    otp = data.get("otp")


    if not email or not otp:

        return jsonify({
            "success": False,
            "message": "Email and OTP are required."
        }), 400


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


    if str(otp) != saved_data["otp"]:

        return jsonify({
            "success": False,
            "message": "Invalid OTP."
        }), 400


    return jsonify({
        "success": True,
        "message": "OTP verified successfully."
    })


# ============================================================
# RESET PASSWORD
# ============================================================

@app.route("/reset-password", methods=["POST"])
def reset_password():

    data = request.get_json(silent=True) or {}

    email = data.get("email")

    new_password = data.get(
        "new_password"
    )


    if not email or not new_password:

        return jsonify({
            "success": False,
            "message":
                "Email and new password are required."
        }), 400


    user = User.query.filter_by(
        email=email
    ).first()


    if not user:

        return jsonify({
            "success": False,
            "message": "User not found."
        }), 404


    user.password = generate_password_hash(
        new_password
    )


    db.session.commit()


    if email in otp_storage:

        del otp_storage[email]


    return jsonify({
        "success": True,
        "message": "Password updated successfully."
    })


# ============================================================
# TEST EMAIL
# ============================================================

@app.route("/test-email")
def test_email():

    try:

        email = app.config.get(
            "MAIL_USERNAME"
        )


        if not email:

            return jsonify({
                "success": False,
                "message":
                    "MAIL_USERNAME is not configured."
            }), 500


        msg = Message(
            subject="Nail Nutrition Test Email",
            recipients=[email]
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
            "message":
                "Test email sent successfully."
        })


    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============================================================
# SAVE REPORT
# ============================================================

@app.route("/save-report", methods=["POST"])
def save_report():

    data = request.get_json(silent=True) or {}


    user_id = data.get("user_id")

    prediction = data.get(
        "prediction"
    )

    confidence = data.get(
        "confidence"
    )

    risk_level = data.get(
        "risk_level"
    )

    description = data.get(
        "description"
    )


    if (
        user_id is None
        or not prediction
        or confidence is None
        or not risk_level
    ):

        return jsonify({
            "success": False,
            "message":
                "Missing required report information."
        }), 400


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
        "message":
            "Report saved successfully."
    })


# ============================================================
# GET REPORTS
# ============================================================

@app.route(
    "/reports/<int:user_id>",
    methods=["GET"]
)
def get_reports(user_id):

    reports = (

        Report.query

        .filter_by(
            user_id=user_id
        )

        .order_by(
            Report.created_at.desc()
        )

        .all()

    )


    report_list = []


    for report in reports:

        report_list.append({

            "id":
                report.id,

            "prediction":
                report.prediction,

            "confidence":
                report.confidence,

            "risk_level":
                report.risk_level,

            "description":
                report.description,

            "created_at":
                report.created_at.strftime(
                    "%d-%m-%Y %H:%M"
                )

        })


    return jsonify(report_list)


# ============================================================
# DELETE REPORT
# ============================================================

@app.route(
    "/report/<int:report_id>",
    methods=["DELETE"]
)
def delete_report(report_id):

    report = db.session.get(
        Report,
        report_id
    )


    if not report:

        return jsonify({
            "success": False,
            "message":
                "Report not found."
        }), 404


    db.session.delete(report)

    db.session.commit()


    return jsonify({
        "success": True,
        "message":
            "Report deleted successfully."
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    filepath = None

    try:

        # --------------------------------------------
        # Validate image
        # --------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image uploaded."
            }), 400


        file = request.files["image"]


        if not file or not file.filename:

            return jsonify({
                "success": False,
                "error": "Invalid image file."
            }), 400


        # --------------------------------------------
        # Save temporarily
        # --------------------------------------------

        safe_filename = os.path.basename(
            file.filename
        )


        filepath = os.path.join(
            UPLOAD_FOLDER,
            safe_filename
        )


        file.save(filepath)


        # --------------------------------------------
        # Preprocess
        # --------------------------------------------

        image_array = preprocess_image(
            filepath
        )


        # --------------------------------------------
        # LAZY LOAD MODEL
        # --------------------------------------------

        model = get_model()


        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )


        predicted_index = int(
            np.argmax(
                predictions[0]
            )
        )


        prediction = CLASS_NAMES[
            predicted_index
        ]


        confidence = float(
            predictions[0][predicted_index]
            * 100
        )


        recommendation = (
            RECOMMENDATIONS[prediction]
        )


        # --------------------------------------------
        # Response
        # --------------------------------------------

        return jsonify({

            "success": True,

            "prediction":
                recommendation["title"],

            "prediction_class":
                prediction,

            "confidence":
                round(confidence, 2),

            "risk_level":
                recommendation["risk_level"],

            "description":
                recommendation["description"],

            "possible_indication":
                recommendation[
                    "possible_indication"
                ],

            "symptoms":
                recommendation["symptoms"],

            "foods":
                recommendation["foods"],

            "home_remedies":
                recommendation[
                    "home_remedies"
                ],

            "lifestyle":
                recommendation["lifestyle"],

            "doctor_advice":
                recommendation[
                    "doctor_advice"
                ],

            "disclaimer":
                (
                    "This application provides an "
                    "AI-assisted screening result based "
                    "on fingernail images. It is intended "
                    "for educational purposes only and "
                    "should not be considered a medical "
                    "diagnosis. Always consult a qualified "
                    "healthcare professional for proper "
                    "evaluation and treatment."
                )

        })


    except Exception as e:

        print(
            f"Prediction error: {e}"
        )


        return jsonify({
            "success": False,
            "error":
                "Unable to analyze the image.",
            "details":
                str(e)
        }), 500


    finally:

        # Remove uploaded image after inference
        if filepath and os.path.exists(filepath):

            try:
                os.remove(filepath)

            except OSError:
                pass


# ============================================================
# RUN LOCALLY
# ============================================================

# ---------------- RUN APP ---------------- #

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
