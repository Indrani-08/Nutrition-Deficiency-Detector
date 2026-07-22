from flask import Flask, request, jsonify
import os
import numpy as np

from utils.model_loader import model, CLASS_NAMES
from utils.preprocess import preprocess_image
from utils.recommendations import RECOMMENDATIONS

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return "Backend is running successfully!"


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

        "disclaimer":
            "This application provides an AI-assisted screening result based on fingernail images. It is intended for educational purposes only and should not be considered a medical diagnosis. Always consult a qualified healthcare professional for proper evaluation and treatment."

    })

if __name__ == "__main__":
    app.run(debug=True)