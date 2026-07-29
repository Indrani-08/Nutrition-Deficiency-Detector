import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

from config import IMAGE_SIZE
CLASS_NAMES = [
    "Healthy",
    "Iron Deficiency",
    "Vitamin B12 Deficiency",
    "Vitamin D Deficiency"
]
print("Loading trained model...")

model = load_model(
    "models/final_nail_model.keras",
    compile=False,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

print("Model loaded successfully!")
def preprocess_image(image_path):

    img = image.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array
def predict(image_path):

    processed_image = preprocess_image(image_path)

    predictions = model.predict(processed_image, verbose=0)

    predicted_index = np.argmax(predictions)

    confidence = np.max(predictions) * 100

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence
image_path = input("Enter the path to the image file: ").strip()

if not os.path.exists(image_path):
    print("❌ Image not found!")
    exit()
img = image.load_img(image_path)

plt.imshow(img)

plt.axis("off")

plt.show()
predicted_class, confidence = predict(image_path)

print("\nPrediction Result")
print("=" * 40)

print("Predicted Class :", predicted_class)
print(f"Confidence      : {confidence:.2f}%")