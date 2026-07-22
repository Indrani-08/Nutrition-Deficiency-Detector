# ============================================
# Project : Nail Nutrition
# Module  : Evaluate Trained Model
# ============================================

import os
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import image_dataset_from_directory

from config import (
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE
)

# ============================================
# Create Results Folder
# ============================================

os.makedirs("results", exist_ok=True)

# ============================================
# Load Test Dataset
# ============================================

print("=" * 60)
print("Loading Test Dataset...")
print("=" * 60)

test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTest Dataset Loaded Successfully!")

# ============================================
# Load Best Model
# ============================================

print("\n" + "=" * 60)
print("Loading Best Model...")
print("=" * 60)

model = load_model(
   "models/final_nail_model.keras",
    compile=False,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

print("Best Model Loaded Successfully!")

# ============================================
# Evaluate Model
# ============================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Best Model Loaded Successfully!")
print("\n" + "=" * 60)
print("Evaluating Model...")
print("=" * 60)

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

# ============================================
# Display Results
# ============================================

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")

# ============================================
# Save Results
# ============================================

with open("results/evaluation_results.txt", "w") as file:

    file.write("=========================================\n")
    file.write("MODEL EVALUATION RESULTS\n")
    file.write("=========================================\n\n")

    file.write(f"Test Loss     : {test_loss:.4f}\n")
    file.write(f"Test Accuracy : {test_accuracy:.4f}\n")

print("\nEvaluation Results Saved Successfully!")

print("\nModule 15 Completed Successfully!")