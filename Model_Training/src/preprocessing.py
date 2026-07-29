import os
import cv2
import numpy as np
from tqdm import tqdm
from segmentation import denoise, segment_nail
from illumination import illumination_pipeline

# ======================================================
# Project : Nail Nutrition
# Module  : Image Preprocessing
# ======================================================

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
RAW_DATASET = os.path.join(BASE_DIR, "dataset", "raw")
PROCESSED_DATASET = os.path.join(BASE_DIR, "dataset", "processed")

# Image size
IMAGE_SIZE = (224, 224)

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_DATASET, exist_ok=True)


def preprocess_image(image):
    """
    Complete preprocessing pipeline
    """

    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)

    # Step 1: Remove camera noise
    image = denoise(image)

    # Step 2: Segment the nail
    image = segment_nail(image)

    # Step 3: Resize again because segmentation crops the image
    image = cv2.resize(image, IMAGE_SIZE)

    # Step 4: Correct illumination
    image = illumination_pipeline(image)

    # Step 5: Normalize
    image = image.astype(np.float32) / 255.0

    return image


print("\n========================================")
print("Starting Image Preprocessing...")
print("========================================\n")

# Loop through each class
for class_name in os.listdir(RAW_DATASET):

    class_path = os.path.join(RAW_DATASET, class_name)

    # Skip non-folder files
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_name}")

    # Create output folder
    save_path = os.path.join(PROCESSED_DATASET, class_name)
    os.makedirs(save_path, exist_ok=True)

    # Process every image
    for image_name in tqdm(os.listdir(class_path)):

        image_path = os.path.join(class_path, image_name)

        # Skip folders accidentally placed inside
        if os.path.isdir(image_path):
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping unreadable image: {image_name}")
            continue

        processed_image = preprocess_image(image)

        # Convert back to uint8 for saving
        processed_image = (processed_image * 255).astype(np.uint8)

        save_image_path = os.path.join(save_path, image_name)

        cv2.imwrite(save_image_path, processed_image)

print("\n========================================")
print(" Image Preprocessing Completed!")
print(" Processed images saved in:")
print(PROCESSED_DATASET)
print("========================================")