# ============================================
# Project : Nail Nutrition
# Module  : Build MobileNetV2 Model
# ============================================

import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.models import Sequential

from config import (
    TRAIN_DIR,
    VALIDATE_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED
)

print("="*60)
print("Loading Dataset...")
print("="*60)

train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

validation_dataset = image_dataset_from_directory(
    VALIDATE_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
)

test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nDatasets Loaded Successfully!")

# Save class names before any dataset transformations
class_names = train_dataset.class_names

print("\nClass Names:")
print(class_names)

NUM_CLASSES = len(class_names)

print("\nNumber of Classes:", NUM_CLASSES)
