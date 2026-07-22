import os
import shutil
import random

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SOURCE_DIR = os.path.join(BASE_DIR, "dataset", "augmented")

TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "validate")
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

# Split ratio
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# -----------------------------
# Split Dataset
# -----------------------------

for class_name in os.listdir(SOURCE_DIR):

    class_path = os.path.join(SOURCE_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(os.path.join(folder, class_name), exist_ok=True)

    # Train
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(TRAIN_DIR, class_name, img)
        )

    # Validation
    for img in val_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(VAL_DIR, class_name, img)
        )

    # Test
    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(TEST_DIR, class_name, img)
        )

    print(f"{class_name}")
    print(f"Train: {len(train_images)}")
    print(f"Validation: {len(val_images)}")
    print(f"Test: {len(test_images)}")
    print("-" * 30)

print("\nDataset Split Completed Successfully!")