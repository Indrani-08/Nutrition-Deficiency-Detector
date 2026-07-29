import os

# ============================================
# Project Paths
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALIDATE_DIR = os.path.join(DATASET_DIR, "validate")
TEST_DIR = os.path.join(DATASET_DIR, "test")

PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
AUGMENTED_DIR = os.path.join(DATASET_DIR, "augmented")

# ============================================
# Model Paths
# ============================================

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")

IMPROVED_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "improved_model_v1.keras"
)
# ============================================
# Improved V2 - EfficientNetB3
# ============================================

V2_IMAGE_SIZE = (300, 300)

V2_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "improved_model_v2_b3.keras"
)
# ============================================
# V3 - EfficientNetB3 + CLAHE
# ============================================

V3_IMAGE_SIZE = (300, 300)

V3_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "improved_model_v3_b3_clahe.keras"
)
# ============================================
# Results Path
# ============================================

RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ============================================
# Image Settings
# ============================================

IMAGE_SIZE = (224, 224)

CHANNELS = 3

NUM_CLASSES = 4

# ============================================
# Training Parameters
# ============================================

BATCH_SIZE = 16

EPOCHS = 40

LEARNING_RATE = 0.0001

SEED = 42

# ============================================
# Class Names
# ============================================

CLASS_NAMES = [
    "healthy_nails",
    "iron_deficiency",
    "vitamin_b12_deficiency",
    "vitamin_d_deficiency"
]