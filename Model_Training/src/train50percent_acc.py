# ============================================
# Project : Nail Nutrition
# Module  : Build & Train MobileNetV2 Model
# ============================================

import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

from config import (
    TRAIN_DIR,
    VALIDATE_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED
)

# ============================================
# Load Dataset
# ============================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

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

print("\nClass Names:")
print(train_dataset.class_names)

NUM_CLASSES = len(train_dataset.class_names)

print("\nNumber of Classes:", NUM_CLASSES)

# ============================================
# Load MobileNetV2
# ============================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

print("MobileNetV2 Loaded Successfully!")

# ============================================
# Freeze Base Model
# ============================================

base_model.trainable = False

print("Base Model Frozen Successfully!")

# ============================================
# Build Classification Model
# ============================================

print("\nBuilding Classification Model...")

model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

print("Classification Model Created Successfully!")

# ============================================
# Display Model Summary
# ============================================

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()

# ============================================
# Compile Model
# ============================================

print("\n" + "=" * 60)
print("COMPILING MODEL")
print("=" * 60)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model Compiled Successfully!")

# ============================================
# Create Callbacks
# ============================================

print("\n" + "=" * 60)
print("CREATING CALLBACKS")
print("=" * 60)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath="models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    verbose=1
)

print("Callbacks Created Successfully!")

# ============================================
# Train Model
# ============================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,      # Keep 5 for testing. Later change to 20.
    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_lr
    ]
)

print("\nTraining Completed Successfully!")
# ============================================
# Load Best Model
# ============================================

print("\n" + "=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

best_model = load_model("models/best_model.keras")

print("Best Model Loaded Successfully!")

# ============================================
# Save Final Model
# ============================================

model.save("models/final_model.keras")

print("\nFinal Model Saved Successfully!")

print("\nProject Module Completed Successfully!")