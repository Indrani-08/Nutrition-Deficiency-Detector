# ============================================
# Project : Nail Nutrition
# Module  : Build & Train MobileNetV2 Model
# ============================================

import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Lambda
)

from tensorflow.keras.models import load_model

from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import (
    TRAIN_DIR,
    VALIDATE_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    LEARNING_RATE,
    EPOCHS
)

# ============================================
# Create Required Directories
# ============================================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# ============================================
# Load Dataset
# ============================================

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

# ============================================
# Dataset Information
# ============================================

print("\nDatasets Loaded Successfully!")

class_names = train_dataset.class_names

print("\nClass Names:")
print(class_names)

NUM_CLASSES = len(class_names)

print("\nNumber of Classes:", NUM_CLASSES)

# ============================================
# Performance Optimization
# ============================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ============================================
# Load EfficientNetB0
# ============================================

print("\nLoading EfficientNetB0...")

base_model = EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

print("EfficientNetB0 Loaded Successfully!")



# ============================================
# Fine Tuning
# ============================================

base_model.trainable = True

for layer in base_model.layers[:-40]:
    layer.trainable = False

print("Fine Tuning Enabled!")

# ============================================
# Build Model
# ============================================

print("\nBuilding Classification Model...")

model = Sequential([

    Lambda(preprocess_input),

    base_model,

    GlobalAveragePooling2D(),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.5),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(0.3),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])

print("Classification Model Created Successfully!")

print("\n"+"="*60)
print("MODEL SUMMARY")
print("="*60)

model.summary()
# ============================================
# Compile Model
# ============================================

print("\n" + "=" * 60)
print("COMPILING MODEL")
print("=" * 60)

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),
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
    patience=7,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    filepath="models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
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
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint,
        reduce_lr
    ],
    verbose=1
)

print("\nTraining Completed Successfully!")

# ============================================
# Load Best Model
# ============================================

print("\n" + "=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

best_model = load_model(
    "models/best_model.keras",
    custom_objects={
        "preprocess_input": preprocess_input
    }
)

print("Best Model Loaded Successfully!")
# ============================================
# Evaluate Best Model
# ============================================

print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

test_loss, test_accuracy = best_model.evaluate(
    test_dataset,
    verbose=1
)

print("\nMODEL EVALUATION RESULTS")
print("=" * 60)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")

# ============================================
# Save Evaluation Results
# ============================================

with open("results/evaluation_results.txt", "w") as file:

    file.write("MODEL EVALUATION RESULTS\n")
    file.write("=" * 40 + "\n\n")
    file.write(f"Test Loss     : {test_loss:.4f}\n")
    file.write(f"Test Accuracy : {test_accuracy:.4f}\n")

print("Evaluation Results Saved Successfully!")

# ============================================
# Accuracy Graph
# ============================================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("results/accuracy.png")
plt.close()

print("Accuracy Graph Saved!")

# ============================================
# Loss Graph
# ============================================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("results/loss.png")
plt.close()

print("Loss Graph Saved!")

# ============================================
# Predictions
# ============================================

print("\nGenerating Predictions...")

predictions = best_model.predict(test_dataset)

predicted_labels = np.argmax(predictions, axis=1)

true_labels = np.concatenate([
    y.numpy() for x, y in test_dataset
])

# ============================================
# Confusion Matrix
# ============================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8,8))

disp.plot(ax=ax)

plt.xticks(rotation=45)

plt.savefig("results/confusion_matrix.png")

plt.close()

print("Confusion Matrix Saved!")

# ============================================
# Classification Report
# ============================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names
)

with open(
    "results/classification_report.txt",
    "w"
) as file:

    file.write(report)

print("Classification Report Saved!")
best_model.save("models/final_nail_model.keras")
print("Final Model Saved Successfully!")