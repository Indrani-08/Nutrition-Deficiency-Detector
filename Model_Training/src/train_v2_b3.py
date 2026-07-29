# ============================================
# Project : Nail Nutrition
# Experiment : V2 - EfficientNetB3
# ============================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Lambda
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input

from config import (
    TRAIN_DIR,
    VALIDATE_DIR,
    TEST_DIR,
    RESULTS_DIR,
    MODEL_DIR,
    V2_IMAGE_SIZE,
    V2_MODEL_PATH,
    BATCH_SIZE,
    SEED,
    LEARNING_RATE,
    EPOCHS,
    CLASS_NAMES
)


# ============================================
# Setup
# ============================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("EXPERIMENT V2 - EfficientNetB3")
print("=" * 60)

print("\nTensorFlow Version:", tf.__version__)
print("Image Size:", V2_IMAGE_SIZE)
print("Batch Size:", BATCH_SIZE)
print("Learning Rate:", LEARNING_RATE)


# ============================================
# Load Dataset
# ============================================

print("\n" + "=" * 60)
print("LOADING DATASET")
print("=" * 60)

train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=V2_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)

validation_dataset = image_dataset_from_directory(
    VALIDATE_DIR,
    image_size=V2_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=V2_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================
# Verify Class Ordering
# ============================================

detected_classes = train_dataset.class_names

print("\nDetected Classes:")
print(detected_classes)

print("\nConfigured Classes:")
print(CLASS_NAMES)

if detected_classes != CLASS_NAMES:
    raise ValueError(
        "CLASS_NAMES does not match dataset class ordering."
    )

NUM_CLASSES = len(detected_classes)


# ============================================
# Dataset Optimization
# ============================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)


# ============================================
# Load EfficientNetB3
# ============================================

print("\n" + "=" * 60)
print("LOADING EfficientNetB3")
print("=" * 60)

base_model = EfficientNetB3(
    input_shape=V2_IMAGE_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

print("EfficientNetB3 loaded successfully.")


# ============================================
# Fine Tuning
# ============================================

base_model.trainable = True

# Same strategy as your original B0 model:
# freeze everything except approximately
# the final 40 layers.

for layer in base_model.layers[:-40]:
    layer.trainable = False

print("\nFine tuning enabled.")

trainable_layers = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

frozen_layers = sum(
    1 for layer in base_model.layers
    if not layer.trainable
)

print("Trainable B3 layers:", trainable_layers)
print("Frozen B3 layers:", frozen_layers)


# ============================================
# Build Model
# ============================================

print("\n" + "=" * 60)
print("BUILDING MODEL")
print("=" * 60)

model = Sequential([

    Lambda(
        preprocess_input,
        input_shape=V2_IMAGE_SIZE + (3,),
        name="efficientnet_preprocessing"
    ),

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

model.summary()


# ============================================
# Compile
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

print("Model compiled successfully.")


# ============================================
# Callbacks
# ============================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)


model_checkpoint = ModelCheckpoint(
    filepath=V2_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)


# ============================================
# Train
# ============================================

print("\n" + "=" * 60)
print("TRAINING EfficientNetB3")
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


print("\nTraining completed.")


# ============================================
# Load Best Model
# ============================================

print("\n" + "=" * 60)
print("LOADING BEST V2 MODEL")
print("=" * 60)

best_model = load_model(
    V2_MODEL_PATH,

    custom_objects={
        "preprocess_input": preprocess_input
    }
)

print("Best V2 model loaded.")


# ============================================
# Test Evaluation
# ============================================

print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)

test_loss, test_accuracy = best_model.evaluate(
    test_dataset,
    verbose=1
)

print("\nV2 TEST RESULTS")
print("=" * 60)

print(
    f"Test Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Accuracy      : {test_accuracy * 100:.2f}%"
)


# ============================================
# Predictions
# ============================================

print("\nGenerating predictions...")

predictions = best_model.predict(
    test_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


true_labels = np.concatenate([
    labels.numpy()
    for _, labels in test_dataset
])


# ============================================
# Classification Report
# ============================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    digits=4
)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "v2_b3_classification_report.txt"
)

with open(report_path, "w") as file:
    file.write(report)


# ============================================
# Confusion Matrix
# ============================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)


fig, ax = plt.subplots(
    figsize=(9, 9)
)


disp.plot(
    ax=ax,
    values_format="d"
)


plt.xticks(
    rotation=35,
    ha="right"
)


plt.title(
    "V2 EfficientNetB3 - Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "v2_b3_confusion_matrix.png"
    )
)


plt.close()


# ============================================
# Accuracy Graph
# ============================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)


plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)


plt.title(
    "V2 EfficientNetB3 Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "v2_b3_accuracy.png"
    )
)


plt.close()


# ============================================
# Loss Graph
# ============================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    history.history["loss"],
    label="Training Loss"
)


plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)


plt.title(
    "V2 EfficientNetB3 Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.tight_layout()


plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "v2_b3_loss.png"
    )
)


plt.close()


# ============================================
# Save Evaluation
# ============================================

evaluation_path = os.path.join(
    RESULTS_DIR,
    "v2_b3_evaluation.txt"
)


with open(evaluation_path, "w") as file:

    file.write(
        "V2 EfficientNetB3 RESULTS\n"
    )

    file.write(
        "=" * 40 + "\n\n"
    )

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    file.write(
        f"Test Accuracy: {test_accuracy:.4f}\n"
    )

    file.write(
        f"Test Accuracy Percentage: "
        f"{test_accuracy * 100:.2f}%\n"
    )


print("\n" + "=" * 60)
print("V2 EXPERIMENT COMPLETE")
print("=" * 60)


print(
    f"\nModel:\n{V2_MODEL_PATH}"
)

print(
    f"\nResults:\n{RESULTS_DIR}"
)