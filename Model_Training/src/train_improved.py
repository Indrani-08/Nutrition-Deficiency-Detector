import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomTranslation,
    RandomContrast,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    Lambda
)

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.utils import image_dataset_from_directory

from config import (
    TRAIN_DIR,
    VALIDATE_DIR,
    TEST_DIR,
    RESULTS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    SEED,
    LEARNING_RATE,
    EPOCHS,
    CLASS_NAMES,
    IMPROVED_MODEL_PATH
)


# ============================================================
# Setup
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(IMPROVED_MODEL_PATH), exist_ok=True)

print("=" * 60)
print("IMPROVED MODEL V1")
print("=" * 60)


# ============================================================
# Load datasets
# ============================================================

train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)

validation_dataset = image_dataset_from_directory(
    VALIDATE_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)

test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# IMPORTANT: verify folder ordering
detected_classes = train_dataset.class_names

print("\nDetected classes:")
print(detected_classes)

print("\nConfigured classes:")
print(CLASS_NAMES)

if detected_classes != CLASS_NAMES:
    raise ValueError(
        "CLASS_NAMES in config.py does not match the dataset "
        "folder ordering."
    )


# ============================================================
# Optimize dataset pipeline
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)


# ============================================================
# Data augmentation
# ============================================================

data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),

        RandomRotation(0.08),

        RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10)
        ),

        RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        ),

        RandomContrast(0.10)
    ],
    name="data_augmentation"
)


# ============================================================
# EfficientNetB0
# ============================================================

base_model = EfficientNetB0(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)

# Freeze everything initially
base_model.trainable = False


# ============================================================
# Classification model
# ============================================================

model = Sequential(
    [
        data_augmentation,

        Lambda(
            preprocess_input,
            name="efficientnet_preprocessing"
        ),

        base_model,

        GlobalAveragePooling2D(),

        BatchNormalization(),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(0.40),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.25),

        Dense(
            len(CLASS_NAMES),
            activation="softmax"
        )
    ]
)


# Build model before summary
model.build(
    input_shape=(None,) + IMAGE_SIZE + (3,)
)

print("\nMODEL SUMMARY")
model.summary()


# ============================================================
# Phase 1: Train classification head
# ============================================================

print("\n" + "=" * 60)
print("PHASE 1 — TRAINING CLASSIFICATION HEAD")
print("=" * 60)

model.compile(
    optimizer=AdamW(
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


head_early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


history_head = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    callbacks=[head_early_stop],
    verbose=1
)


# ============================================================
# Phase 2: Fine-tune EfficientNet
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 — FINE TUNING")
print("=" * 60)

base_model.trainable = True


# Freeze everything except the final 40 layers
for layer in base_model.layers[:-40]:
    layer.trainable = False


# Keep BatchNormalization layers frozen
#
# This is useful with a small dataset because updating BN
# statistics during fine-tuning can destabilize the pretrained
# representation.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False


trainable_count = sum(
    1 for layer in base_model.layers if layer.trainable
)

print(
    f"Trainable EfficientNet layers: {trainable_count}"
)


# Fine-tuning uses a LOWER learning rate
fine_tune_lr = LEARNING_RATE / 10


model.compile(
    optimizer=AdamW(
        learning_rate=fine_tune_lr,
        weight_decay=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


checkpoint = ModelCheckpoint(
    IMPROVED_MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)


early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)


remaining_epochs = max(EPOCHS - 10, 1)

history_fine = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=remaining_epochs,
    callbacks=[
        checkpoint,
        early_stop,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# Load best improved model
# ============================================================

print("\nLoading best Improved V1 model...")

best_model = tf.keras.models.load_model(
    IMPROVED_MODEL_PATH,
    custom_objects={
        "preprocess_input": preprocess_input
    }
)


# ============================================================
# Evaluate
# ============================================================

print("\n" + "=" * 60)
print("TEST EVALUATION")
print("=" * 60)

test_loss, test_accuracy = best_model.evaluate(
    test_dataset,
    verbose=1
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")
print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# Predictions
# ============================================================

probabilities = best_model.predict(
    test_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    probabilities,
    axis=1
)

true_labels = np.concatenate(
    [
        labels.numpy()
        for _, labels in test_dataset
    ]
)


# ============================================================
# Classification report
# ============================================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=CLASS_NAMES,
    digits=4
)

print("\nCLASSIFICATION REPORT")
print("=" * 60)
print(report)


report_path = os.path.join(
    RESULTS_DIR,
    "improved_v1_classification_report.txt"
)

with open(report_path, "w") as file:
    file.write(report)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(9, 9))

disp.plot(
    ax=ax,
    values_format="d"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.title(
    "Improved Model V1 — Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "improved_v1_confusion_matrix.png"
    )
)

plt.close()


# ============================================================
# Combine histories
# ============================================================

training_accuracy = (
    history_head.history["accuracy"]
    + history_fine.history["accuracy"]
)

validation_accuracy = (
    history_head.history["val_accuracy"]
    + history_fine.history["val_accuracy"]
)

training_loss = (
    history_head.history["loss"]
    + history_fine.history["loss"]
)

validation_loss = (
    history_head.history["val_loss"]
    + history_fine.history["val_loss"]
)


# ============================================================
# Accuracy graph
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    training_accuracy,
    label="Training Accuracy"
)

plt.plot(
    validation_accuracy,
    label="Validation Accuracy"
)

plt.axvline(
    x=len(history_head.history["accuracy"]) - 1,
    linestyle="--",
    label="Fine Tuning Begins"
)

plt.title(
    "Improved V1 Training Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "improved_v1_accuracy.png"
    )
)

plt.close()


# ============================================================
# Loss graph
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    training_loss,
    label="Training Loss"
)

plt.plot(
    validation_loss,
    label="Validation Loss"
)

plt.axvline(
    x=len(history_head.history["loss"]) - 1,
    linestyle="--",
    label="Fine Tuning Begins"
)

plt.title(
    "Improved V1 Training Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "improved_v1_loss.png"
    )
)

plt.close()


# ============================================================
# Save evaluation
# ============================================================

evaluation_path = os.path.join(
    RESULTS_DIR,
    "improved_v1_evaluation.txt"
)

with open(evaluation_path, "w") as file:

    file.write(
        "IMPROVED MODEL V1\n"
    )

    file.write(
        "=" * 40 + "\n\n"
    )

    file.write(
        f"Test Loss: {test_loss:.4f}\n"
    )

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy:.4f}\n"
    )

    file.write(
        f"Test Accuracy Percentage: "
        f"{test_accuracy * 100:.2f}%\n"
    )


print("\n" + "=" * 60)
print("IMPROVED V1 COMPLETE")
print("=" * 60)

print(
    f"Model saved to:\n{IMPROVED_MODEL_PATH}"
)

print(
    f"\nResults saved to:\n{RESULTS_DIR}"
)