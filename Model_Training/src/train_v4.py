# ============================================================
# Project : Nail Nutrition
# Model   : V4 Clean EfficientNetB3
#
# Pipeline:
# Clean dataset
#       ↓
# Moderate augmentation (TRAIN ONLY)
#       ↓
# EfficientNetB3 ImageNet
#       ↓
# Stage 1 - Frozen base model
#       ↓
# Stage 2 - Fine tuning
#       ↓
# Evaluation
#       ↓
# Confusion Matrix + Classification Report
#       ↓
# Final Model
# ============================================================

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.layers import (
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    RandomFlip,
    RandomRotation,
    RandomZoom,
    RandomContrast,
    RandomTranslation
)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from tensorflow.keras.utils import image_dataset_from_directory

from tensorflow.keras.applications import EfficientNetB3


# ============================================================
# Configuration
# ============================================================

SEED = 42

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 8

STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 20

STAGE1_LR = 1e-3
STAGE2_LR = 1e-5

NUM_CLASSES = 4


# ============================================================
# Reproducibility
# ============================================================

tf.keras.utils.set_random_seed(SEED)


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "clean"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

VALIDATE_DIR = os.path.join(
    DATASET_DIR,
    "validate"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "v4"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# Model Paths
# ============================================================

STAGE1_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "v4_stage1.keras"
)

BEST_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "v4_best.keras"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "v4_clean_efficientnetb3.keras"
)


# ============================================================
# Verify Dataset
# ============================================================

print("=" * 70)
print("V4 CLEAN EFFICIENTNETB3")
print("=" * 70)

print("\nDataset directory:")
print(DATASET_DIR)

for path in [
    TRAIN_DIR,
    VALIDATE_DIR,
    TEST_DIR
]:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Dataset directory not found:\n{path}"
        )


# ============================================================
# Load Dataset
# ============================================================

print("\n" + "=" * 70)
print("LOADING CLEAN DATASET")
print("=" * 70)

train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

validation_dataset = image_dataset_from_directory(
    VALIDATE_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# Class Names
# ============================================================

class_names = train_dataset.class_names

print("\nClasses:")

for index, class_name in enumerate(class_names):

    print(
        f"{index} -> {class_name}"
    )

if len(class_names) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}."
    )


# ============================================================
# Save Class Names
# ============================================================

class_names_path = os.path.join(
    RESULTS_DIR,
    "class_names.json"
)

with open(
    class_names_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


# ============================================================
# Performance Optimization
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

# We deliberately don't cache the training dataset here.
# Dataset is small, but this keeps the input pipeline simple
# while augmentation remains random during each epoch.

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# Data Augmentation
# ============================================================

print("\n" + "=" * 70)
print("CREATING DATA AUGMENTATION")
print("=" * 70)

data_augmentation = tf.keras.Sequential(
    [

        # Horizontal flip is reasonable for nail orientation.
        RandomFlip(
            "horizontal"
        ),

        # Small rotations only.
        RandomRotation(
            0.05
        ),

        # Small zoom changes.
        RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10)
        ),

        # Mild contrast variation to simulate
        # different camera/lighting conditions.
        RandomContrast(
            0.10
        ),

        # Small translations.
        RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        )
    ],
    name="nail_augmentation"
)

print("Moderate augmentation created.")


# ============================================================
# Build EfficientNetB3
# ============================================================

print("\n" + "=" * 70)
print("BUILDING EFFICIENTNETB3 MODEL")
print("=" * 70)

base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    )
)


# ============================================================
# Stage 1 - Freeze Base Model
# ============================================================

base_model.trainable = False


# ============================================================
# Model Architecture
# ============================================================

inputs = Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3
    ),
    name="input_image"
)

x = data_augmentation(
    inputs
)

# IMPORTANT:
# EfficientNet models in modern tf.keras include their
# expected input rescaling internally.
#
# Therefore we DO NOT add:
#
# x = x / 255.0
#
# and do not call preprocess_input here.

x = base_model(
    x,
    training=False
)

x = GlobalAveragePooling2D(
    name="global_average_pool"
)(x)

x = Dense(
    256,
    activation="relu",
    name="dense_256"
)(x)

x = Dropout(
    0.4,
    name="dropout_1"
)(x)

x = Dense(
    128,
    activation="relu",
    name="dense_128"
)(x)

x = Dropout(
    0.25,
    name="dropout_2"
)(x)

outputs = Dense(
    NUM_CLASSES,
    activation="softmax",
    name="predictions"
)(x)

model = Model(
    inputs,
    outputs,
    name="V4_Clean_EfficientNetB3"
)


# ============================================================
# Model Summary
# ============================================================

print("\n" + "=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()


# ============================================================
# Stage 1 Compile
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 - FEATURE EXTRACTION")
print("=" * 70)

model.compile(

    optimizer=Adam(
        learning_rate=STAGE1_LR
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# Stage 1 Callbacks
# ============================================================

stage1_callbacks = [

    ModelCheckpoint(
        STAGE1_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ============================================================
# Stage 1 Training
# ============================================================

history_stage1 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=STAGE1_EPOCHS,

    callbacks=stage1_callbacks,

    verbose=1
)


print("\nStage 1 completed.")


# ============================================================
# Load Best Stage 1 Model
# ============================================================

print("\nLoading best Stage 1 model...")

model = load_model(
    STAGE1_MODEL_PATH
)

print("Best Stage 1 model loaded.")


# ============================================================
# Find EfficientNet Base Model Again
# ============================================================

base_model = None

for layer in model.layers:

    if isinstance(
        layer,
        tf.keras.Model
    ) and "efficientnet" in layer.name.lower():

        base_model = layer
        break


if base_model is None:

    raise ValueError(
        "EfficientNetB3 base model could not be found."
    )


# ============================================================
# Stage 2 - Fine Tuning
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 - FINE TUNING")
print("=" * 70)

base_model.trainable = True


# ============================================================
# Freeze Most Layers
# ============================================================

# Fine tune approximately the final 40 layers.
# Earlier layers keep their pretrained features.

fine_tune_at = max(
    0,
    len(base_model.layers) - 40
)

for layer in base_model.layers[:fine_tune_at]:

    layer.trainable = False


# ============================================================
# Keep Batch Normalization Frozen
# ============================================================

# Important for small datasets.
# Updating BN statistics with very small batches can hurt
# pretrained representations.

for layer in base_model.layers:

    if isinstance(
        layer,
        tf.keras.layers.BatchNormalization
    ):

        layer.trainable = False


trainable_layers = sum(
    1
    for layer in base_model.layers
    if layer.trainable
)

print(
    f"EfficientNet layers: {len(base_model.layers)}"
)

print(
    f"Trainable EfficientNet layers: {trainable_layers}"
)


# ============================================================
# Stage 2 Compile
# ============================================================

# MUST recompile after changing trainable layers.

model.compile(

    optimizer=Adam(
        learning_rate=STAGE2_LR
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# Stage 2 Callbacks
# ============================================================

stage2_callbacks = [

    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]


# ============================================================
# Stage 2 Training
# ============================================================

history_stage2 = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=STAGE2_EPOCHS,

    callbacks=stage2_callbacks,

    verbose=1
)


print("\nStage 2 completed.")


# ============================================================
# Load Best Fine-Tuned Model
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST V4 MODEL")
print("=" * 70)

best_model = load_model(
    BEST_MODEL_PATH
)

print("Best V4 model loaded.")


# ============================================================
# Combine Training Histories
# ============================================================

training_accuracy = (
    history_stage1.history["accuracy"]
    +
    history_stage2.history["accuracy"]
)

validation_accuracy = (
    history_stage1.history["val_accuracy"]
    +
    history_stage2.history["val_accuracy"]
)

training_loss = (
    history_stage1.history["loss"]
    +
    history_stage2.history["loss"]
)

validation_loss = (
    history_stage1.history["val_loss"]
    +
    history_stage2.history["val_loss"]
)

stage1_length = len(
    history_stage1.history["accuracy"]
)


# ============================================================
# Accuracy Graph
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    training_accuracy,
    label="Training Accuracy"
)

plt.plot(
    validation_accuracy,
    label="Validation Accuracy"
)

plt.axvline(
    x=stage1_length - 1,
    linestyle="--",
    label="Fine Tuning Begins"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "V4 Clean EfficientNetB3 Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Loss Graph
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    training_loss,
    label="Training Loss"
)

plt.plot(
    validation_loss,
    label="Validation Loss"
)

plt.axvline(
    x=stage1_length - 1,
    linestyle="--",
    label="Fine Tuning Begins"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "V4 Clean EfficientNetB3 Loss"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "loss.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Evaluate Test Dataset
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(
    test_dataset,
    verbose=1
)

print(
    f"\nTest Loss     : {test_loss:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy:.4f}"
)

print(
    f"Test Accuracy : {test_accuracy * 100:.2f}%"
)


# ============================================================
# Save Evaluation Results
# ============================================================

evaluation_path = os.path.join(
    RESULTS_DIR,
    "evaluation_results.txt"
)

with open(
    evaluation_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "V4 CLEAN EFFICIENTNETB3\n"
    )

    file.write("=" * 50 + "\n\n")

    file.write(
        f"Test Loss: {test_loss:.6f}\n"
    )

    file.write(
        f"Test Accuracy: {test_accuracy:.6f}\n"
    )

    file.write(
        f"Test Accuracy Percentage: "
        f"{test_accuracy * 100:.2f}%\n"
    )


# ============================================================
# Predictions
# ============================================================

print("\nGenerating predictions...")

predictions = best_model.predict(
    test_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# True Labels
# ============================================================

true_labels = np.concatenate(
    [
        labels.numpy()
        for images, labels in test_dataset
    ]
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# Plot Confusion Matrix
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

disp.plot(
    ax=ax,
    values_format="d"
)

plt.title(
    "V4 Clean EfficientNetB3 - Confusion Matrix"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# Classification Report
# ============================================================

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    digits=4,
    zero_division=0
)

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# Save Classification Report
# ============================================================

report_path = os.path.join(
    RESULTS_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "V4 CLEAN EFFICIENTNETB3\n"
    )

    file.write("=" * 70 + "\n\n")

    file.write(report)


# ============================================================
# Save Final Model
# ============================================================

best_model.save(
    FINAL_MODEL_PATH
)

print("\n" + "=" * 70)
print("V4 TRAINING COMPLETE")
print("=" * 70)

print("\nFinal model:")

print(
    FINAL_MODEL_PATH
)

print("\nResults:")

print(
    RESULTS_DIR
)