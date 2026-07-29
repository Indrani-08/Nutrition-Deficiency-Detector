# ============================================================
# Project    : Nail Nutrition
# Experiment : V3
# Model      : EfficientNetB3 + CLAHE + Bilateral Filtering
# ============================================================

import os
import cv2
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
    V3_IMAGE_SIZE,
    V3_MODEL_PATH,
    BATCH_SIZE,
    SEED,
    LEARNING_RATE,
    EPOCHS,
    CLASS_NAMES
)


# ============================================================
# SETUP
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 70)
print("V3 - EfficientNetB3 + CLAHE + Bilateral Filtering")
print("=" * 70)

print("\nTensorFlow Version :", tf.__version__)
print("Image Size         :", V3_IMAGE_SIZE)
print("Batch Size         :", BATCH_SIZE)
print("Learning Rate      :", LEARNING_RATE)
print("Maximum Epochs     :", EPOCHS)


# ============================================================
# IMAGE ENHANCEMENT
# ============================================================

def enhance_nail_image(image):
    """
    Apply controlled enhancement to one nail image.

    Pipeline:
        RGB
         ↓
        LAB colour space
         ↓
        CLAHE on luminance only
         ↓
        RGB
         ↓
        Mild bilateral filtering

    CLAHE is applied only to the lightness channel to avoid
    unnecessarily modifying nail colour information.
    """

    # TensorFlow image -> NumPy uint8

    image = np.asarray(image)

    image = np.clip(
        image,
        0,
        255
    ).astype(np.uint8)


    # --------------------------------------------------------
    # RGB -> LAB
    # --------------------------------------------------------

    lab_image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )


    # --------------------------------------------------------
    # Separate LAB channels
    # --------------------------------------------------------

    l_channel, a_channel, b_channel = cv2.split(
        lab_image
    )


    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )


    enhanced_l = clahe.apply(
        l_channel
    )


    # --------------------------------------------------------
    # Recombine channels
    # --------------------------------------------------------

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel
        )
    )


    # --------------------------------------------------------
    # LAB -> RGB
    # --------------------------------------------------------

    enhanced_rgb = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB
    )


    # --------------------------------------------------------
    # Mild bilateral filtering
    # --------------------------------------------------------

    enhanced_rgb = cv2.bilateralFilter(
        enhanced_rgb,
        d=5,
        sigmaColor=25,
        sigmaSpace=25
    )


    return enhanced_rgb.astype(np.uint8)


# ============================================================
# TENSORFLOW WRAPPER FOR ONE IMAGE
# ============================================================

def tensorflow_enhancement(image):

    enhanced = tf.numpy_function(
        func=enhance_nail_image,
        inp=[image],
        Tout=tf.uint8
    )

    enhanced.set_shape(
        V3_IMAGE_SIZE + (3,)
    )

    return enhanced


# ============================================================
# ENHANCE ONE BATCH
# ============================================================

def enhance_batch(images, labels):

    enhanced_images = tf.map_fn(
        tensorflow_enhancement,
        images,
        fn_output_signature=tf.TensorSpec(
            shape=V3_IMAGE_SIZE + (3,),
            dtype=tf.uint8
        )
    )

    enhanced_images = tf.cast(
        enhanced_images,
        tf.float32
    )

    return enhanced_images, labels


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASETS")
print("=" * 70)


train_dataset = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=V3_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True
)


validation_dataset = image_dataset_from_directory(
    VALIDATE_DIR,
    image_size=V3_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False
)


test_dataset = image_dataset_from_directory(
    TEST_DIR,
    image_size=V3_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# VERIFY CLASS ORDER
# ============================================================

detected_classes = train_dataset.class_names

print("\nDetected Classes:")
print(detected_classes)

print("\nConfigured Classes:")
print(CLASS_NAMES)


if detected_classes != CLASS_NAMES:

    raise ValueError(
        "\nCLASS_NAMES in config.py does not match "
        "the alphabetical folder ordering.\n"
        f"Detected : {detected_classes}\n"
        f"Config   : {CLASS_NAMES}"
    )


NUM_CLASSES = len(
    detected_classes
)


# ============================================================
# APPLY CLAHE + FILTERING
# ============================================================

print("\n" + "=" * 70)
print("APPLYING IMAGE ENHANCEMENT")
print("=" * 70)


train_dataset = train_dataset.map(
    enhance_batch,
    num_parallel_calls=tf.data.AUTOTUNE
)


validation_dataset = validation_dataset.map(
    enhance_batch,
    num_parallel_calls=tf.data.AUTOTUNE
)


test_dataset = test_dataset.map(
    enhance_batch,
    num_parallel_calls=tf.data.AUTOTUNE
)


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

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


# ============================================================
# PREVIEW ENHANCED IMAGES
# ============================================================

print("\nSaving enhancement preview...")


for images, labels in train_dataset.take(1):

    preview_count = min(
        4,
        images.shape[0]
    )

    fig, axes = plt.subplots(
        1,
        preview_count,
        figsize=(15, 4)
    )

    # If only one image somehow exists
    if preview_count == 1:
        axes = [axes]

    for i in range(preview_count):

        axes[i].imshow(
            images[i].numpy().astype(np.uint8)
        )

        class_index = int(
            labels[i].numpy()
        )

        axes[i].set_title(
            CLASS_NAMES[class_index]
        )

        axes[i].axis("off")


    plt.suptitle(
        "V3 Enhanced Nail Images"
    )

    plt.tight_layout()


    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "v3_enhancement_preview.png"
        )
    )


    plt.close()


print(
    "Enhancement preview saved to results/"
    "v3_enhancement_preview.png"
)


# ============================================================
# LOAD EfficientNetB3
# ============================================================

print("\n" + "=" * 70)
print("LOADING EfficientNetB3")
print("=" * 70)


base_model = EfficientNetB3(
    input_shape=V3_IMAGE_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)


print(
    "EfficientNetB3 loaded successfully."
)


# ============================================================
# FINE TUNING
# ============================================================

base_model.trainable = True


# Keep same fine-tuning strategy as V2
# so CLAHE is the main experimental change.

for layer in base_model.layers[:-40]:

    layer.trainable = False


trainable_layers = sum(
    1
    for layer in base_model.layers
    if layer.trainable
)


frozen_layers = sum(
    1
    for layer in base_model.layers
    if not layer.trainable
)


print(
    "\nFine tuning enabled."
)

print(
    "Trainable B3 layers:",
    trainable_layers
)

print(
    "Frozen B3 layers:",
    frozen_layers
)


# ============================================================
# BUILD MODEL
# ============================================================

print("\n" + "=" * 70)
print("BUILDING V3 MODEL")
print("=" * 70)


model = Sequential([

    Lambda(
        preprocess_input,
        input_shape=V3_IMAGE_SIZE + (3,),
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


# ============================================================
# COMPILE
# ============================================================

print("\n" + "=" * 70)
print("COMPILING MODEL")
print("=" * 70)


model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print(
    "Model compiled successfully."
)


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True,

    verbose=1
)


model_checkpoint = ModelCheckpoint(

    filepath=V3_MODEL_PATH,

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


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING V3")
print("=" * 70)


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


print(
    "\nV3 training completed."
)


# ============================================================
# LOAD BEST V3 MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING BEST V3 MODEL")
print("=" * 70)


best_model = load_model(

    V3_MODEL_PATH,

    custom_objects={
        "preprocess_input": preprocess_input
    }
)


print(
    "Best V3 model loaded successfully."
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("V3 TEST EVALUATION")
print("=" * 70)


test_loss, test_accuracy = best_model.evaluate(

    test_dataset,

    verbose=1
)


print("\nV3 TEST RESULTS")
print("=" * 70)


print(
    f"Test Loss     : {test_loss:.4f}"
)


print(
    f"Test Accuracy : {test_accuracy:.4f}"
)


print(
    f"Accuracy      : {test_accuracy * 100:.2f}%"
)


# ============================================================
# PREDICTIONS
# ============================================================

print(
    "\nGenerating V3 predictions..."
)


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


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    true_labels,

    predicted_labels,

    target_names=CLASS_NAMES,

    digits=4
)


print("\n" + "=" * 70)
print("V3 CLASSIFICATION REPORT")
print("=" * 70)


print(report)


classification_report_path = os.path.join(

    RESULTS_DIR,

    "v3_b3_clahe_classification_report.txt"
)


with open(
    classification_report_path,
    "w"
) as file:

    file.write(report)


print(
    "\nClassification report saved."
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

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

    "V3 EfficientNetB3 + CLAHE - Confusion Matrix"
)


plt.tight_layout()


confusion_matrix_path = os.path.join(

    RESULTS_DIR,

    "v3_b3_clahe_confusion_matrix.png"
)


plt.savefig(
    confusion_matrix_path
)


plt.close()


print(
    "Confusion matrix saved."
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

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

    "V3 EfficientNetB3 + CLAHE Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend()


plt.tight_layout()


accuracy_path = os.path.join(

    RESULTS_DIR,

    "v3_b3_clahe_accuracy.png"
)


plt.savefig(
    accuracy_path
)


plt.close()


# ============================================================
# LOSS GRAPH
# ============================================================

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

    "V3 EfficientNetB3 + CLAHE Loss"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Loss"
)


plt.legend()


plt.tight_layout()


loss_path = os.path.join(

    RESULTS_DIR,

    "v3_b3_clahe_loss.png"
)


plt.savefig(
    loss_path
)


plt.close()


# ============================================================
# SAVE EVALUATION RESULTS
# ============================================================

evaluation_path = os.path.join(

    RESULTS_DIR,

    "v3_b3_clahe_evaluation.txt"
)


with open(
    evaluation_path,
    "w"
) as file:

    file.write(
        "V3 EfficientNetB3 + CLAHE RESULTS\n"
    )

    file.write(
        "=" * 50 + "\n\n"
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


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("V3 EXPERIMENT COMPLETE")
print("=" * 70)


print(
    f"\nModel saved to:\n"
    f"{V3_MODEL_PATH}"
)


print(
    f"\nResults saved to:\n"
    f"{RESULTS_DIR}"
)


print("\nGenerated files:")

print(
    " - v3_enhancement_preview.png"
)

print(
    " - v3_b3_clahe_accuracy.png"
)

print(
    " - v3_b3_clahe_loss.png"
)

print(
    " - v3_b3_clahe_confusion_matrix.png"
)

print(
    " - v3_b3_clahe_classification_report.txt"
)

print(
    " - v3_b3_clahe_evaluation.txt"
)