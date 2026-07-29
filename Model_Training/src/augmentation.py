import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm

# ======================================================
# Project : Nail Nutrition
# Module  : Data Augmentation
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DATASET = os.path.join(BASE_DIR, "dataset", "processed")
AUGMENTED_DATASET = os.path.join(BASE_DIR, "dataset", "augmented")

os.makedirs(AUGMENTED_DATASET, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.25,
    shear_range=0.10,
    brightness_range=[0.9, 1.15],
    horizontal_flip=False,
    fill_mode="nearest"
)

AUGMENTATIONS_PER_IMAGE = 8

print("\nStarting Data Augmentation...\n")

for class_name in os.listdir(PROCESSED_DATASET):

    class_path = os.path.join(PROCESSED_DATASET, class_name)

    if not os.path.isdir(class_path):
        continue

    save_path = os.path.join(AUGMENTED_DATASET, class_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"\nProcessing: {class_name}")

    for image_name in tqdm(os.listdir(class_path)):

        image_path = os.path.join(class_path, image_name)

        if os.path.isdir(image_path):
            continue

        image = load_img(image_path)

        image = img_to_array(image)

        image = image.reshape((1,) + image.shape)

        # Save original image
        load_img(image_path).save(os.path.join(save_path, image_name))

        count = 0

        for batch in datagen.flow(
                image,
                batch_size=1,
                save_to_dir=save_path,
                save_prefix="aug",
                save_format="jpg"):

            count += 1

            if count >= AUGMENTATIONS_PER_IMAGE:
                break

print("\nData Augmentation Completed Successfully.")