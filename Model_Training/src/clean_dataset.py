# ============================================
# Project : Nail Nutrition
# Module  : Dataset Cleaning Pipeline
#
# Tasks:
# 1. Scan train/validate/test
# 2. Detect exact duplicate images
# 3. Detect conflicting labels
# 4. Remove duplicate groups
# 5. Remove poor-quality images conservatively
# 6. Create a clean master dataset
# 7. Create fresh train/validate/test splits
# 8. Generate cleanup reports
# ============================================

import os
import cv2
import shutil
import hashlib
import random
from collections import defaultdict, Counter

# ============================================
# Configuration
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")

ORIGINAL_SPLITS = [
    os.path.join(DATASET_DIR, "train"),
    os.path.join(DATASET_DIR, "validate"),
    os.path.join(DATASET_DIR, "test")
]

CLEAN_MASTER_DIR = os.path.join(DATASET_DIR, "clean_master")
CLEAN_SPLIT_DIR = os.path.join(DATASET_DIR, "clean")

RESULTS_DIR = os.path.join(BASE_DIR, "results")

CLASS_NAMES = [
    "healthy_nails",
    "iron_deficiency",
    "vitamin_b12_deficiency",
    "vitamin_d_deficiency"
]

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

SEED = 42

TRAIN_RATIO = 0.70
VALIDATE_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================
# Conservative Quality Thresholds
# ============================================

# We deliberately use strict thresholds here.
# We do NOT want to delete every bright image.

MIN_BLUR_SCORE = 25

MAX_BRIGHTNESS = 245

MIN_BRIGHTNESS = 20


# ============================================
# Create Directories
# ============================================

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================
# Image Hash
# ============================================

def calculate_hash(image_path):

    hasher = hashlib.md5()

    with open(image_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ============================================
# Image Quality Check
# ============================================

def check_image_quality(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False, "UNREADABLE"

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ------------------------------
    # Blur detection
    # ------------------------------

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # ------------------------------
    # Brightness
    # ------------------------------

    brightness = gray.mean()

    # ------------------------------
    # Conservative rejection
    # ------------------------------

    if blur_score < MIN_BLUR_SCORE:
        return False, "EXTREMELY_BLURRY"

    if brightness > MAX_BRIGHTNESS:
        return False, "EXTREMELY_OVEREXPOSED"

    if brightness < MIN_BRIGHTNESS:
        return False, "EXTREMELY_DARK"

    return True, "OK"


# ============================================
# Scan Dataset
# ============================================

print("=" * 70)
print("SCANNING ORIGINAL DATASET")
print("=" * 70)

images = []

for split_path in ORIGINAL_SPLITS:

    split_name = os.path.basename(split_path)

    for class_name in CLASS_NAMES:

        class_path = os.path.join(
            split_path,
            class_name
        )

        if not os.path.exists(class_path):

            print(
                f"WARNING: Directory not found: {class_path}"
            )

            continue

        for filename in os.listdir(class_path):

            if not filename.lower().endswith(
                VALID_EXTENSIONS
            ):
                continue

            full_path = os.path.join(
                class_path,
                filename
            )

            images.append({
                "path": full_path,
                "split": split_name,
                "class": class_name,
                "filename": filename
            })


print(f"\nTotal images found: {len(images)}")


# ============================================
# Calculate Exact Hashes
# ============================================

print("\n" + "=" * 70)
print("CHECKING EXACT DUPLICATES")
print("=" * 70)

hash_groups = defaultdict(list)

for item in images:

    try:

        image_hash = calculate_hash(
            item["path"]
        )

        item["hash"] = image_hash

        hash_groups[image_hash].append(item)

    except Exception as e:

        print(
            f"Hash error: {item['path']} -> {e}"
        )


# ============================================
# Analyse Duplicate Groups
# ============================================

duplicate_groups = []
conflicting_groups = []

for image_hash, group in hash_groups.items():

    if len(group) > 1:

        duplicate_groups.append(
            (image_hash, group)
        )

        labels = {
            item["class"]
            for item in group
        }

        if len(labels) > 1:

            conflicting_groups.append(
                (image_hash, group)
            )


print(
    f"Duplicate groups found: "
    f"{len(duplicate_groups)}"
)

print(
    f"Conflicting duplicate groups: "
    f"{len(conflicting_groups)}"
)


# ============================================
# Write Duplicate Report
# ============================================

duplicate_report_path = os.path.join(
    RESULTS_DIR,
    "cleanup_duplicates.txt"
)

with open(
    duplicate_report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "DATASET DUPLICATE CLEANUP REPORT\n"
    )

    report.write("=" * 70 + "\n\n")

    for number, (image_hash, group) in enumerate(
        duplicate_groups,
        start=1
    ):

        labels = {
            item["class"]
            for item in group
        }

        conflict = len(labels) > 1

        report.write(
            f"Duplicate Group {number}\n"
        )

        report.write(
            f"Hash: {image_hash}\n"
        )

        report.write(
            f"Conflicting labels: {conflict}\n"
        )

        for item in group:

            report.write(
                f"  {item['split']} / "
                f"{item['class']} / "
                f"{item['filename']}\n"
            )

        report.write("\n")


# ============================================
# Remove Duplicate Groups
# ============================================

print("\n" + "=" * 70)
print("REMOVING DUPLICATES")
print("=" * 70)

excluded_paths = set()

for image_hash, group in duplicate_groups:

    labels = {
        item["class"]
        for item in group
    }

    # ----------------------------------------
    # Conflicting duplicate
    # ----------------------------------------
    # Same image has different labels.
    # Remove ALL copies because we cannot
    # determine the correct label safely.
    # ----------------------------------------

    if len(labels) > 1:

        print("\nCONFLICT:")

        for item in group:

            excluded_paths.add(
                item["path"]
            )

            print(
                "Removed:",
                item["split"],
                "/",
                item["class"],
                "/",
                item["filename"]
            )

    # ----------------------------------------
    # Same-label duplicate
    # ----------------------------------------
    # Keep ONE copy only.
    # ----------------------------------------

    else:

        # Keep first image

        for item in group[1:]:

            excluded_paths.add(
                item["path"]
            )


# ============================================
# Quality Filtering
# ============================================

print("\n" + "=" * 70)
print("CHECKING IMAGE QUALITY")
print("=" * 70)

clean_images = []

quality_removed = []

for item in images:

    if item["path"] in excluded_paths:
        continue

    valid, reason = check_image_quality(
        item["path"]
    )

    if valid:

        clean_images.append(item)

    else:

        quality_removed.append(
            (item, reason)
        )

        print(
            f"Quality removed: "
            f"{item['filename']} "
            f"({reason})"
        )


# ============================================
# Quality Removal Report
# ============================================

quality_report_path = os.path.join(
    RESULTS_DIR,
    "quality_removed.txt"
)

with open(
    quality_report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "IMAGES REMOVED BY QUALITY FILTER\n"
    )

    report.write("=" * 70 + "\n\n")

    for item, reason in quality_removed:

        report.write(
            f"{item['split']} / "
            f"{item['class']} / "
            f"{item['filename']} "
            f"-> {reason}\n"
        )


# ============================================
# Rebuild Clean Master Dataset
# ============================================

print("\n" + "=" * 70)
print("CREATING CLEAN MASTER DATASET")
print("=" * 70)

if os.path.exists(CLEAN_MASTER_DIR):

    shutil.rmtree(
        CLEAN_MASTER_DIR
    )

for class_name in CLASS_NAMES:

    os.makedirs(
        os.path.join(
            CLEAN_MASTER_DIR,
            class_name
        ),
        exist_ok=True
    )


# ============================================
# Copy Clean Images
# ============================================

class_counter = defaultdict(int)

for item in clean_images:

    class_name = item["class"]

    class_counter[class_name] += 1

    extension = os.path.splitext(
        item["filename"]
    )[1]

    # Unique filename prevents collisions
    # when different splits contain files
    # called 000001.jpg etc.

    new_filename = (
        f"{class_counter[class_name]:05d}"
        f"{extension.lower()}"
    )

    destination = os.path.join(
        CLEAN_MASTER_DIR,
        class_name,
        new_filename
    )

    shutil.copy2(
        item["path"],
        destination
    )


# ============================================
# Master Dataset Statistics
# ============================================

print("\nClean master dataset:")

for class_name in CLASS_NAMES:

    class_dir = os.path.join(
        CLEAN_MASTER_DIR,
        class_name
    )

    count = len([
        filename
        for filename in os.listdir(class_dir)
        if filename.lower().endswith(
            VALID_EXTENSIONS
        )
    ])

    print(
        f"{class_name:<30} {count}"
    )


# ============================================
# Fresh Train / Validation / Test Split
# ============================================

print("\n" + "=" * 70)
print("CREATING NEW DATASET SPLITS")
print("=" * 70)

if os.path.exists(CLEAN_SPLIT_DIR):

    shutil.rmtree(
        CLEAN_SPLIT_DIR
    )

random.seed(SEED)


for class_name in CLASS_NAMES:

    class_path = os.path.join(
        CLEAN_MASTER_DIR,
        class_name
    )

    files = [
        filename
        for filename in os.listdir(class_path)
        if filename.lower().endswith(
            VALID_EXTENSIONS
        )
    ]

    random.shuffle(files)

    total = len(files)

    train_end = int(
        total * TRAIN_RATIO
    )

    validation_end = train_end + int(
        total * VALIDATE_RATIO
    )

    train_files = files[:train_end]

    validation_files = files[
        train_end:validation_end
    ]

    test_files = files[
        validation_end:
    ]

    split_files = {
        "train": train_files,
        "validate": validation_files,
        "test": test_files
    }

    for split_name, filenames in split_files.items():

        destination_dir = os.path.join(
            CLEAN_SPLIT_DIR,
            split_name,
            class_name
        )

        os.makedirs(
            destination_dir,
            exist_ok=True
        )

        for filename in filenames:

            source = os.path.join(
                class_path,
                filename
            )

            destination = os.path.join(
                destination_dir,
                filename
            )

            shutil.copy2(
                source,
                destination
            )


# ============================================
# Final Dataset Statistics
# ============================================

print("\n" + "=" * 70)
print("FINAL CLEAN DATASET")
print("=" * 70)

final_counts = {}

for split_name in [
    "train",
    "validate",
    "test"
]:

    print(
        f"\n{split_name.upper()}"
    )

    final_counts[split_name] = {}

    for class_name in CLASS_NAMES:

        path = os.path.join(
            CLEAN_SPLIT_DIR,
            split_name,
            class_name
        )

        count = len([
            filename
            for filename in os.listdir(path)
            if filename.lower().endswith(
                VALID_EXTENSIONS
            )
        ])

        final_counts[split_name][
            class_name
        ] = count

        print(
            f"{class_name:<30} {count}"
        )


# ============================================
# Cleanup Summary
# ============================================

summary_path = os.path.join(
    RESULTS_DIR,
    "cleanup_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "NAIL DATASET CLEANUP SUMMARY\n"
    )

    report.write("=" * 70 + "\n\n")

    report.write(
        f"Original images: {len(images)}\n"
    )

    report.write(
        f"Duplicate groups: "
        f"{len(duplicate_groups)}\n"
    )

    report.write(
        f"Conflicting duplicate groups: "
        f"{len(conflicting_groups)}\n"
    )

    report.write(
        f"Images excluded due to duplicates: "
        f"{len(excluded_paths)}\n"
    )

    report.write(
        f"Images removed by quality filter: "
        f"{len(quality_removed)}\n"
    )

    report.write(
        f"Final clean images: "
        f"{len(clean_images)}\n\n"
    )

    report.write(
        "FINAL SPLIT COUNTS\n"
    )

    report.write("-" * 70 + "\n")

    for split_name in final_counts:

        report.write(
            f"\n{split_name.upper()}\n"
        )

        for class_name, count in (
            final_counts[split_name].items()
        ):

            report.write(
                f"{class_name}: {count}\n"
            )


print("\n" + "=" * 70)
print("DATASET CLEANING COMPLETE")
print("=" * 70)

print(
    "\nClean dataset created at:"
)

print(CLEAN_SPLIT_DIR)

print(
    "\nReports saved in:"
)

print(RESULTS_DIR)

print(
    "\nIMPORTANT: Original dataset was NOT modified."
)