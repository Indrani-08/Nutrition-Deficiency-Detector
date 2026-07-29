# ============================================================
# Project : Nail Nutrition Deficiency Detection
# Module  : Dataset Image Quality Analysis
# ============================================================

import os
import cv2
import csv
import hashlib
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

from config import (
    TRAIN_DIR,
    VALIDATE_DIR,
    TEST_DIR,
    RESULTS_DIR
)


# ============================================================
# CONFIGURATION
# ============================================================

QUALITY_DIR = os.path.join(
    RESULTS_DIR,
    "quality_analysis"
)

os.makedirs(
    QUALITY_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Thresholds
#
# These are initial screening thresholds, NOT absolute rules.
# We will inspect the results before removing anything.
# ------------------------------------------------------------

MIN_WIDTH = 150
MIN_HEIGHT = 150

BLUR_THRESHOLD = 50.0

DARK_THRESHOLD = 45.0
BRIGHT_THRESHOLD = 220.0

# Percentage of pixels that are almost white
OVEREXPOSED_PIXEL_THRESHOLD = 245

# If more than 60% of pixels are almost white, flag image
OVEREXPOSED_PERCENTAGE_THRESHOLD = 60.0


# ============================================================
# DATASET LOCATIONS
# ============================================================

DATASETS = {
    "train": TRAIN_DIR,
    "validate": VALIDATE_DIR,
    "test": TEST_DIR
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_blur(image):
    """
    Calculates Laplacian variance.

    Lower value generally means blurrier image.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def calculate_brightness(image):
    """
    Returns average grayscale brightness.

    Range:
        0   = black
        255 = white
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        np.mean(gray)
    )


def calculate_contrast(image):
    """
    Standard deviation of grayscale pixel values.

    Very low values may indicate low-contrast images.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return float(
        np.std(gray)
    )


def calculate_overexposed_percentage(image):
    """
    Calculates percentage of almost-white pixels.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    overexposed_pixels = np.sum(
        gray >= OVEREXPOSED_PIXEL_THRESHOLD
    )

    total_pixels = gray.size

    percentage = (
        overexposed_pixels /
        total_pixels
    ) * 100

    return float(percentage)


def calculate_dark_percentage(image):
    """
    Percentage of extremely dark pixels.
    Useful for detecting underexposed images.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    dark_pixels = np.sum(
        gray <= 20
    )

    percentage = (
        dark_pixels /
        gray.size
    ) * 100

    return float(percentage)


def calculate_file_hash(file_path):
    """
    MD5 hash for exact duplicate detection.
    """

    hash_md5 = hashlib.md5()

    try:

        with open(file_path, "rb") as file:

            for chunk in iter(
                lambda: file.read(4096),
                b""
            ):

                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    except Exception:

        return None


# ============================================================
# ANALYZE SINGLE IMAGE
# ============================================================

def analyze_image(
    file_path,
    dataset_name,
    class_name
):

    filename = os.path.basename(
        file_path
    )

    image = cv2.imread(
        file_path
    )


    # --------------------------------------------------------
    # Corrupt / unreadable image
    # --------------------------------------------------------

    if image is None:

        return {
            "dataset": dataset_name,
            "class": class_name,
            "filename": filename,
            "path": file_path,

            "width": 0,
            "height": 0,

            "brightness": 0,
            "contrast": 0,
            "blur_score": 0,

            "overexposed_percent": 0,
            "dark_percent": 0,

            "hash": "",

            "flags": "CORRUPT"
        }


    height, width = image.shape[:2]


    # --------------------------------------------------------
    # Quality measurements
    # --------------------------------------------------------

    blur_score = calculate_blur(
        image
    )

    brightness = calculate_brightness(
        image
    )

    contrast = calculate_contrast(
        image
    )

    overexposed_percent = calculate_overexposed_percentage(
        image
    )

    dark_percent = calculate_dark_percentage(
        image
    )

    file_hash = calculate_file_hash(
        file_path
    )


    # --------------------------------------------------------
    # Generate warning flags
    # --------------------------------------------------------

    flags = []


    if width < MIN_WIDTH or height < MIN_HEIGHT:

        flags.append(
            "LOW_RESOLUTION"
        )


    if blur_score < BLUR_THRESHOLD:

        flags.append(
            "BLURRY"
        )


    if brightness < DARK_THRESHOLD:

        flags.append(
            "VERY_DARK"
        )


    if brightness > BRIGHT_THRESHOLD:

        flags.append(
            "VERY_BRIGHT"
        )


    if (
        overexposed_percent
        >
        OVEREXPOSED_PERCENTAGE_THRESHOLD
    ):

        flags.append(
            "OVEREXPOSED"
        )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "dataset": dataset_name,
        "class": class_name,
        "filename": filename,
        "path": file_path,

        "width": width,
        "height": height,

        "brightness": round(
            brightness,
            2
        ),

        "contrast": round(
            contrast,
            2
        ),

        "blur_score": round(
            blur_score,
            2
        ),

        "overexposed_percent": round(
            overexposed_percent,
            2
        ),

        "dark_percent": round(
            dark_percent,
            2
        ),

        "hash": file_hash,

        "flags": (
            "|".join(flags)
            if flags
            else "OK"
        )
    }


# ============================================================
# SCAN DATASET
# ============================================================

results = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


print("=" * 70)
print("DATASET IMAGE QUALITY ANALYSIS")
print("=" * 70)


for dataset_name, dataset_path in DATASETS.items():

    print(
        f"\nScanning {dataset_name.upper()} dataset..."
    )


    if not os.path.exists(
        dataset_path
    ):

        print(
            "Dataset directory not found:",
            dataset_path
        )

        continue


    class_names = sorted(
        os.listdir(dataset_path)
    )


    for class_name in class_names:

        class_path = os.path.join(
            dataset_path,
            class_name
        )


        if not os.path.isdir(
            class_path
        ):

            continue


        count = 0


        for filename in os.listdir(
            class_path
        ):

            if not filename.lower().endswith(
                valid_extensions
            ):

                continue


            file_path = os.path.join(
                class_path,
                filename
            )


            result = analyze_image(
                file_path,
                dataset_name,
                class_name
            )


            results.append(
                result
            )


            count += 1


        print(
            f"  {class_name:<30} {count}"
        )


# ============================================================
# SAVE COMPLETE CSV REPORT
# ============================================================

csv_path = os.path.join(
    QUALITY_DIR,
    "image_quality_report.csv"
)


if results:

    fieldnames = list(
        results[0].keys()
    )


    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


print(
    "\nQuality report saved:"
)

print(
    csv_path
)


# ============================================================
# FIND EXACT DUPLICATES
# ============================================================

hash_groups = defaultdict(
    list
)


for result in results:

    if result["hash"]:

        hash_groups[
            result["hash"]
        ].append(result)


duplicate_groups = {

    file_hash: items

    for file_hash, items
    in hash_groups.items()

    if len(items) > 1
}


duplicate_report_path = os.path.join(
    QUALITY_DIR,
    "exact_duplicates.txt"
)


with open(
    duplicate_report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EXACT DUPLICATE IMAGE REPORT\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    if not duplicate_groups:

        file.write(
            "No exact duplicates detected.\n"
        )


    for group_number, (
        file_hash,
        items
    ) in enumerate(
        duplicate_groups.items(),
        start=1
    ):

        file.write(
            f"Duplicate Group {group_number}\n"
        )

        file.write(
            f"Hash: {file_hash}\n"
        )


        for item in items:

            file.write(
                f"  {item['dataset']} / "
                f"{item['class']} / "
                f"{item['filename']}\n"
            )


        file.write(
            "\n"
        )


# ============================================================
# FLAG STATISTICS
# ============================================================

flag_counter = Counter()


for result in results:

    flags = result[
        "flags"
    ].split("|")


    for flag in flags:

        flag_counter[
            flag
        ] += 1


# ============================================================
# CLASS STATISTICS
# ============================================================

class_statistics = defaultdict(
    lambda: {
        "total": 0,
        "flagged": 0,
        "blur": [],
        "brightness": [],
        "contrast": []
    }
)


for result in results:

    key = (
        result["dataset"],
        result["class"]
    )


    stats = class_statistics[
        key
    ]


    stats[
        "total"
    ] += 1


    if result["flags"] != "OK":

        stats[
            "flagged"
        ] += 1


    if result["flags"] != "CORRUPT":

        stats[
            "blur"
        ].append(
            result["blur_score"]
        )

        stats[
            "brightness"
        ].append(
            result["brightness"]
        )

        stats[
            "contrast"
        ].append(
            result["contrast"]
        )


# ============================================================
# SUMMARY REPORT
# ============================================================

summary_path = os.path.join(
    QUALITY_DIR,
    "quality_summary.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "NAIL DATASET QUALITY ANALYSIS\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    file.write(
        f"Total images analyzed: {len(results)}\n\n"
    )


    # --------------------------------------------------------
    # Flags
    # --------------------------------------------------------

    file.write(
        "FLAG COUNTS\n"
    )

    file.write(
        "-" * 70 + "\n"
    )


    for flag, count in sorted(
        flag_counter.items()
    ):

        file.write(
            f"{flag:<25} {count}\n"
        )


    file.write(
        "\n"
    )


    # --------------------------------------------------------
    # Duplicates
    # --------------------------------------------------------

    file.write(
        "DUPLICATE INFORMATION\n"
    )

    file.write(
        "-" * 70 + "\n"
    )


    file.write(
        f"Exact duplicate groups: "
        f"{len(duplicate_groups)}\n"
    )


    duplicate_image_count = sum(

        len(items)

        for items
        in duplicate_groups.values()

    )


    file.write(
        f"Images belonging to duplicate groups: "
        f"{duplicate_image_count}\n\n"
    )


    # --------------------------------------------------------
    # Per-class statistics
    # --------------------------------------------------------

    file.write(
        "PER-CLASS STATISTICS\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )


    for key in sorted(
        class_statistics.keys()
    ):

        dataset_name, class_name = key

        stats = class_statistics[
            key
        ]


        file.write(
            f"{dataset_name.upper()} - {class_name}\n"
        )


        file.write(
            "-" * 70 + "\n"
        )


        file.write(
            f"Total images : {stats['total']}\n"
        )

        file.write(
            f"Flagged      : {stats['flagged']}\n"
        )


        if stats["total"] > 0:

            flagged_percentage = (

                stats["flagged"]
                /
                stats["total"]

            ) * 100


            file.write(
                f"Flagged %    : "
                f"{flagged_percentage:.2f}%\n"
            )


        if stats["blur"]:

            file.write(
                f"Average blur score : "
                f"{np.mean(stats['blur']):.2f}\n"
            )


        if stats["brightness"]:

            file.write(
                f"Average brightness : "
                f"{np.mean(stats['brightness']):.2f}\n"
            )


        if stats["contrast"]:

            file.write(
                f"Average contrast : "
                f"{np.mean(stats['contrast']):.2f}\n"
            )


        file.write(
            "\n"
        )


print(
    "Summary saved:"
)

print(
    summary_path
)


print(
    "\nDuplicate report saved:"
)

print(
    duplicate_report_path
)


# ============================================================
# CREATE FLAG DISTRIBUTION GRAPH
# ============================================================

plot_flags = {

    flag: count

    for flag, count
    in flag_counter.items()

    if flag != "OK"
}


if plot_flags:

    labels = list(
        plot_flags.keys()
    )

    counts = list(
        plot_flags.values()
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.bar(
        labels,
        counts
    )


    plt.title(
        "Dataset Quality Flags"
    )

    plt.xlabel(
        "Quality Issue"
    )

    plt.ylabel(
        "Number of Images"
    )


    plt.xticks(
        rotation=45,
        ha="right"
    )


    plt.tight_layout()


    graph_path = os.path.join(
        QUALITY_DIR,
        "quality_flags.png"
    )


    plt.savefig(
        graph_path
    )

    plt.close()


# ============================================================
# CREATE SAMPLE GRID OF FLAGGED IMAGES
# ============================================================

flagged_results = [

    result

    for result in results

    if result["flags"] != "OK"

]


# Limit visualization only.
# All images are still included in CSV.

sample_results = flagged_results[
    :20
]


if sample_results:

    columns = 4

    rows = int(
        np.ceil(
            len(sample_results)
            /
            columns
        )
    )


    plt.figure(
        figsize=(
            16,
            rows * 4
        )
    )


    for index, result in enumerate(
        sample_results,
        start=1
    ):

        image = cv2.imread(
            result["path"]
        )


        plt.subplot(
            rows,
            columns,
            index
        )


        if image is not None:

            image_rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            plt.imshow(
                image_rgb
            )


        plt.title(
            f"{result['class']}\n"
            f"{result['flags']}",
            fontsize=8
        )


        plt.axis(
            "off"
        )


    plt.tight_layout()


    sample_path = os.path.join(
        QUALITY_DIR,
        "flagged_samples.png"
    )


    plt.savefig(
        sample_path
    )

    plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)

print(
    "QUALITY ANALYSIS COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nTotal images:",
    len(results)
)


print(
    "Flagged images:",
    len(flagged_results)
)


print(
    "Exact duplicate groups:",
    len(duplicate_groups)
)


print(
    "\nResults directory:"
)

print(
    QUALITY_DIR
)