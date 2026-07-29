import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import TRAIN_DIR, RESULTS_DIR


# ============================================================
# SETTINGS
# ============================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "roi_tests"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# NAIL ROI EXTRACTION
# ============================================================

def extract_nail_roi(image):
    """
    Experimental nail ROI extraction.

    This is NOT the final segmentation method.

    Steps:
        1. Resize for consistent processing
        2. Convert RGB -> YCrCb
        3. Estimate skin/hand region
        4. Morphological cleanup
        5. Find candidate contours
        6. Select central plausible region
        7. Return bounding-box crop
    """

    original = image.copy()

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Resize only for processing
    # --------------------------------------------------------

    process_width = 500

    scale = process_width / width

    process_height = int(
        height * scale
    )

    resized = cv2.resize(
        image,
        (process_width, process_height)
    )


    # --------------------------------------------------------
    # RGB -> YCrCb
    # --------------------------------------------------------

    ycrcb = cv2.cvtColor(
        resized,
        cv2.COLOR_RGB2YCrCb
    )


    # --------------------------------------------------------
    # Broad skin-region threshold
    #
    # This is deliberately broad because skin tones and
    # lighting conditions vary considerably.
    # --------------------------------------------------------

    lower = np.array(
        [0, 125, 70],
        dtype=np.uint8
    )

    upper = np.array(
        [255, 180, 140],
        dtype=np.uint8
    )


    mask = cv2.inRange(
        ycrcb,
        lower,
        upper
    )


    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )


    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    debug = resized.copy()


    if not contours:

        return None, mask, debug


    # --------------------------------------------------------
    # Candidate filtering
    # --------------------------------------------------------

    candidates = []

    image_area = (
        process_width * process_height
    )


    center_x = process_width / 2
    center_y = process_height / 2


    for contour in contours:

        area = cv2.contourArea(
            contour
        )


        # Ignore tiny regions
        if area < image_area * 0.002:
            continue


        x, y, w, h = cv2.boundingRect(
            contour
        )


        # Ignore enormous regions
        if w * h > image_area * 0.80:
            continue


        contour_center_x = x + w / 2
        contour_center_y = y + h / 2


        distance = np.sqrt(
            (contour_center_x - center_x) ** 2
            +
            (contour_center_y - center_y) ** 2
        )


        # Prefer sizeable objects near image centre
        score = (
            area /
            (1 + distance)
        )


        candidates.append(
            (
                score,
                x,
                y,
                w,
                h
            )
        )


    if not candidates:

        return None, mask, debug


    # --------------------------------------------------------
    # Best candidate
    # --------------------------------------------------------

    candidates.sort(
        reverse=True,
        key=lambda item: item[0]
    )


    _, x, y, w, h = candidates[0]


    # --------------------------------------------------------
    # Add padding
    # --------------------------------------------------------

    padding_x = int(
        w * 0.15
    )

    padding_y = int(
        h * 0.15
    )


    x1 = max(
        0,
        x - padding_x
    )

    y1 = max(
        0,
        y - padding_y
    )

    x2 = min(
        process_width,
        x + w + padding_x
    )

    y2 = min(
        process_height,
        y + h + padding_y
    )


    # --------------------------------------------------------
    # Debug rectangle
    # --------------------------------------------------------

    cv2.rectangle(
        debug,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        3
    )


    # --------------------------------------------------------
    # Convert coordinates back to original image
    # --------------------------------------------------------

    inverse_scale = 1 / scale


    original_x1 = int(
        x1 * inverse_scale
    )

    original_y1 = int(
        y1 * inverse_scale
    )

    original_x2 = int(
        x2 * inverse_scale
    )

    original_y2 = int(
        y2 * inverse_scale
    )


    roi = original[
        original_y1:original_y2,
        original_x1:original_x2
    ]


    if roi.size == 0:

        return None, mask, debug


    return roi, mask, debug


# ============================================================
# TEST ONE IMAGE
# ============================================================

def test_image(
    image_path,
    class_name,
    number
):

    image_bgr = cv2.imread(
        image_path
    )


    if image_bgr is None:

        print(
            "Could not load:",
            image_path
        )

        return


    image = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


    roi, mask, debug = extract_nail_roi(
        image
    )


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(15, 5)
    )


    ax1 = fig.add_subplot(
        1,
        3,
        1
    )

    ax1.imshow(
        image
    )

    ax1.set_title(
        "Original"
    )

    ax1.axis(
        "off"
    )


    ax2 = fig.add_subplot(
        1,
        3,
        2
    )

    ax2.imshow(
        mask,
        cmap="gray"
    )

    ax2.set_title(
        "Detected Mask"
    )

    ax2.axis(
        "off"
    )


    ax3 = fig.add_subplot(
        1,
        3,
        3
    )


    if roi is not None:

        ax3.imshow(
            roi
        )

        ax3.set_title(
            "Detected ROI"
        )

    else:

        ax3.text(
            0.5,
            0.5,
            "ROI NOT FOUND",
            horizontalalignment="center",
            verticalalignment="center"
        )

        ax3.set_title(
            "Failed"
        )


    ax3.axis(
        "off"
    )


    plt.suptitle(
        class_name
    )


    plt.tight_layout()


    output_path = os.path.join(
        OUTPUT_DIR,
        f"{class_name}_{number}.png"
    )


    plt.savefig(
        output_path
    )


    plt.close()


    print(
        "Saved:",
        output_path
    )


# ============================================================
# TEST MULTIPLE DATASET IMAGES
# ============================================================

print("=" * 60)
print("NAIL ROI EXTRACTION TEST")
print("=" * 60)


for class_name in os.listdir(
    TRAIN_DIR
):

    class_path = os.path.join(
        TRAIN_DIR,
        class_name
    )


    if not os.path.isdir(
        class_path
    ):

        continue


    print(
        "\nTesting:",
        class_name
    )


    image_files = [

        file

        for file in os.listdir(
            class_path
        )

        if file.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )

    ]


    # Test 5 images from each class
    image_files = image_files[:5]


    for index, filename in enumerate(
        image_files
    ):

        image_path = os.path.join(
            class_path,
            filename
        )


        test_image(
            image_path,
            class_name,
            index + 1
        )


print("\n" + "=" * 60)
print("ROI TEST COMPLETE")
print("=" * 60)

print(
    "\nCheck:"
)

print(
    OUTPUT_DIR
)