import cv2
import numpy as np
from skimage import morphology


def denoise(img):
    """Remove camera noise while preserving edges."""
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=7,
        hColor=7,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def segment_nail(img):
    """
    Segment nail using HSV thresholding.
    Returns cropped nail image.
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 120])
    upper = np.array([180, 90, 255])

    mask = cv2.inRange(hsv, lower, upper)

    mask = morphology.remove_small_objects(mask.astype(bool), min_size=300)
    mask = morphology.remove_small_holes(mask, area_threshold=300)

    mask = (mask.astype(np.uint8)) * 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return img

    largest = max(contours, key=cv2.contourArea)

    nail_mask = np.zeros_like(mask)

    cv2.drawContours(
        nail_mask,
        [largest],
        -1,
        255,
        thickness=cv2.FILLED
    )

    x, y, w, h = cv2.boundingRect(largest)

    nail_crop = img[y:y+h, x:x+w]

    return nail_crop