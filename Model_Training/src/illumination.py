import cv2
import numpy as np


def gray_world_white_balance(img):
    """Correct color cast using Gray World assumption."""
    result = img.astype(np.float32)

    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    result[:, :, 0] *= avg_gray / (avg_b + 1e-6)
    result[:, :, 1] *= avg_gray / (avg_g + 1e-6)
    result[:, :, 2] *= avg_gray / (avg_r + 1e-6)

    return np.clip(result, 0, 255).astype(np.uint8)


def remove_glare(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def clahe_enhancement(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def illumination_pipeline(img):

    img = gray_world_white_balance(img)

    img = remove_glare(img)

    img = clahe_enhancement(img)

    return img