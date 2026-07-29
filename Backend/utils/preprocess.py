import numpy as np
from PIL import Image

IMAGE_SIZE = (300, 300)


def preprocess_image(img_path):
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE)

        img_array = np.asarray(img, dtype=np.float32)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array