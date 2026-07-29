import numpy as np
from tensorflow.keras.preprocessing import image

IMAGE_SIZE = (300, 300)

def preprocess_image(img_path):
    img = image.load_img(
        img_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array