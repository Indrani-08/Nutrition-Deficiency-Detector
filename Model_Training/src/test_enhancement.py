import cv2
import matplotlib.pyplot as plt

from preprocessing_v3 import enhance_nail_image


IMAGE_PATH = "../dataset/train/iron_deficiency/YOUR_IMAGE.jpg"


image = cv2.imread(IMAGE_PATH)

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

enhanced = enhance_nail_image(image)


plt.figure(figsize=(6, 5))
plt.imshow(image)
plt.title("Original")
plt.axis("off")
plt.tight_layout()
plt.show()


plt.figure(figsize=(6, 5))
plt.imshow(enhanced)
plt.title("CLAHE + Bilateral Filter")
plt.axis("off")
plt.tight_layout()
plt.show()