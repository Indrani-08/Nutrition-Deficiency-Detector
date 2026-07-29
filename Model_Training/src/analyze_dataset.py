import os
import matplotlib.pyplot as plt

from config import TRAIN_DIR, VALIDATE_DIR, TEST_DIR

os.makedirs("results", exist_ok=True)
def count_images(directory):
    counts = {}

    for class_name in sorted(os.listdir(directory)):
        class_path = os.path.join(directory, class_name)

        if not os.path.isdir(class_path):
            continue

        count = sum(
            1
            for file in os.listdir(class_path)
            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            )
        )

        counts[class_name] = count

    return counts


train_counts = count_images(TRAIN_DIR)
validation_counts = count_images(VALIDATE_DIR)
test_counts = count_images(TEST_DIR)

print("\nTRAIN DATASET")
print("=" * 50)
for name, count in train_counts.items():
    print(f"{name:<30} {count}")

print("\nVALIDATION DATASET")
print("=" * 50)
for name, count in validation_counts.items():
    print(f"{name:<30} {count}")

print("\nTEST DATASET")
print("=" * 50)
for name, count in test_counts.items():
    print(f"{name:<30} {count}")


# -------------------------
# Plot training distribution
# -------------------------

classes = list(train_counts.keys())
counts = list(train_counts.values())

plt.figure(figsize=(10, 6))

plt.bar(classes, counts)

plt.title("Training Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig("results/class_distribution.png")

plt.show()