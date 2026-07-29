import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "v4_clean_efficientnetb3.keras"
)

CLASS_NAMES = [
    "healthy_nails",
    "iron_deficiency",
    "vitamin_b12_deficiency",
    "vitamin_d_deficiency"
]

_model = None


def get_model():
    global _model

    if _model is None:
        print(f"Loading V4 model from: {MODEL_PATH}")

        # TensorFlow is imported only when prediction is requested
        import tensorflow as tf
        from tensorflow.keras.applications.efficientnet import preprocess_input

        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "preprocess_input": preprocess_input
            },
            compile=False
        )

        print("V4 EfficientNetB3 model loaded successfully!")

    return _model