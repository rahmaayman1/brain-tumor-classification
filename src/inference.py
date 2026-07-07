import numpy as np
import tensorflow as tf
from PIL import Image
from src.config import MODEL_PATH, CLASS_NAMES, CLASS_DESCRIPTIONS
from src.preprocessing import preprocess_image

# Load model once at startup (not on every request)
_model = None

def get_model():
    """Load model lazily (only once)."""
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model

def predict(image: Image.Image) -> dict:
    """
    Run inference on a single image.

    Args:
        image: PIL Image object

    Returns:
        dict with predicted class, confidence, and all probabilities
    """
    model = get_model()

    img_array = preprocess_image(image)

    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]

    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    return {
        'predicted_class': predicted_class,
        'confidence': round(confidence * 100, 2),
        'description': CLASS_DESCRIPTIONS[predicted_class],
        'all_probabilities': {
            CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }
    }