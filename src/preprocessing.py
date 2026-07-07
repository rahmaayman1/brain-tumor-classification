import numpy as np
from PIL import Image
from src.config import IMAGE_SIZE

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess a single PIL image for EfficientNetB0 inference.

    Args:
        image: PIL Image object

    Returns:
        numpy array of shape (1, 224, 224, 3), values 0-255
    """
    # Convert to RGB (handles grayscale and RGBA inputs)
    image = image.convert('RGB')

    # Resize to model input size
    image = image.resize(IMAGE_SIZE)

    # Convert to numpy array
    img_array = np.array(image, dtype='float32')

    # EfficientNet expects 0-255 (not normalized)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array