import pytest
import numpy as np
from PIL import Image
from src.inference import predict
from src.preprocessing import preprocess_image

def test_preprocess_output_shape():
    img = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
    result = preprocess_image(img)
    assert result.shape == (1, 224, 224, 3)

def test_preprocess_pixel_range():
    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 128)
    result = preprocess_image(img)
    assert result.min() >= 0
    assert result.max() <= 255

def test_predict_output_keys():
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = predict(img)
    assert 'predicted_class' in result
    assert 'confidence' in result
    assert 'all_probabilities' in result
    assert 'description' in result

def test_predict_valid_class():
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = predict(img)
    assert result['predicted_class'] in [
        'glioma', 'meningioma', 'notumor', 'pituitary'
    ]

def test_predict_probabilities_sum():
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    result = predict(img)
    total = sum(result['all_probabilities'].values())
    assert abs(total - 100.0) < 0.1