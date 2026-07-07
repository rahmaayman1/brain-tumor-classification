import numpy as np
from PIL import Image
from src.preprocessing import preprocess_image

def test_grayscale_converted_to_rgb():
    img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8), mode='L')
    result = preprocess_image(img)
    assert result.shape == (1, 224, 224, 3)

def test_different_input_sizes():
    for size in [(100, 100), (512, 512), (300, 400)]:
        img = Image.fromarray(
            np.zeros((size[0], size[1], 3), dtype=np.uint8)
        )
        result = preprocess_image(img)
        assert result.shape == (1, 224, 224, 3)