import pytest
import numpy as np
from steg_decode import StegDecoder
from steg_hide import hide
from PIL import Image

@pytest.fixture
def real_image():
    return np.array(Image.open("icons8-lock-48.png"))

@pytest.fixture
def zeros_image(real_image: np.ndarray):
    return np.zeros_like(real_image)


@pytest.mark.parametrize("message", ["Hello World, what up? For real!"])
def test_hide_and_decode_real(real_image: np.ndarray, message: str):
    assert message == StegDecoder(hide(real_image, message)).decode()
    
@pytest.mark.parametrize("message", ["Hi my friend.... What do you mean?"])
def test_hide_and_decode_zeros(zeros_image: np.ndarray, message: str):
    assert message == StegDecoder(hide(zeros_image, message)).decode()
    

@pytest.mark.parametrize("message", ["Hbjkf", "mqwqqq", "0123456789"])
def test_hide_and_decode_real_invalid(real_image: np.ndarray, message: str):
    assert StegDecoder(hide(real_image, message)).decode() is None
    
@pytest.mark.parametrize("message", ["wa dijfdvfd ylffu mqqeak", "fca kfns"])
def test_hide_and_decode_zeros_invalid(zeros_image: np.ndarray, message: str):
    assert StegDecoder(hide(zeros_image, message)).decode() is None