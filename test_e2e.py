import pytest
import numpy as np
import cv2
from steg_decode import StegDecoder
from steg_hide import hide

@pytest.fixture
def real_image():
    return cv2.cvtColor(cv2.imread("icons8-lock-48.png"), cv2.COLOR_BGR2RGB)

@pytest.fixture
def zeros_image(real_image: np.ndarray):
    return np.zeros_like(real_image)


@pytest.mark.parametrize("message", ["Hello World", "Goodbye", "For real"])
def test_hide_and_decode_real(real_image: np.ndarray, message: str):
    assert message == StegDecoder(hide(real_image, message)).decode()
    
@pytest.mark.parametrize("message", ["What did you mean", "Hi my friend"])
def test_hide_and_decode_zeros(zeros_image: np.ndarray, message: str):
    assert message == StegDecoder(hide(zeros_image, message)).decode()
    

@pytest.mark.parametrize("message", ["Hbjkf", "mqwqqq", "0123456789"])
def test_hide_and_decode_real_invalid(real_image: np.ndarray, message: str):
    assert StegDecoder(hide(real_image, message)).decode() is None
    
@pytest.mark.parametrize("message", ["wa dijfdvfd ylffu mqqeak", "fca kfns"])
def test_hide_and_decode_zeros_invalid(zeros_image: np.ndarray, message: str):
    assert StegDecoder(hide(zeros_image, message)).decode() is None