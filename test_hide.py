import pytest
import numpy as np
from steg_hide import hide

@pytest.fixture
def to_hide():
    return "Hello World"

@pytest.fixture
def empty_image():
    return np.zeros((10, 10, 3), dtype=np.uint8)

# I'm so sorry for this
@pytest.fixture
def empty_image_result():
    # Hello World in binary is 
    # 010 010 000 110 010 101 101 100 011 011 000 110 111 100 100 
    # 000 010 101 110 110 111 101 110 010 011 011 000 110 010 0 
    result = np.zeros((10, 10, 3), dtype=np.uint8)
    result_flat = result.flatten()
    result_flat[:45] = [0,1,0, 0,1,0, 0,0,0, 1,1,0, 0,1,0, 1,0,1, 1,0,1, 1,0,0, 0,1,1, 0,1,1, 0,0,0, 1,1,0, 1,1,1, 1,0,0, 1,0,0, ]
    result_flat[45:90] = [0,0,0, 0,1,0, 1,0,1, 1,1,0, 1,1,0, 1,1,1, 1,0,1, 1,1,0, 0,1,0, 0,1,1, 0,1,1, 0,0,0, 1,1,0, 0,1,0, 0,0,0, ]
    return result_flat.reshape(result.shape)

def test_hide_empty_image(empty_image: np.ndarray, empty_image_result: np.ndarray, to_hide: str):
    # Test hide() function with an empty image
    result = hide(empty_image, to_hide)
    assert np.all(result == empty_image_result)
