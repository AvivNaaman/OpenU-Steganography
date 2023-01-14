import pytest
from steg_decode import StegDecoder
import cv2
import numpy as np

@pytest.fixture
def steg_decoder():
    return StegDecoder(cv2.imread("tests/icons8-lock-48_HelloWorld.png"))

def test_load_dictionary(steg_decoder: StegDecoder):
    assert steg_decoder.dictionary is not None
    assert "hello" in steg_decoder.dictionary
    assert len(steg_decoder.dictionary) == 25322
    
def test_get_next_char_zeros(steg_decoder: StegDecoder):
    steg_decoder.flat_image = np.zeros(8, dtype=np.uint8)
    assert steg_decoder._get_next_char(0, 0) == "\x00"

def test_get_next_char(steg_decoder: StegDecoder):
    # Assign binary data of 3 chars: "0aZ" from higher to lower bits x 2 sequences
    # 00000000 00000011 00000110 00000101 00000001 00000000 00000001 00000010
    # 00000000 00000011 00000110 00000101 00000001 00000000 00000001 00000010
    steg_decoder.flat_image = np.tile([
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,1,1,
        0,0,0,0,0,1,1,0,
        0,0,0,0,0,1,0,1,
        0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,1,
        0,0,0,0,0,0,1,0,
    ], 2)
    for j in range(2):
        assert steg_decoder._get_next_char(8*j, 0) == "Z"
        assert steg_decoder._get_next_char(8*j, 1) == "a"
        assert steg_decoder._get_next_char(8*j, 2) == "0"