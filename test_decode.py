import pytest
from steg_decode import StegDecoder
import cv2
import numpy as np

@pytest.fixture
def steg_decoder():
    return StegDecoder(cv2.imread("tests/icons8-lock-48_HelloWorld.png"))

def test_load_dictionary(steg_decoder: StegDecoder):
    # Dict loading
    assert steg_decoder.dictionary is not None
    assert "hello" in steg_decoder.dictionary
    assert len(steg_decoder.dictionary) == 25322

def test_get_dictionary(steg_decoder: StegDecoder):
    # Dict get method
    assert steg_decoder._dict_get_word("hello") == "hello"
    assert steg_decoder._dict_get_word("HELLO") == "HELLO"
    assert steg_decoder._dict_get_word("fghdsajhgkdjlah") is None
    
@pytest.fixture
def zeros_decoder(steg_decoder: StegDecoder):
    steg_decoder.flat_image = np.zeros(8, dtype=np.uint8)
    return steg_decoder

def test_get_next_char_zeros(zeros_decoder: StegDecoder):
    for j in range(8):
        assert zeros_decoder._get_next_char(0, j) == "\x00"

@pytest.fixture
def _0aZ(steg_decoder: StegDecoder):
    # Assign binary data of 3 chars: "0aZ" from higher to lower bits x 2 sequences
    # 00000000 00000011 00000110 00000101 00000001 00000000 00000001 00000010
    # 00000000 00000011 00000110 00000101 00000001 00000000 00000001 00000010
    steg_decoder.flat_image = np.tile([
        0, 3, 6, 5, 1, 0, 1, 2,
    ], 2).astype(np.uint8)
    return steg_decoder

def test_get_next_char(_0aZ: StegDecoder):
    for j in range(2):
        assert _0aZ._get_next_char(8*j, 0) == "Z"
        assert _0aZ._get_next_char(8*j, 1) == "a"
        assert _0aZ._get_next_char(8*j, 2) == "0"

def test_get_next_char_options_zeros(zeros_decoder: StegDecoder):
    r = zeros_decoder._get_next_char_options(0)
    for result in r:
        assert result == "\x00"

def test_get_next_char_options(_0aZ: StegDecoder):
    for i in range(2):
        r = _0aZ._get_next_char_options(8*i)
        for exp, result in zip("Za0", r):
            assert exp == result

def test_get_next_word_nothing(zeros_decoder: StegDecoder, _0aZ: StegDecoder):
    assert zeros_decoder._get_next_word(0) is None
    assert _0aZ._get_next_word(0) is None

@pytest.fixture
def hello_world(steg_decoder: StegDecoder):
    steg_decoder.flat_image = np.zeros((10,10,3), dtype=np.uint8)
    

def test_get_hello_world():
    pass
