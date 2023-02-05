import pytest
from steg_decode import StegDecoder
from PIL import Image
import numpy as np

@pytest.fixture
def steg_decoder():
    return StegDecoder(np.array(Image.open("tests/icons8-lock-48_HelloWorld.png")))

def test_load_dictionary(steg_decoder: StegDecoder):
    # Dict loading
    assert steg_decoder.dictionary is not None
    assert "hello" in steg_decoder.dictionary
    assert len(steg_decoder.dictionary) == 3000

def test_get_dictionary(steg_decoder: StegDecoder):
    # Dict get method
    assert steg_decoder._single_word_hueristic("hello") == "hello"
    assert steg_decoder._single_word_hueristic("HELLO") == "HELLO"
    assert steg_decoder._single_word_hueristic("fghdsajhgkdjlah") is None
    
@pytest.fixture
def zeros_decoder(steg_decoder: StegDecoder):
    steg_decoder.flat_image = np.zeros(1000, dtype=np.uint8)
    return steg_decoder

@pytest.fixture
def _0aZ(steg_decoder: StegDecoder):
    # Assign binary data of 3 chars: "0aZ" from higher to lower bits x 2 sequences
    # 00000000 00000011 00000110 00000101 00000001 00000000 00000001 00000010
    # 00000000 00000011 00000110 00000101 00000001 00000000 00000001 00000010
    steg_decoder.flat_image = np.tile([
        0, 3, 6, 5, 1, 0, 1, 2,
    ], 2).astype(np.uint8)
    return steg_decoder


@pytest.fixture
def hello_world_zeros(request, zeros_decoder: StegDecoder):
    # We'll be putting `Hello` @ byte offset 0 & `world` @ byte offset 2 - just for the test.
    # `Hello ` in binary is 01001000 01100101 01101100 01101100 01101111 00100000
    # `world`  in binary is 01110111 01101111 01110010 01101100 01100100 
    HELLO_OFFSET = 0
    WORLD_OFFSET = 2
    
    Hello_ = [
        0, 1, 0, 0, 1, 0, 0, 0,
        0, 1, 1, 0, 0, 1, 0, 1,
        0, 1, 1, 0, 1, 1, 0, 0,
        0, 1, 1, 0, 1, 1, 0, 0,
        0, 1, 1, 0, 1, 1, 1, 1,
        0, 0, 1, 0, 0, 0, 0, 0
    ]
    
    world = [
        0, 1, 1, 1, 0, 1, 1, 1,
        0, 1, 1, 0, 1, 1, 1, 1,
        0, 1, 1, 1, 0, 0, 1, 0,
        0, 1, 1, 0, 1, 1, 0, 0,
        0, 1, 1, 0, 0, 1, 0, 0
    ]
    # apply offset of 3rd byte to world:
    world = [w << WORLD_OFFSET for w in world]
    # mask of bits to keep values of 
    base_mask = ~(np.array([1<<HELLO_OFFSET for _ in Hello_] + [1<<WORLD_OFFSET for _ in world]).astype(np.uint8))
    
    offset: int = request.param
    # Make all bits that needs to be replaced 0
    zeros_decoder.flat_image[offset:offset+len(Hello_ + world)] &= base_mask
    # apply the data
    zeros_decoder.flat_image[offset:offset+len(Hello_ + world)] |= np.array(Hello_ + world, dtype=np.uint8)
    
    # Request param i the offset to put the data in.
    return zeros_decoder
    
@pytest.mark.parametrize("hello_world_zeros", [0,1,2,3,7,8,15,16,32,48,62,63,65], indirect=True)
def test_get_hello_world(hello_world_zeros: StegDecoder):
    assert hello_world_zeros.decode() == "HelloWorld"
