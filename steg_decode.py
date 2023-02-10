import argparse
import logging
import sys
from typing import List, Optional, Set
import numpy as np
from PIL import Image
from string import ascii_letters

N_LOOKUP_BITS = 3
MESSAGE_VALID_THRESH = 0.5
WORD_SEP = " "
VALID_CHARS = set(chr(i) for i in range(ord('A'), ord('Z')+1)) \
    | set(chr(i) for i in range(ord('a'), ord('z')+1)) \
    | {".", ",", " ", "!", "?"}
    
def _load_dictionary() -> Set[str]:
    with open("dictionary.txt") as f:
        return set(
            j.strip().lower() for j in f.read().splitlines() 
            if not j.strip().startswith("#") and j.strip()
        )
dictionary = _load_dictionary()

def is_message_valid(message: str) -> bool:
    words = message.split(WORD_SEP)
    return len(words) > 20 and \
        sum(1 for w in words if w.rstrip(" ,.!?") in dictionary) / len(words) > MESSAGE_VALID_THRESH
            
def decode_strings(flat_image: np.ndarray, offset: int):
    """ 
    For the given offset, which sets the byte offset for the decoding from the beginning of the array,
    Returns N_LOOKUP_BITS strings, each string at index j extracted from the j-th bit of each byte in the input array.
    """
    result = []
    for i in range(N_LOOKUP_BITS):
        # Extract bits at the required index
        mask = 1 << i
        masked_normed = (flat_image & mask) >> i
        # Prevent out of bounds
        end_indx = offset + 8 * ((flat_image.nbytes-offset) // 8)
        # Group 8 bits --> bytes, and convert ascii values to chars.
        decoded = np.packbits(masked_normed[offset:end_indx] \
            .reshape(-1,8), axis=-1) \
            .tobytes().decode("charmap")
        result.append(decoded)
    return result

def decode_recursive(strings: List[str], message: str, current_string_index: int) -> Optional[str]:
    message_valid = is_message_valid(message)
    for i, string in enumerate(strings):
        if not string:
            continue
        if string[0] not in VALID_CHARS:
            continue
        # First char of word is ascii.
        if message and message[-1] == WORD_SEP and string[0] not in ascii_letters and string[0] != WORD_SEP:
            continue
        # Must be a space when switching words (not at the beginning of the message)
        if i != current_string_index and message and string[0] != WORD_SEP and message[-1] != WORD_SEP:
            continue
        result = decode_recursive([s[1:] for s in strings], message + string[0], i)
        if result:
            return result
    return message if message_valid else None


def extract_message(strings: List[str]) -> Optional[str]:
    """ 
    Given a list of strings, tries to look up for the hidden message in the strings.
    The message may be split across strings, but always continues from the last index of the previous string.
    """
    current_index = 0
    for current_index in range(len(strings[0])):
        current_strings = [string[current_index:] for string in strings]
        result = decode_recursive(current_strings, "", 0)
        if result is not None:
            return result
    # Never found :(
    return None

def decode(image: np.ndarray):
    """ Extracts a message from the given image. """
    # Decoding can begin in each offset from 0 to 7.
    # Byte size is 8, so 8+ will just repeat older results.
    flat_image = image.flatten()
    for i in range(8):
        strings = decode_strings(flat_image, i)
        m = extract_message(strings)
        if m is not None:
            return m

def main():
    # Argv parsing
    parser = argparse.ArgumentParser(
        description = 'Decode a hidden message from an image',
        epilog = 'Aviv Naaman [2023]'
    )
    parser.add_argument("image", help="Path to source image, to extract the message from")
    args = parser.parse_args()
    
    # Read image
    image = np.array(Image.open(args.image))
    if image is None:
        print("Error: Could not read image. Exiting.")
        sys.exit(1)
    
    # Extract message
    message = decode(image)
    
    if message is None:
        print("Error: Could not find a message in image. Exiting.")
        sys.exit(1)
    
    # Save result
    with open("ID.txt", "w") as f:
        f.write(message)

if __name__ == "__main__":
    main()