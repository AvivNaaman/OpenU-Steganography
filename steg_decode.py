import argparse
import sys
from string import ascii_letters
from typing import List, Optional, Set

import numpy as np
from PIL import Image

# Number of lower bits to look for chars for message decoding.
N_LOOKUP_BITS = 3
# Fraction of words in the message that must be present in the dictionary.
MESSAGE_VALID_THRESH = 0.5
# Minimum number of words in the message.
MINIMUM_WORDS_IN_MESSAGE = 20
# A char that separates words in the message.
WORD_SEP = " "
# All the letters that may appear in a word.
WORD_LETTERS = ascii_letters
# Symbols that may appear in a message, additionally to seperators and core chars.
EXTRA_MESSAGE_SYMBOLS = ",.!?"
# All the allowed chars in a message.
ALL_VALID_CHARS = set(WORD_LETTERS) | {WORD_SEP} | set(EXTRA_MESSAGE_SYMBOLS)
BITS_IN_BYTE = 8
    
def _load_dictionary() -> Set[str]:
    with open("dictionary.txt") as f:
        return set(
            j.strip().lower() for j in f.read().splitlines() 
            if not j.strip().startswith("#") and j.strip()
        )
dictionary = _load_dictionary()

def is_message_valid(message: str) -> bool:
    """ Returns whether message is considered a valid one. """
    words = message.split(WORD_SEP)
    return len(words) >= MINIMUM_WORDS_IN_MESSAGE and \
        sum(w.rstrip(WORD_SEP+EXTRA_MESSAGE_SYMBOLS) in dictionary for w in words) / len(words) > MESSAGE_VALID_THRESH
            
def decode_strings(flat_image: np.ndarray, offset: int):
    """ 
    For the given offset, which sets the byte offset for the decoding from the beginning 
    of the array, Returns N_LOOKUP_BITS strings, each string at index j extracted 
    from the j-th bit of each byte in the input array.
    """
    result = []
    for i in range(N_LOOKUP_BITS):
        # Extract bits at the required index
        mask = 1 << i
        masked_normed = (flat_image & mask) >> i
        # Prevent out of bounds
        end_indx = offset + BITS_IN_BYTE * ((flat_image.nbytes-offset) // BITS_IN_BYTE)
        # Pack extracted bits into bytes, decode to string by charmap (extended ascii)
        decoded = np.packbits(masked_normed[offset:end_indx] \
            .reshape(-1, BITS_IN_BYTE), axis=-1) \
            .tobytes().decode("charmap")
        result.append(decoded)
    return result

def find_message_recursive(strings: List[str], message: str, curr_str_indx: int) -> Optional[str]:
    """
    Looks up for a valid message in the strings as required. Gets the strings as an input,
    currently built message, and the index of the last used string from strings.
    Returns a message if found and valid, None otherwise.
    """
    for i, string in enumerate(strings):
        # Invalid char - current string does not match.
        if not string or string[0] not in ALL_VALID_CHARS:
            continue
        # There must be a space before or after changing the current source string.
        if i != curr_str_indx and message and string[0] != WORD_SEP and message[-1] != WORD_SEP:
            continue
        # Recursive step - pass the (potentially) new message, look up on next char of each string.
        result = find_message_recursive([s[1:] for s in strings], message + string[0], i)
        # return if found a valid message.
        if result:
            return result
    return message if is_message_valid(message) else None


def find_message(strings: List[str]) -> Optional[str]:
    """ 
    Given a list of strings, tries to look up for the hidden message in the strings.
    Returns the message if found, None otherwise.
    """
    # Try to find a message in each index of the strings.
    for current_index in range(len(strings[0])):
        current_strings = [string[current_index:] for string in strings]
        result = find_message_recursive(current_strings, "", 0)
        # return only if found a valid message.
        if result is not None:
            return result
    return None

def decode(image: np.ndarray) -> Optional[str]:
    """ 
    Extracts a message from the given image.
    Returns the message if found, None otherwise.
    """
    # Decoding can begin in each offset from 0 to 7.
    # Byte size is 8, so 8+ will just repeat older results.
    flat_image = image.flatten()
    for i in range(BITS_IN_BYTE):
        strings = decode_strings(flat_image, i)
        m = find_message(strings)
        # return only if found a valid message.
        if m is not None:
            return m
    return None

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
        print("Error: Could not find a message in image!")
        sys.exit(1)
    
    # Save result
    with open("ID.txt", "w") as f:
        f.write(message)

if __name__ == "__main__":
    main()