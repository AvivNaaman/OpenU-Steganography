import argparse
import sys
from typing import List, Optional, Set
import numpy as np
from PIL import Image
from string import ascii_letters

N_LOOKUP_BITS = 3
MESSAGE_VALID_THRESH = 0.5
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
    if len(message) <= 20:
        return False
    words = message.split(" ,.!?")
    return sum(1 for w in words if w.rstrip(" ,.!?") in dictionary) / len(words) > MESSAGE_VALID_THRESH
            
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

def extract_message_with_current(working_string: str, all_strings: List[str]) -> Optional[str]:
    # Must always begin with a letter
    if working_string[0] not in ascii_letters:
        return None

    message = ""
    index = 0
    while index < len(working_string):
        # If any string contains space, switch to it right away.
        for string in all_strings:
            if string[index] == " ":
                working_string = string
        # Char is invalid, try to look for a valid one in other strings.
        # if not found, end of message.
        if working_string[index] not in VALID_CHARS:
            stop = False
            for string in all_strings:
                if string[index] in ascii_letters:
                    working_string = string
                    stop = True
                    break
            if stop:
                break
        message += working_string[index]
    
    return message if is_message_valid(message) else None

def extract_message(strings: List[str]) -> Optional[str]:
    """ 
    Given a list of strings, tries to look up for the hidden message in the strings.
    The message may be split across strings, but always continues from the last index of the previous string.
    """
    max_index = len(strings[0])
    current_index = 0
    while current_index < max_index:
        current_strings = [string[current_index:] for string in strings]
        for current_string in current_strings:
            result = extract_message_with_current(current_string, current_strings)
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
    message = StegDecoder(image).decode()
    
    if message is None:
        print("Error: Could not find a message in image. Exiting.")
        sys.exit(1)
    
    # Save result
    with open("ID.txt", "w") as f:
        f.write(message)

if __name__ == "__main__":
    main()