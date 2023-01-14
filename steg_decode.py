import argparse
import sys
from typing import Generator, List, Optional, Set
import cv2
import numpy as np

N_LOOKUP_BITS = 3

def load_dictionary() -> Set[str]:
    with open("dictionary.txt") as f:
        return set(
            j for j in f.read().splitlines() 
            if not j.strip().startswith("#")
        )

def get_next_char(flat_image: np.ndarray, indx: int, byte_position: int) -> str:
    assert 0 < byte_position < 8, "Byte position must be between 0 and 7!"
    
    char_int_value = 0
    bit_mask = 1 << byte_position
    
    # Process a single byte
    for i in range(8):
        # Get current bit
        bit = flat_image[indx + i] & bit_mask
        # Add to char value
        char_int_value |= bit << i
    
    # Convert to char
    return chr(char_int_value)

def get_next_char_options(flat_image: np.ndarray, indx: int) -> Generator[str, None, None]:
    for j in range(N_LOOKUP_BITS):
        yield get_next_char(flat_image, indx, j)

def dict_get_word(dictionary: Set[str], word: str) -> Optional[str]:
    # That can be replaced with a wiser distance metric for matching words.
    if word in dictionary:
        return word
    return None

def get_next_word(flat_image: np.ndarray, start_indx: int, dictionary: Set[str]) -> Optional[str]:
    options = ["" for _ in range(N_LOOKUP_BITS)]
    max_word_length = max(len(j) for j in dictionary)
    for word_length in range(1, max_word_length + 1):
        # Get next chars for each possible index
        next_chars = get_next_char_options(flat_image, start_indx + (word_length - 1) * 8)
        
        # Add found chars to options
        for i, char in enumerate(next_chars):
            options[i] += char
            
        # Check if any of the options is a word, return if OK
        for option in options:
            word = dict_get_word(dictionary, option)
            if word is not None:
                return word
    # Failure
    return None

def decode(image: np.ndarray):
    flat_image = image.flatten()
    message = ""
    dictionary = load_dictionary()
    # Lookup can begin in every place in the input array.
    start_indx = 0
    while start_indx < flat_image.nbytes:
        word = get_next_word(flat_image, start_indx, dictionary)
        # Failed to find a word
        if word is None:
            # Keep trying if no words found yet.
            if not message:
                continue
            # Otherwise, we're done.
            return message
        # Found a word - add it to the message
        message += word
        # Move start index to the end of the word - where next word may be.
        start_indx += len(word) * 8
        

def main():
    # Argv parsing
    parser = argparse.ArgumentParser(
        description = 'Decode a hidden message from an image',
        epilog = 'Aviv Naaman [2023]'
    )
    parser.add_argument("image", help="Path to source image, to extract the message from")
    args = parser.parse_args()
    
    # Read image
    image = cv2.imread(args.image)
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