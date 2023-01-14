import argparse
import sys
from typing import Generator, Optional, Set
import cv2
import numpy as np


class StegDecoder:
    N_LOOKUP_BITS = 3

    def __init__(self, image: np.ndarray):
        self.flat_image = image.flatten()
        self.dictionary = self._load_dictionary()
        self.max_word_length = max(len(j) for j in self.dictionary)
        

    def _load_dictionary(self) -> Set[str]:
        with open("dictionary.txt") as f:
            return set(
                j.strip() for j in f.read().splitlines() 
                if not j.strip().startswith("#") and j.strip()
            )

    def _get_next_char(self, indx: int, byte_position: int) -> str:
        assert 0 <= byte_position < 8, "Byte position must be between 0 and 7!"
        
        char_int_value = 0
        bit_mask = 1 << byte_position
        
        # Process a single byte
        for i in range(8):
            # Get current bit
            bit = self.flat_image[indx + i] & bit_mask
            # Add to char value
            char_int_value |= bit << i
        
        # Convert to char
        return chr(char_int_value)

    def _get_next_char_options(self, indx: int) -> Generator[str, None, None]:
        for j in range(self.N_LOOKUP_BITS):
            yield self._get_next_char(indx, j)

    def _dict_get_word(self, word: str) -> Optional[str]:
        # That can be replaced with a wiser distance metric for matching words.
        if word in self.dictionary:
            return word
        return None

    def _get_next_word(self, start_indx: int) -> Optional[str]:
        options = ["" for _ in range(self.N_LOOKUP_BITS)]
        for word_length in range(1, self.max_word_length + 1):
            # Get next chars for each possible index
            next_chars = self._get_next_char_options(start_indx + (word_length - 1) * 8)
            
            # Add found chars to options
            for i, char in enumerate(next_chars):
                options[i] += char
                
            # Check if any of the options is a word, return if OK
            for option in options:
                word = self._dict_get_word(option)
                if word is not None:
                    return word
        # Failure
        return None

    def decode(self) -> Optional[str]:
        message = ""
        # Lookup can begin in every place in the input array.
        start_indx = 0
        while start_indx < self.flat_image.nbytes:
            word = self._get_next_word(start_indx)
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
    message = StegDecoder(image).decode()
    
    if message is None:
        print("Error: Could not find a message in image. Exiting.")
        sys.exit(1)
    
    # Save result
    with open("ID.txt", "w") as f:
        f.write(message)

if __name__ == "__main__":
    main()