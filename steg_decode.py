import argparse
import sys
from typing import List, Optional, Set
import numpy as np
from PIL import Image


class StegDecoder:
    N_LOOKUP_BITS = 3

    def __init__(self, image: np.ndarray):
        self.flat_image = image.flatten()
        self.dictionary = self._load_dictionary()
        self.max_word_length = max(len(j) for j in self.dictionary)
        self._all_chars_options = self._extract_chars()
        

    def _load_dictionary(self) -> Set[str]:
        with open("dictionary.txt") as f:
            return set(
                j.strip().lower() for j in f.read().splitlines() 
                if not j.strip().startswith("#") and j.strip()
            )

    

    def _dict_get_word(self, word: str) -> Optional[str]:
        # That can be replaced with a wiser distance metric for matching words.
        if word.lower().rstrip(" .,?!") in self.dictionary:
            return word
        return None

    def _extract_chars(self):
        """ Given an array, this method extracts all chars that may be a part of a message in this one. """
        all_options = []
        for i in range(self.N_LOOKUP_BITS):
            mask = 1 << i
            # Array of boolean values for the current index i from the end.
            masked_normed = (self.flat_image & mask) >> i
            # For each index inside an 8-bit byte, pack the bits into a byte for that index.
            curr_bit_options = []
            for k in range(8):
                end_indx = k + 8 * ((self.flat_image.nbytes-k) // 8)
                chr_arr = np.packbits(masked_normed[k:end_indx] \
                    .reshape(-1,8), axis=-1) \
                    .tobytes().decode("charmap")
                curr_bit_options.append(chr_arr)
            all_options.append(curr_bit_options)
        return all_options
    
    def next_word_at_offset_index(self, bits_arrays: List[str], current_index: int):
        # Try to get longest word from dictionary starting at current_index from any of the bits_arrays:
        longest_word = None
        for word_length in range(1, self.max_word_length):
            for bit_array in bits_arrays:
                word = "".join(bit_array[current_index:current_index+word_length])
                if self._dict_get_word(word):
                    longest_word = word
        return longest_word
    
    def decode_at_offset(self, offset: int) -> Optional[str]:
        """ 
        This method tries to extract text from the image, starting at the given offset.
        Byte is 8-bit long, so trying to calculate this with k and k+8 will have the same result.
        """
        assert 0 <= offset < 8, "Byte offset must be between 0 and 7."
        # This contains N_LOOKUP_BITS arrays, each of which contains all the possible chars for the current index.
        relevant_arrays: List[str] = [j[offset] for j in self._all_chars_options]
        current_index = 0
        current_message = ""
        # Try fetching a word from the dictionary, starting at the current index.
        while current_index < len(relevant_arrays[0]):
            next_word = self.next_word_at_offset_index(relevant_arrays, current_index)
            if next_word is None:
                # Already building a message - check if valid.
                if len(current_message) >= 5:
                    return current_message
                # Reset invalid message
                elif current_message:
                    current_message = ""
            else:
                current_message += next_word
                current_index += len(next_word)
                continue
            # Keep searching
            current_index += 1
        return None
            
            
    def decode(self) -> Optional[str]:
        # Offset of 0-7 bytes from beginning of image array.
        for posssible_offset in range(8):
            res = self.decode_at_offset(posssible_offset)
            if res:
                return res


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