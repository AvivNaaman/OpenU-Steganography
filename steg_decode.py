import argparse
import sys
from typing import List, Optional, Set
import numpy as np
from PIL import Image
import string


class StegDecoder:
    N_LOOKUP_BITS = 3
    # [A-Za-z,.\s!?]:
    VALID_CHARS = set(chr(i) for i in range(ord('A'), ord('Z')+1)) \
        | set(chr(i) for i in range(ord('a'), ord('z')+1)) \
        | {".", ",", " ", "!", "?"}

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

    def _single_word_hueristic(self, word: str) -> Optional[str]:
        # A valid word has at least one alpha char, and all alpha chars are grouped together, surroneded by special chars.
        if any(j in string.ascii_letters for j in word) and \
            all(k in string.ascii_letters for k in word.strip(" .,?!")):
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
                # chr_arr = "".join(j if j in self.VALID_CHARS else '\0' for j in chr_arr)
                curr_bit_options.append(chr_arr)
            all_options.append(curr_bit_options)
        return all_options
    
    def _word_options(self, bits_arrays: List[str], current_index: int) -> List[str]:
        possible_words = []
        bits_arrays = bits_arrays.copy()
        for word_length in range(1, self.max_word_length):
            for bit_array in bits_arrays:
                # Hueristic - if found char that is invalid, stop looking in the array.
                if not bit_array[current_index+word_length-1] in self.VALID_CHARS:
                    bits_arrays.remove(bit_array)
                    continue
                
                word = "".join(bit_array[current_index:current_index+word_length])
                
                # Hueristic - for a single word lookup.
                if self._single_word_hueristic(word):
                    possible_words.append(word)

        return possible_words
    
    MESSAGE_VALID_THRESH = 0.5
    
    def is_message_valid(self, message: str) -> bool:
        if len(message) <= 20:
            return False
        words = message.split(" ,.!?")
        return sum(1 for w in words if w.rstrip(" ,.!?") in self.dictionary) / len(words) > self.MESSAGE_VALID_THRESH
    
    def decode_recursively(self, relevant_arrays: List[str],
                           current_index: int,
                           current_message: str) -> List[str]:
        next_words = self._word_options(relevant_arrays, current_index)
        print(next_words)
        if not next_words:
            if self.is_message_valid(current_message):
                return [current_message]
            return []
        else:
            results = []
            for word_option in next_words:
                # There must be a space/special char between words.
                if not word_option[0].isalpha() and not current_message[-1].isalpha():
                    continue
                results += self.decode_recursively(relevant_arrays,
                                        current_index+len(word_option),
                                        current_message+word_option) 
            return results
    
    def decode_at_offset(self, offset: int) -> Optional[str]:
        """ 
        This method tries to extract text from the image, starting at the given offset.
        Byte is 8-bit long, so trying to calculate this with k and k+8 will have the same result.
        """
        assert 0 <= offset < 8, "Byte offset must be between 0 and 7."
        # This contains N_LOOKUP_BITS arrays, each of which contains all the possible chars for the current index.
        relevant_arrays: List[str] = [j[offset] for j in self._all_chars_options]
        current_index = 96703 # TODO: Make 0
        current_message = ""
        # Try fetching a word from the dictionary, starting at the current index.
        while current_index < 96900:
            results = self.decode_recursively(relevant_arrays, current_index, current_message)
            if results:
                print(" | ".join(results))
                return results[0]
            current_index += 1
        return None
            
            
    def decode(self) -> Optional[str]:
        # Offset of 0-7 bytes from beginning of image array.
        for posssible_offset in range(8):
            res = self.decode_at_offset(6)
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