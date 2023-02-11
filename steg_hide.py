import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def hide(img: np.ndarray, msg: str):
    """ Hides msg in the lower bits of img, bit by bit, in standard order (ltr,top-bottom) """
    # Make sure array is capable of storing all the data
    assert img.nbytes * 8 >= len(msg), "Message too long for image"

    # Construct a numpy array of single bits to hide
    bits_arr = np.unpackbits(np.frombuffer(msg.encode(), dtype=np.uint8)) 
    
    # 3-d image --> 1-d array
    flat_image = img.flatten()
    # Remove last bit of all modified indices
    flat_image[:len(bits_arr)] &= 0b11111110
    # Set last bit of all modified indices to the bit to hide
    flat_image[:len(bits_arr)] |= bits_arr

    # Get shape back
    return flat_image.reshape(img.shape)

def main():
    # Argv parsing
    parser = argparse.ArgumentParser(
        description = 'Hides a message in an image',
        epilog = 'Aviv Naaman [2023]'
    )
    parser.add_argument("image", help="Path to image, to hide the message in")
    parser.add_argument("message", help="Message to hide")
    args = parser.parse_args()
    
    # Read image
    image = np.array(Image.open(args.image))
    if image is None:
        print("Error: Could not read image. Exiting.")
        sys.exit(1)
    
    # Hide message
    result = hide(image, args.message)
    
    # Save result
    dest_file_name = Path(args.image).stem + "_hidden.png"
    Image.fromarray(result).save(dest_file_name)

if __name__ == "__main__":
    main()