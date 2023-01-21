import argparse
import sys
from PIL import Image
import numpy as np
from pathlib import Path

def hide(img: np.ndarray, msg: str):
    # Make sure array is capable of storing all the data
    assert img.nbytes * 8 >= len(msg), "Message too long for image"
    
    # 3-d image --> 1-d array
    flat_image = img.flatten()
    
    flat_index = 0
    for char in msg:
        # Convert char to ASCII value
        char_int_value = ord(char)
        # Process a single byte
        for i in range(8):
            # Value of current bit to process
            bit = (char_int_value >> (7-i)) & 1
            # Hide in the least significant bit
            flat_image[flat_index] = (flat_image[flat_index] & 254) | bit
            flat_index += 1
    
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