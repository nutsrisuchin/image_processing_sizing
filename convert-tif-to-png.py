import os
from PIL import Image, UnidentifiedImageError

# Define the input and output directories
input_dir = "raw_image"
output_dir = "png_image"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate through all files in the input directory
for filename in os.listdir(input_dir):
    input_path = os.path.join(input_dir, filename)
    
    try:
        with Image.open(input_path) as img:
            if img.format == "TIFF":
                # Save with PNG extension
                output_filename = os.path.splitext(filename)[0] + ".png"
                output_path = os.path.join(output_dir, output_filename)
                img.save(output_path, "PNG")
                print(f"Converted {filename} to {output_filename}")
            else:
                print(f"Skipped {filename}: not a TIFF image")
    except UnidentifiedImageError:
        print(f"Skipped {filename}: not an image file")
