from PIL import Image
import os

def crop_image(input_image_path, output_image_path, crop_area):
    """
    Crop the image to the specified area.

    :param input_image_path: Path to the input image file
    :param output_image_path: Path to save the cropped image
    :param crop_area: A tuple (left, top, right, bottom) defining the crop area
    """
    # Open the image file
    with Image.open(input_image_path) as img:
        # Crop the image
        cropped_img = img.crop(crop_area)
        
        # Save the cropped image
        cropped_img.save(output_image_path)
        print(f"Cropped image saved at: {output_image_path}")

# Example usage:
input_image = "input.jpg"  # Path to the input image
output_image = "cropped_image.jpg"  # Path where the cropped image will be saved
crop_area = (100, 100, 400, 400)  # (left, top, right, bottom) coordinates for the crop area

crop_image(input_image, output_image, crop_area)
