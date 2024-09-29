from PIL import Image
import os

# Define the folder containing the images
input_folder = 'Data'
output_folder = 'Compressed_Data'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more formats if needed
        img_path = os.path.join(input_folder, filename)
        with Image.open(img_path) as img:
            # Resize the image
            resized_img = img.resize((320, 320), Image.LANCZOS)
            
            # Save the resized image to the output folder
            # If it's a JPEG, specify the quality and optimize the image
            if filename.endswith(".jpg"):
                resized_img.save(os.path.join(output_folder, filename), "JPEG", quality=85, optimize=True)
            # If it's a PNG, optimize it
            elif filename.endswith(".png"):
                resized_img.save(os.path.join(output_folder, filename), "PNG", optimize=True)

print("Image resizing completed.")
