from PIL import Image
import os

# Define the folder containing the images
input_folder = 'Data10'
output_folder = 'Data10_comp'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all images in the input folder
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    with Image.open(img_path) as img:
        # Resize the image
        resized_img = img.resize((320, 320), Image.LANCZOS)
        # Save the resized image to the output folder
        # resized_img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        resized_img.save(os.path.join(output_folder, filename), quality=85,  optimize=True)

print("Image resizing completed.")
