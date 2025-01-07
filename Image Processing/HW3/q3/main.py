# Itamar Brotzky, 207931296
# Or Diner, 207035809

import cv2
import numpy as np
from matplotlib import pyplot as plt

def plot_images(original_image, cleaned_image):
    # Display
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Fixed Image")
    plt.imshow(cleaned_image, cmap='gray')
    plt.show()

# Part A

# Load the image
image = cv2.imread('broken.jpg', cv2.IMREAD_GRAYSCALE)

# Apply median filtering
median_filtered = cv2.medianBlur(image, 5)  # Kernel size = 5x5

# Apply bilateral filter
bilateral_filtered = cv2.bilateralFilter(median_filtered, d=4, sigmaColor=30, sigmaSpace=30)

# Save and display results
cv2.imwrite("fixed_broken_part_a.jpg", bilateral_filtered)

# Show the images
plot_images(image, bilateral_filtered)


# Part B

# Load the noised images
noised_images = np.load("noised_images.npy")

# Perform pixel-wise averaging
averaged_image = np.mean(noised_images, axis=0).astype(np.uint8)

# Post-processing with bilateral filter
final_image = cv2.bilateralFilter(averaged_image, d=3, sigmaColor=20, sigmaSpace=20)

# Save and display results
cv2.imwrite("fixed_broken_part_b.jpg", final_image)

# Show the images
plot_images(image, final_image)
