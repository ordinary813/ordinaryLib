# Itamar Brotzky, 207931296
# Or Diner, 207035809

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import matplotlib.pyplot as plt

def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    # Your code goes here
    # Ensure the input image is a float64 for precise calculations
    im = im.astype(np.float64)

    # Get the dimensions of the input image
    rows, cols = im.shape

    # Initialize the output image
    cleanIm = np.zeros_like(im)

    # Create spatial Gaussian mask (gs)
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    gs = np.exp(-(x ** 2 + y ** 2) / (2 * stdSpatial ** 2))

    # Pad the image to handle border pixels
    padded_im = np.pad(im, pad_width=radius, mode='reflect')

    # Loop over each pixel in the image
    for i in range(rows):
        for j in range(cols):
            # Extract the local window
            window = padded_im[i:i + 2 * radius + 1, j:j + 2 * radius + 1]

            # Compute intensity Gaussian mask (gi)
            gi = np.exp(-((window - im[i, j]) ** 2) / (2 * stdIntensity ** 2))

            # Compute combined mask
            combined_mask = gs * gi

            # Normalize the combined mask
            combined_mask /= np.sum(combined_mask)

            # Compute the filtered value for the current pixel
            cleanIm[i, j] = np.sum(combined_mask * window)

    # Convert the result back to uint8
    cleanIm = np.clip(cleanIm, 0, 255).astype(np.uint8)

    return cleanIm


def plot_images(original_image, clean_image):
    # Plots the 2 images the cleaned one next to the original one
    plt.subplot(121)
    plt.title("image")
    plt.imshow(original_image, cmap='gray')

    plt.subplot(122)
    plt.title("cleaned image")
    plt.imshow(clean_image, cmap='gray')

    plt.show()


# Load the noisy images
taj_image = cv2.imread("taj.jpg", cv2.IMREAD_GRAYSCALE)
balls_image = cv2.imread("balls.jpg", cv2.IMREAD_GRAYSCALE)
noisy_gray_image = cv2.imread("NoisyGrayImage.png", cv2.IMREAD_GRAYSCALE)

# Apply the bilateral filter
clean_taj = clean_Gaussian_noise_bilateral(taj_image, radius=9, stdSpatial=8.5, stdIntensity=45)
clean_gray_image = clean_Gaussian_noise_bilateral(noisy_gray_image, radius=7, stdSpatial=30, stdIntensity=90)
clean_balls = clean_Gaussian_noise_bilateral(balls_image, radius=12, stdSpatial=9, stdIntensity=30)

# Show the images
plot_images(taj_image, clean_taj)
plot_images(balls_image, clean_balls)
plot_images(noisy_gray_image, clean_gray_image)

# Write the images to the folder
cv2.imwrite("clean_taj.jpg", clean_taj)
cv2.imwrite("clean_gray_image.jpg", clean_gray_image)
cv2.imwrite("clean_balls.jpg", clean_balls)
