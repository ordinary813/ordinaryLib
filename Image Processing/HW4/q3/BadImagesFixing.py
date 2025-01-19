# Itamar Brotzky, 207931296
# Or Diner, 207035809

# Please replace the above comments with your names and ID numbers in the same format.

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

def clean_baby(im):
    # Define the source points (corners of the input images)
    source_points = np.array([[[6, 20], [111, 20], [111, 130], [6, 130]],
                              [[78, 162], [145, 117], [245, 160], [132, 244]],
                              [[182, 5], [249, 70], [176, 120], [121, 51]]]
                             , dtype=np.float32)

    # Define the destination points (corners of the output image)
    image_height, image_width = im.shape[0:2]
    # image_height,image_width = im.shape

    destination_points = np.array([[0, 0], [image_width-1, 0], [image_width-1, image_height-1], [0, image_height-1]], dtype=np.float32)

    # Extracted images array
    images_arr = []

    # Extract all the images
    for points in source_points:

        # Compute the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(points, destination_points)

        # Perform the perspective warp
        transformed_image = cv2.warpPerspective(im, matrix, (image_width, image_height))

        if points[0][0] == 6:
            kernel_size = 9
        else:
            kernel_size = 11
        filtered_image = cv2.medianBlur(transformed_image, ksize=kernel_size)

        # Add the new image to the array
        images_arr.append(filtered_image)

    # Stack the images into a 3D array
    stacked_images = np.stack(images_arr, axis=-1)

    # Compute the median along the last axis (across the three images)
    median_image = np.mean(stacked_images, axis=-1).astype(np.uint8)

    adjusted = cv2.convertScaleAbs(median_image, alpha=1.2, beta=-35)

    return  adjusted


# CLEAN WINDMILL
def clean_windmill(im):
    # Compute the 2D Fourier Transform
    f = fft2(im)
    fshift = fftshift(f)

    rows, cols = im.shape
    crow, ccol = rows // 2, cols // 2

    # Find abnormally large coefficients in the magnitude spectrum
    mask = np.ones((rows, cols), np.uint8)
    threshold = 200000
    for i in range(rows):
        for j in range(cols):
            # Skip the center of the magnitude spectrum (radius 10)
            if (i - crow) ** 2 + (j - ccol) ** 2 <= 10 ** 2:
                continue
            # Check if the magnitude of the Fourier coefficient is greater than the threshold
            if np.abs(fshift[i, j]) > threshold:
                mask[i, j] = 0

    # Apply the mask
    fshift = fshift * mask

    # Inverse Fourier Transform to get the cleaned image
    f_ishift = ifftshift(fshift)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back


# CLEAN WATERMELON
def clean_watermelon(im):
    # sharpen image
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1],])
    im_sharpened = cv2.filter2D(im, -1, kernel)

    return im_sharpened


# CLEAN UMBRELLA
def construct_degradation_kernel(shape, shift):
    # builds a degragation kernel with a predetermined shift
    kernel = np.zeros(shape)
    kernel[0, 0] = 1  # Unshifted component
    kernel[shift[0], shift[1]] = 1  # Shifted component
    return kernel / 2  # Normalize to preserve energy



def clean_umbrella(im, custom_shift=(4, 79)):
    # application of the simpler filter inverse
    lam = 0.0001

    # Construct kernel
    h = construct_degradation_kernel(im.shape, custom_shift)

    g = im

    G = fft2(g)
    H = fft2(h)
    H_conj = np.conjugate(H)

    F = (H_conj / (H * H_conj + lam)) * G
    f = ifft2(F)

    return np.real(f)


# CLEAN USA FLAG
def get_neighbors(image, row, col):
    neighbors = []

    # Add the 6 left pixels if valid
    for i in range(1, 7):
        if col - i >= 0:
            neighbors.append(image[row, col - i])
        else:
            neighbors.append(0)  # out-of-bounds

    # Add the current pixel
    neighbors.append(image[row, col])

    # Add the 6 right pixels if valid
    for i in range(1, 7):
        if col + i < image.shape[1]:
            neighbors.append(image[row, col + i])
        else:
            neighbors.append(0)  # out-of-bounds

    return neighbors


def clean_USAflag(image):
    # Your code goes here
    # application of median filter HORIZONTALLY starting at predetermined thresholds
    filtered_image = np.copy(image)

    # Apply horizontal median filter only to rows > 75 and cols > 120
    for x in range(image.shape[0]):  # Iterate over each row
        if x < 85:
            for y in range(140, image.shape[1]):
                filtered_image[x, y] = np.median(get_neighbors(image, x, y))
        else:
            for y in range(image.shape[1]):
                filtered_image[x, y] = np.median(get_neighbors(image, x, y))

    return filtered_image


def clean_house(im):

    # The kernel that created the blured image
    kernel = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])

    # Avoid deviation by zero
    epsilon = 10**(-16)

    # Compute the Fourier Transform of the image and kernel
    im_fft = np.fft.fft2(im)
    kernel_fft = np.fft.fft2(kernel, s=im.shape)

    # Deconvolve the image
    deconvolve_image = im_fft / (kernel_fft + epsilon)

    # Inverse Fourier Transform
    restored = np.real(np.fft.ifft2(deconvolve_image))

    return restored


def clean_bears(im):

    corrected_image = cv2.convertScaleAbs(im, alpha=4, beta=-115)

    return corrected_image
