# Itamar Brotzky, 207931296
# Or Diner, 207035809

import cv2
import numpy as np
import matplotlib.pyplot as plt

# original image
image0 = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)

image1 = cv2.imread("image_1.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("image_2.jpg", cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread("image_3.jpg", cv2.IMREAD_GRAYSCALE)
image4 = cv2.imread("image_4.jpg", cv2.IMREAD_GRAYSCALE)
image5 = cv2.imread("image_5.jpg", cv2.IMREAD_GRAYSCALE)
image6 = cv2.imread("image_6.jpg", cv2.IMREAD_GRAYSCALE)
image7 = cv2.imread("image_7.jpg", cv2.IMREAD_GRAYSCALE)
image8 = cv2.imread("image_8.jpg", cv2.IMREAD_GRAYSCALE)
image9 = cv2.imread("image_9.jpg", cv2.IMREAD_GRAYSCALE)


def calculate_mse(image_A, image_B):
    """
    Calculates the Mean Squared Error (MSE) between two grayscale images.

    Parameters:
        image_A (numpy.ndarray): First grayscale image (2D array).
        image_B (numpy.ndarray): Second grayscale image (2D array).

    Returns:
        float: The Mean Squared Error between the two images.
    """
    # Ensure the images have the same dimensions
    if image_A.shape != image_B.shape:
        raise ValueError("The two images must have the same dimensions.")

    # Compute the Mean Squared Error
    mse = np.mean((image_A.astype(np.float64) - image_B.astype(np.float64)) ** 2)

    return mse


def plot_3_images(original_image, exercise_image, our_image):
    plt.figure(figsize=(16, 8))
    plt.subplot(141)
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")
    plt.title("Original Image")

    plt.subplot(142)
    plt.imshow(exercise_image, cmap="gray")
    plt.axis("off")
    plt.title("Exercise Image")

    plt.subplot(143)
    plt.imshow(our_image, cmap="gray")
    plt.axis("off")
    plt.title("Reconstructed Image")

    print("MSE =", calculate_mse(our_image, exercise_image))

    plt.subplot(144)
    plt.imshow((our_image - exercise_image), cmap="gray")
    plt.axis("off")
    plt.title("Difference Image")

    plt.show()


# Image 1 - Row average

custom_image = np.zeros((image0.shape[0], image0.shape[1]), dtype=np.uint8)

for row in range(image0.shape[0]):
    average = np.mean(image0[row, :])
    custom_image[row, :] = average

plot_3_images(image0, image1, custom_image)
cv2.imwrite("Our_image_1.jpg", custom_image)


# Image 2 - Gaussian Blur

custom_image = cv2.GaussianBlur(image0, ksize=(11, 11), sigmaX=15)

plot_3_images(image0, image2, custom_image)
cv2.imwrite("Our_image_2.jpg", custom_image)


# Image 3 - Median Filter

custom_image = cv2.medianBlur(image0, 11)

plot_3_images(image0, image3, custom_image)
cv2.imwrite("Our_image_3.jpg", custom_image)


# Image 4 - Convolution with a vertical kernel (each row is [0,...0,1,0...,0])

def create_cubic_vertical_kernel(size):
    matrix = np.zeros((size, size))
    matrix[:, size // 2] = 1
    matrix = matrix / np.sum(matrix)
    return matrix

kernel = create_cubic_vertical_kernel(15)

custom_image = cv2.filter2D(image0, -1, kernel)
plot_3_images(image0, image4, custom_image)
cv2.imwrite("Our_image_4.jpg", custom_image)


# Image 5 - sharp_part

custom_image = cv2.GaussianBlur(image0, ksize=(11, 11), sigmaX=15)
sharp_part = image0 - custom_image

plot_3_images(image0, image5, sharp_part+128)
cv2.imwrite("Our_image_5.jpg", sharp_part+128)


# Image 6 - Horizontal edge detection

kernel = np.array([
    [-0.26, -0.52, -0.26],
    [0,        0,      0],
    [0.26,  0.52,   0.26]
    ])

custom_image = cv2.filter2D(image0, -1, kernel)

plot_3_images(image0, image6, custom_image)
cv2.imwrite("Our_image_6.jpg", custom_image)


# Image 7 - Vertical Cyclic Translation

custom_image = np.roll(image0, image0.shape[0] // 2, axis=0)

plot_3_images(image0, image7, custom_image)
cv2.imwrite("Our_image_7.jpg", custom_image)


# Image 8 - Grayscale image

custom_image = image0

plot_3_images(image0, image8, custom_image)
cv2.imwrite("Our_image_8.jpg", custom_image)


# Image 9 - Sharpening

kernel = np.array([
    [0,    -0.585,      0],
    [-0.585, 3.34, -0.585],
    [0,    -0.585,      0]
])

custom_image = cv2.filter2D(image0, -1, kernel)

plot_3_images(image0, image9, custom_image)
cv2.imwrite("Our_image_9.jpg", custom_image)