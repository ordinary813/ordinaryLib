# Itamar Brotzky, 207931296
# Or Diner, 207035809

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import matplotlib.pyplot as plt
import numpy as np


def brightness_contrast_stretching(image, alpha=1.0, beta=0):
	"""
    Adjusts the brightness and contrast of an image using stretching.

    Parameters:
        image (numpy.ndarray): The input image read using cv2.imread().
        alpha (float): Contrast factor. (1.0 = no change, >1.0 increases contrast).
        beta (int): Brightness adjustment. (0 = no change, >0 brightens, <0 darkens).

    Returns:
        numpy.ndarray: The brightness and contrast-stretched image.
    """
	# Ensure the image is valid
	if image is None:
		raise ValueError("The input image is None. Please provide a valid image.")

	# Apply brightness and contrast adjustment
	stretched_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

	cv2.imshow("Original Image", image)
	cv2.imshow(f"Gamma Corrected (alph={alpha}, beta={beta})", stretched_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return stretched_image


def gamma_correction(image, gamma):
	"""
    Performs gamma correction on an image.

    Parameters:
        image (numpy.ndarray): The input image read using cv2.imread().
        gamma (float): The gamma value for correction (gamma > 0).
                      Values < 1 brighten the image, and values > 1 darken it.

    Returns:
        numpy.ndarray: The gamma-corrected image.
    """
	# Ensure gamma is positive
	if gamma <= 0:
		raise ValueError("Gamma value must be greater than 0.")

	# Build a lookup table to map pixel values [0, 255] based on the gamma value
	inv_gamma = 1.0 / gamma
	lookup_table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")

	# Apply the gamma correction using the lookup table
	corrected_image = cv2.LUT(image, lookup_table)

	cv2.imshow("Original Image", image)
	cv2.imshow(f"Gamma Corrected (γ={gamma})", corrected_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return corrected_image


def histogram_equalization(image):
	"""
    Performs histogram equalization on an image.

    Parameters:
        image (numpy.ndarray): The input image read using cv2.imread().

    Returns:
        numpy.ndarray: The image after histogram equalization.
    """
	# Check if the image is grayscale or color
	if len(image.shape) == 2:  # Grayscale image
		equalized_image = cv2.equalizeHist(image)
	elif len(image.shape) == 3:  # Color image (BGR)
		# Split the image into its channels
		b, g, r = cv2.split(image)

		# Equalize the histogram of each channel separately
		b_eq = cv2.equalizeHist(b)
		g_eq = cv2.equalizeHist(g)
		r_eq = cv2.equalizeHist(r)

		# Merge the equalized channels back into a color image
		equalized_image = cv2.merge((b_eq, g_eq, r_eq))
	else:
		raise ValueError("Unsupported image format.")

	cv2.imshow("Original Image", image)
	cv2.imshow(f"equalized_image", equalized_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return equalized_image


def apply_fix(image, id):
	# Your code goes here
	if id == 1:
		return histogram_equalization(image)
	elif id == 2:
		return gamma_correction(image, gamma=1.4)
	elif id == 3:
		return brightness_contrast_stretching(image, alpha=0.9, beta=15)
	else:
		return image


for i in range(1,4):
	if i == 1:
		path = f'{i}.png'
	else:
		path = f'{i}.jpg'
	image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	if image is None:
		print("Wrong image")
		continue
	fixed_image = apply_fix(image, i)
	plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray', vmin=0, vmax=255)
