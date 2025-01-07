# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt


def clean_baby(im):
	# Your code goes here

def clean_windmill(im):
	# Your code goes here
	# Compute the 2D Fourier Transform of the image
	f = fft2(im)
	fshift = fftshift(f)
	
	# Get the magnitude spectrum
	magnitude_spectrum = np.log(np.abs(fshift) + 1)
	
	# Display the magnitude spectrum
	plt.figure(figsize=(10, 10))
	plt.imshow(magnitude_spectrum, cmap='gray')
	plt.title('Magnitude Spectrum')
	plt.show()
	
	# Create a mask to remove the noise
	rows, cols = im.shape
	crow, ccol = rows // 2 , cols // 2
	
	# Masking the specific angle with high frequency noise
	mask = np.ones((rows, cols), np.uint8)
	mask[crow-10:crow+10, ccol-10:ccol+10] = 0
	
	# Apply the mask to the shifted Fourier transform
	fshift = fshift * mask
	
	# Inverse Fourier Transform to get the cleaned image
	f_ishift = ifftshift(fshift)
	img_back = ifft2(f_ishift)
	img_back = np.abs(img_back)
	
	return img_back

def clean_watermelon(im):
	# Your code goes here

def clean_umbrella(im):
	# Your code goes here

def clean_USAflag(im):
	# Your code goes here

def clean_house(im):
	# Your code goes here

def clean_bears(im):
	# Your code goes here


