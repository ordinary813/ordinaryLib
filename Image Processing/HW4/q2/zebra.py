# Itamar Brotzky, 207931296
# Or Diner, 207035809
# Please replace the above comments with your names and ID numbers in the same format.


import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Your code goes here

# Original Image Dimensions
H, W = image.shape

# Fourier Transform and Magnitude Spectrum of Original Image
F = fft2(image)
F_shifted = fftshift(F)
F_magnitude = np.log(1 + np.abs(F_shifted))

# Part B: Scaling with Zero Padding (2H x 2W)
padded_F = np.zeros((2 * H, 2 * W), dtype=np.complex128)
padded_F[H//2:H + H//2, W//2:W + W//2] = F_shifted
padded_F_shifted = ifftshift(padded_F)
padded_image = np.abs(ifft2(padded_F_shifted))
padded_F_magnitude = np.log(1 + np.abs(padded_F))

# Part C: Resizing Using Fourier Scaling Formula
scaled_F = np.zeros((2 * H, 2 * W), dtype=np.complex128)

# Map frequencies using Fourier scaling formula
for u in range(0, 2*H):
    for v in range(0, 2*W):
        shifted_u = u%H
        shifted_v = v%W
        if u%2 == 1 or v%2 == 1:
            scaled_F[u, v] = 0
        else:
            scaled_F[u, v] = F_shifted[u // 2, v // 2]



scaled_F_shifted = ifftshift(scaled_F)
scaled_image = np.abs(ifft2(scaled_F_shifted))
scaled_F_magnitude = np.log(1 + np.abs(scaled_F))

# Display Results
plt.figure(figsize=(10,10))
plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(F_magnitude, cmap='gray')

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(padded_F_magnitude, cmap='gray')

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(padded_image, cmap='gray')

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(scaled_F_magnitude, cmap='gray')

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(scaled_image, cmap='gray')
plt.show()