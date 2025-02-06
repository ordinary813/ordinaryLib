import numpy as np

# import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# matplotlib.use('TkAgg')

# Function to compute the Fourier transform of an image
def compute_fourier(image):
    f_transform = np.fft.fftshift(np.fft.fft2(image))
    magnitude_spectrum = np.log(np.abs(f_transform) + 1)
    return magnitude_spectrum


im = np.zeros((256, 256))
rows, cols = im.shape
crows, ccols = rows // 2, cols // 2
im[:, ccols] = 1
im[:, ccols + 1] = 1
im[:, ccols - 1] = 1

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].set_title('Original Image')
img_display = ax[0].imshow(im, cmap='gray', vmin=0, vmax=1)

ax[1].set_title('Magnitude Spectrum Fourier')
magnitude_spectrum = compute_fourier(im)
fourier_display = ax[1].imshow(magnitude_spectrum, cmap='gray')


def update(i):
    global im
    updateColumn(i, im)

    img_display.set_array(im)
    magnitude_spectrum = compute_fourier(im)
    fourier_display.set_array(magnitude_spectrum)

    return img_display, fourier_display

def updateColumn(i, im):
    if(i < rows//2):
        im[i, ccols] = 0
        im[rows - i - 1, ccols] = 0

        im[i, ccols + 1] = 0
        im[rows - i - 1, ccols + 1] = 0

        im[i, ccols - 1] = 0
        im[rows - i - 1, ccols - 1] = 0

ani = animation.FuncAnimation(fig, update, frames=len(im)//2 + 10, interval=50 ,blit=False)
ani.save('fourier.gif', writer='imagemagick', fps=16)