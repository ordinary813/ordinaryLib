import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
	img_size = imgs_arr[0].shape
	res = []
	
	for img in imgs_arr:
		X = img.reshape(img_size[0] * img_size[1], 1)
		km = KMeans(n_clusters=n_colors)
		km.fit(X)
		
		img_compressed = km.cluster_centers_[km.labels_]
		img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

		res.append(img_compressed.reshape(img_size[0], img_size[1]))
	
	return np.array(res)

# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
	image_arrays = []
	lst = [file for file in os.listdir(folder) if file.endswith(formats)]
	for filename in lst:
		file_path = os.path.join(folder, filename)
		image = cv2.imread(file_path)
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		image_arrays.append(gray_image)
	return np.array(image_arrays), lst

# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
	# Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values will work
	x_pos = 70 + 40 * idx
	y_pos = 274
	while image[y_pos, x_pos] == 0:
		y_pos -= 1
	return 274 - y_pos

# Sections c, d
# Remember to uncomment compare_hist before using it!

#a
images, names = read_dir('data')
numbers, numberNames = read_dir('numbers')

img = images[0]

cv2.imshow(names[0], img) 
cv2.waitKey(0)
cv2.destroyAllWindows() 
# exit()

#b
img = numbers[0]

cv2.imshow(numberNames[0], img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
# exit()

#c

#  Input: 2 accumulated histograms
# Output: their EMD
def calculate_EMD(C_a, C_b):
	return np.sum(np.abs(C_a - C_b))

#  Input: source image and a target to find in it
# Output: wether the target's histogram matches a window of the same size in the source image 
def compare_hist(src_image, target):
	target_hist = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
	target_cumsum = np.cumsum(target_hist, dtype = np.float32)

	height, width = target.shape
	windows = np.lib.stride_tricks.sliding_window_view(src_image, (height, width))

	# offset range of the windows
	# x and y are the actual offset
	# we only check specific windows where the top number appears
	hh, ww = windows.shape[:2]
	for x in range(25, 50):
		for y in range(100, 145):
			window = windows[y, x]
			window_hist = cv2.calcHist([window], [0], None, [256], [0, 256]).flatten()
			window_cumsum = np.cumsum(window_hist, dtype = np.float32)

			emd = calculate_EMD(target_cumsum, window_cumsum)
			if emd < 260:
				return True
	return False

#  Input: source image and the numbers image array
# Output: an integer of the top most number on the scale
def top_num_on_scale(src_image, numbers):
    i = len(numbers) - 1
    while i >= 0:
        res = compare_hist(src_image, numbers[i])
        if res is True:
            return i
        i -= 1
    return -1

#d
top_nums = np.zeros(len(images))
top_nums -= 1

for i in range(len(images)):
    src_img = images[i]
    res = top_num_on_scale(src_img, numbers)
    print(f'Histogram {names[i]}\'s top number: {res}')
    top_nums[i] = res

#e
quantisized_imgs = quantization(images, 3)

threshold = 215
binary_imgs = []

for img in quantisized_imgs:
    if len(img.shape) == 3:  # check image dimensions
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    binary_imgs.append(binary_img)

binary_imgs = np.array(binary_imgs)

cv2.imshow("Thresholded Image", binary_imgs[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
# exit()

#f
#  Input: binary image
# Output: bar heights list
def get_image_counts(bin_image):
    bars = np.zeros([10])
    for i in range(len(bars)):
        bars[i] = get_bar_height(bin_image, i)
    return bars

#g
for i, bin_img in enumerate(binary_imgs):
    bars = get_image_counts(bin_img)
    max_height = np.max(bars)
    top_num = top_num_on_scale(images[i], numbers)
    counts = [int(np.round(top_num * bar_height / max_height)) for bar_height in bars]

    print(f'Histogram {names[i]} gave {counts}')


