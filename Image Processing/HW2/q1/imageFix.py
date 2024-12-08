# Student_Name1, Student_ID1
# Student_Name2, Student_ID2

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import matplotlib.pyplot as plt
import numpy as np

def apply_fix(image, id):
	if id == 1:

	elif id == 2:

	elif id == 3:

	else
		print('Invalid image id.')


for i in range(1,4):
	path = f'{i}.jpg'
	image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	fixed_image = apply_fix(image, i)
	plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray', vmin=0, vmax=255)
			