# Itamar Brotzky, 207931296
# Or Diner, 207035809

# Please replace the above comments with your names and ID numbers in the same format.

import cv2
import numpy as np
import os
import shutil


#matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine: bool):
	"""
	Calculate the transformation from the 1st image to the xth image based on matches.

	Args:
	    matches (numpy.ndarray): An array of shape (N, 2, 2) where each row is a pair of points
	                             (src_point, dst_point), with each point represented as (x, y).
	    is_affine (bool): If True, calculate an affine transformation; otherwise, calculate a projective transformation.

	Returns:
	    numpy.ndarray: The transformation matrix (3x3 for projective, 2x3 for affine).
	"""
	src_points, dst_points = matches[:,0], matches[:,1]

	# Compute the transformation matrix
	if is_affine:
		# Calculate affine transformation matrix
		T, _ = cv2.estimateAffine2D(src_points, dst_points, method=cv2.RANSAC)
	else:
		# Calculate projective transformation matrix (homography)
		T, _ = cv2.findHomography(src_points, dst_points, method=cv2.RANSAC)
	return T


def stitch(img1, img2):
	# Create a binary mask where img2 has non-zero pixels
	mask = (img2 > 0).any(axis=-1)  # Shape: (H, W)

	# Pad the mask to handle boundary conditions
	padded_mask = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)

	# Check the neighbors in the 3x3 region
	neighbor_mask = (
			padded_mask[:-2, :-2] & padded_mask[:-2, 1:-1] & padded_mask[:-2, 2:] &  # Top row
			padded_mask[1:-1, :-2] & padded_mask[1:-1, 1:-1] & padded_mask[1:-1, 2:] &  # Middle row
			padded_mask[2:, :-2] & padded_mask[2:, 1:-1] & padded_mask[2:, 2:]  # Bottom row
	)

	# Combine the neighbor mask with the original mask
	valid_mask = neighbor_mask & mask

	# Use the valid mask to prioritize img2
	stitched_image = np.where(valid_mask[..., None], img2, img1)

	return stitched_image


def inverse_transform_target_image(target_img, original_transform, output_size):
	"""
	Perform the inverse transform to bring the xth image into the 1st image's canvas.

	Args:
		target_img (numpy.ndarray): The xth image.
		original_transform (numpy.ndarray): The transformation matrix from the 1st to xth image.
		output_size (tuple): The output canvas size (width, height).

	Returns:
		numpy.ndarray: The transformed image.
	"""
	if original_transform.shape == (2, 3):  # Affine transformation
		# Convert 2x3 affine to 3x3 by adding [0, 0, 1]
		affine_transform = np.vstack([original_transform, [0, 0, 1]])
		# Invert the 3x3 matrix
		inverse_transform = np.linalg.inv(affine_transform)[:2, :]  # Take first 2 rows back as 2x3
		# Use cv2.warpAffine for affine transformations
		transformed_image = cv2.warpAffine(target_img, inverse_transform,(output_size[1], output_size[0])) # OpenCV uses (width, height)

	else:  # Projective transformation (Homography, 3x3 matrix)
		# Invert the 3x3 homography matrix
		inverse_transform = np.linalg.inv(original_transform).astype(np.float32)
		# Use cv2.warpPerspective for projective transformations
		transformed_image = cv2.warpPerspective(target_img, inverse_transform,(output_size[1], output_size[0])) # OpenCV uses (width, height)

	return transformed_image


# returns list of pieces file names
def prepare_puzzle(puzzle_dir):
	edited = os.path.join(puzzle_dir, 'abs_pieces')
	if os.path.exists(edited):
		shutil.rmtree(edited)
	os.mkdir(edited)
	
	affine = 4 - int("affine" in puzzle_dir)
	
	matches_data = os.path.join(puzzle_dir, 'matches.txt')
	n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

	matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images-1,affine,2,2)
	
	return matches, affine == 3, n_images


if __name__ == '__main__':
	lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']

	for puzzle_dir in lst:
		print(f'Starting {puzzle_dir}')
		
		puzzle = os.path.join('puzzles', puzzle_dir)
		
		pieces_pth = os.path.join(puzzle, 'pieces')
		edited = os.path.join(puzzle, 'abs_pieces')
		
		matches, is_affine, n_images = prepare_puzzle(puzzle)
		# Add your code here

		# Get all image file paths in the folder
		image_files = sorted(
			[os.path.join(pieces_pth, f) for f in os.listdir(pieces_pth) if f.endswith(('.png', '.jpg', '.jpeg'))])

		# Ensure the number of images matches the given n_images
		if len(image_files) != n_images:
			raise ValueError("Number of images in the folder does not match n_images.")

		# Load the first image (canvas) and initialize the final image
		first_image = cv2.imread(image_files[0])
		
		# Creates new black canvas
		canvas = np.zeros(first_image.shape, dtype=np.uint8)

		# Place the first image onto the canvas
		canvas[:first_image.shape[0], :first_image.shape[1]] = first_image

		# Save the absolute piece in its folder
		cv2.imwrite(os.path.join(edited, "piece_1_absolute.jpg"), first_image)

		# Process and stitch each subsequent image
		for i in range(1, n_images):
			# Load the target image
			target_img = cv2.imread(image_files[i])

			# Get the matches for the current pair (1st, xth)
			current_matches = np.array(matches[i - 1])  # Matches for (1st, i+1) pair

			# Compute the transformation from the 1st image to the current image
			transform = get_transform(current_matches, is_affine)

			# Inverse-transform the current image onto the canvas
			transformed_target = inverse_transform_target_image(target_img, transform, first_image.shape[0:2])

			# Resize the transformed target to match the canvas size
			transformed_target_resized = cv2.resize(transformed_target, (canvas.shape[1], canvas.shape[0]))

			# Save the absolute piece in its folder
			cv2.imwrite(os.path.join(edited, f"piece_{i+1}_absolute.jpg"), transformed_target_resized)

			# Stitch the transformed target image onto the canvas
			canvas = stitch(canvas, transformed_target_resized)

		sol_file = f'solution.jpg'
		cv2.imshow("Final Stitched Image", canvas)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		cv2.imwrite(os.path.join(puzzle, sol_file), canvas)