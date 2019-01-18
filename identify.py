import numpy as np
import cv2
import os
import sys

def create_basis(root, file_paths, num_columns):
	global SIZE
	if (len(file_paths) == 0):
		return []

	try:
		img_book = np.zeros((SIZE**2, len(file_paths)))
		counter = 0
		for path in file_paths:
			if path.endswith(".jpg") or path.endswith(".png"):
				img = img_to_column_matrix(root + "/" + path)
				img_book[:,counter] = np.subtract(img, np.average(img))
				counter = counter + 1

		img_book = img_book[:, :counter-1]
		
		u, s, vt = np.linalg.svd(img_book, full_matrices=False)

		if u.shape[1] < num_columns:
			return_shape = np.zeros((u.shape[0], num_columns))
			return_shape[:,:u.shape[1]] = u
			return return_shape

		return u[:, :num_columns]

	except Exception as e:
		print (path, e)
	
	return []	

def img_to_column_matrix(filename):
	global SIZE
	"""
	Converts an image to 100x100 size and then to a 10000x1 array with values ranging from 0 to 1.
	"""
	img = cv2.imread(filename, 0)
	width, height = img.shape
	if width > height:
		img = cv2.resize(img, (0, 0), fx = SIZE / height, fy = SIZE / height)
	else:
		img = cv2.resize(img, (0, 0), fx = SIZE / width, fy = SIZE / width)
	img = np.true_divide(img[:SIZE,:SIZE].reshape(SIZE**2), 255)
	return img

def dist_to_basis(test_img, basis):
	"""
	Returns the distance between an image array and the image array's projection onto the item's basis.
	"""
	return np.linalg.norm(test_img - np.matmul(np.matmul(basis, np.transpose(basis)), test_img))


def create_basis_library(directory):
	global item_labels, NUM_COLUMNS, SIZE
	"""
	Reads all the images, gets the bases, and returns them as a massive numpy array.
	"""	
	# Create the bases for all the images

	counter = 0
	for root, dirs, files in os.walk(directory):
		counter = counter + 1

	basis_library = np.zeros((SIZE**2, counter*NUM_COLUMNS))
	success_count = 0
	for root, dirs, files in os.walk(directory):
		basis_img = create_basis(root, files, NUM_COLUMNS)
		if len(basis_img) > 0:
			item_labels.append(root)
			print (root)
			basis_library[:, success_count:(success_count+NUM_COLUMNS)] = basis_img
			success_count = success_count + NUM_COLUMNS

	basis_library = basis_library[:, :success_count]
	return basis_library

def test_images(test_imgs, basis_library):
	# Use the arguments as comparison images.
	global item_labels, NUM_COLUMNS

	return_results = []
	for img_path in test_imgs:
		img = img_to_column_matrix(img_path)

		min_item = ""
		min_dist = 100000000
		for i in range(0, int(basis_library.shape[1]/NUM_COLUMNS)):
			basis = basis_library[:, i*NUM_COLUMNS:(i+1)*NUM_COLUMNS]
			item_dist = dist_to_basis(img, basis)
			if item_dist < min_dist:
				min_dist = item_dist
				min_item = item_labels[i]

		return_results.append((min_item, min_dist))
	return return_results



NUM_COLUMNS = 4
SIZE = 28
item_labels = []

if sys.argv[1] == "load":
	basis_library = np.load("./library.npy")
	item_labels = np.load("./labels.npy")
else:
	basis_library = create_basis_library(sys.argv[1])
	np.save("./library.npy", basis_library)
	np.save("./labels.npy", item_labels)

print ("Completed the library. Comparing the image...")

print (test_images(sys.argv[2:], basis_library))