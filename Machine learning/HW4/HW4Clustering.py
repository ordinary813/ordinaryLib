import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class Kmeans:

	def __init__(self, n_clusters, max_iter=100, random_state=123):
		self.n_clusters = n_clusters
		self.max_iter = max_iter
		self.random_state = random_state

	def initialize_centroids(self, X):
		np.random.RandomState(self.random_state)
		random_idx = np.random.permutation(X.shape[0])
		centroids = X[random_idx[:self.n_clusters]]
		return centroids

	def reassign_centroids(self, X, labels):
		centroids = np.zeros((self.n_clusters, X.shape[1]))
		for k in range(self.n_clusters):
			centroids[k] = np.mean(X[labels == k], axis=0)
		return centroids

	def compute_distance(self, X, centroids):
		distance = np.zeros((X.shape[0], self.n_clusters))
		for k in range(self.n_clusters):
			row_norm = np.linalg.norm(X - centroids[k, :], axis=1)
			distance[:, k] = np.square(row_norm)
		return distance

	def find_closest_cluster(self, distance):
		return np.argmin(distance, axis=1)

	def compute_sse(self, X, labels, centroids):
		distance = np.zeros(X.shape[0])
		for k in range(self.n_clusters):
			distance[labels == k] = np.linalg.norm(X[labels == k] - centroids[k], axis=1)
		return np.sum(np.square(distance))

	def fit(self, X):
		self.centroids = self.initialize_centroids(X)
		for i in range(self.max_iter):
			old_centroids = self.centroids
			distance = self.compute_distance(X, self.centroids)
			# For each point, calculate distance to all k clustes.
			self.labels = self.find_closest_cluster(distance)
			self.centroids = self.reassign_centroids(X, self.labels)
			if np.all(old_centroids == self.centroids):
				break
		self.error = self.compute_sse(X, self.labels, self.centroids)

	def predict(self, X):
		distance = self.compute_distance(X, self.centroids)
		return self.find_closest_cluster(distance)
	

data = pd.read_csv("https://sharon.srworkspace.com/ml/datasets/hw4/exams.csv", header=None)
data = data.to_numpy()

x = data[:,:-1]
y = data[:,-1]

zero_labeled_samples = x[y == 0]
one_labeled_samples = x[y == 1]

# plt.scatter(zero_labeled_samples[:, 0], zero_labeled_samples[:, 1], color='red', label='Label 0', s=15)

# plt.scatter(one_labeled_samples[:, 0], one_labeled_samples[:, 1], color='blue', label='Label 1', s=15)

# plt.xlabel('X1')
# plt.ylabel('X2')

# plt.legend()
# plt.show()

n_clusters = 2
clust = Kmeans(n_clusters)

clust.fit(x)

labels = clust.labels
centroids = clust.centroids

# c0 = x[labels == 0]
# c1 = x[labels == 1]

# plt.scatter(c0[:, 0], c0[:, 1], c='green', label='cluster 1')
# plt.scatter(c1[:, 0], c1[:, 1], c='blue', label='cluster 2')
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='centroid')
# plt.legend()

# plt.show()


sse = []
list_k = list(range(1, 11))

for k in list_k:
    # Initialize and fit KMeans model
    clust = Kmeans(n_clusters=k)
    clust.fit(data[:, :2])  # Use only first two columns as features
    sse.append(clust.error)  # Append SSE for current clustering

# Plot sse against k
# plt.figure(figsize=(6, 6))
# plt.plot(list_k, sse, '-o')
# plt.xlabel(r'Number of clusters ($k$)')
# plt.ylabel('Sum of squared distance')
# plt.title('Elbow Method for Optimal k')
# plt.xticks(list_k)
# plt.grid(True)
# plt.show()


n_clusters = 5
clust = Kmeans(n_clusters)

clust.fit(x)

labels = clust.labels
centroids = clust.centroids

# c0 = x[clust.labels == 0]
# c1 = x[clust.labels == 1]
# c2 = x[clust.labels == 2]
# c3 = x[clust.labels == 3]
# c4 = x[clust.labels == 4]

# plt.scatter(c0[:, 0], c0[:, 1], c='gold', label='cluster 1')
# plt.scatter(c1[:, 0], c1[:, 1], c='red', label='cluster 2')
# plt.scatter(c2[:, 0], c2[:, 1], c='brown', label='cluster 3')
# plt.scatter(c3[:, 0], c3[:, 1], c='orange', label='cluster 4')
# plt.scatter(c4[:, 0], c4[:, 1], c='orange', label='cluster 5')
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='centroid')
# plt.legend()

# plt.show()

import urllib.request

def read_image(url):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    return cv2.imdecode(arr, -1)

from sklearn.cluster import KMeans
import cv2

img = read_image('https://sharon.srworkspace.com/ml/datasets/hw4/image.jpg')
img_size = img.shape

#Reshape it to be 2-dimension
X = img.reshape(img_size[0] * img_size[1], img_size[2])		# Turn hxwx3 into (h*w)x3

# Run the Kmeans algorithm
km = KMeans(n_clusters=20)
km.fit(X)

'''
The km has the following properties:
(*) km.labels_ is an array size (pixels, 20), will give each pixel its class from 20 classes (values are between 0-19)
(*) km.cluster_centers_ is an array size 20x3, where the ith row represents the color value for the ith label.
	For example, cluster_centers_[0] = [r,g,b], the first center.
'''

# Use the centroids to compress the image
img_compressed = km.cluster_centers_[km.labels_]
img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

# Reshape X_recovered to have the same dimension as the original image 128 * 128 * 3'''
img_compressed = img_compressed.reshape(img_size[0], img_size[1], img_size[2])

# Plot the original and the compressed image next to each other'''
fig, ax = plt.subplots(1, 2, figsize = (12, 8))

ax[0].imshow(img)
ax[0].set_title('Original Image')

ax[1].imshow(img_compressed)
ax[1].set_title(f'Compressed Image with {km.n_clusters} colors')

for ax in fig.axes:
    ax.axis('off')

plt.tight_layout()
plt.show()