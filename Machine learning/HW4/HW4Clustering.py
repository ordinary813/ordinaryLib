import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# class Kmeans:

# 	def __init__(self, n_clusters, max_iter=100, random_state=123):
# 		self.n_clusters = n_clusters
# 		self.max_iter = max_iter
# 		self.random_state = random_state

# 	def initialize_centroids(self, X):
# 		np.random.RandomState(self.random_state)
# 		random_idx = np.random.permutation(X.shape[0])
# 		centroids = X[random_idx[:self.n_clusters]]
# 		return centroids

# 	def reassign_centroids(self, X, labels):
# 		centroids = np.zeros((self.n_clusters, X.shape[1]))
# 		# Implement here
# 		return centroids

# 	def compute_distance(self, X, centroids):
# 		distance = np.zeros((X.shape[0], self.n_clusters))
# 		for k in range(self.n_clusters):
# 			row_norm = np.linalg.norm(X - centroids[k, :], axis=1)
# 			distance[:, k] = np.square(row_norm)
# 		return distance

# 	def find_closest_cluster(self, distance):
# 		return np.argmin(distance, axis=1)

# 	def compute_sse(self, X, labels, centroids):
# 		distance = np.zeros(X.shape[0])
# 		for k in range(self.n_clusters):
# 			distance[labels == k] = np.linalg.norm(X[labels == k] - centroids[k], axis=1)
# 		return np.sum(np.square(distance))

# 	def fit(self, X):
# 		self.centroids = self.initialize_centroids(X)
# 		for i in range(self.max_iter):
# 			old_centroids = self.centroids
# 			# For each point, calculate distance to all k clustes.
# 			self.labels =	# Assign the labels with closest distance' cluster.
# 			self.centroids = # Update the centroids
# 			if np.all(old_centroids == self.centroids):
# 				break
# 		self.error = self.compute_sse(X, self.labels, self.centroids)

# 	def predict(self, X):
# 		distance = self.compute_distance(X, self.centroids)
# 		return self.find_closest_cluster(distance)
	

data = pd.read_csv("https://sharon.srworkspace.com/ml/datasets/hw4/exams.csv", header=None)
data = data.to_numpy()

x = data[:,:-1]
y = data[:,-1]

zero_labeled = x[y == 0]
one_labeled = x[y == 1]

# Scatter plot for label 0 (red)
plt.scatter(zero_labeled[:, 0], zero_labeled[:, 1], color='red', label='y = 0', s=15)

# Scatter plot for label 1 (blue)
plt.scatter(one_labeled[:, 0], one_labeled[:, 1], color='blue', label='y = 1', s=15)

# Set labels for each axis
plt.xlabel('X1')
plt.ylabel('X2')

# Show legend
plt.legend()

# Show the plot
plt.show()