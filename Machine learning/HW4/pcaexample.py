import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
data = np.random.multivariate_normal(mean, cov, 100)

# Plot original data
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.8)
plt.title("Original Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.axis('equal')
plt.show()

# Center the data
mean_data = np.mean(data, axis=0)
centered_data = data - mean_data

# Calculate covariance matrix
cov_matrix = np.cov(centered_data, rowvar=False)

# Perform eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvectors by eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# Select the first principal component
first_pc = eigenvectors_sorted[:, 0]

# Project data onto the first principal component
projected_data = np.dot(centered_data, first_pc)

# Plot original data with eigenvectors as arrows
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], alpha=0.8)
plt.title("Original Data with Eigenvectors")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.axis('equal')

# Plot eigenvectors as arrows
for i in range(len(eigenvalues)):
    plt.arrow(mean_data[0], mean_data[1], eigenvectors_sorted[0, i] * np.sqrt(eigenvalues[i]),
              eigenvectors_sorted[1, i] * np.sqrt(eigenvalues[i]), color='r', width=0.05, head_width=0.2)

plt.show()

# Plot the projected data
plt.figure(figsize=(6, 4))
plt.scatter(projected_data, np.zeros_like(projected_data), alpha=0.8)
plt.title("Data Projected onto First Principal Component")
plt.xlabel("Principal Component 1")
plt.grid(True)
plt.show()