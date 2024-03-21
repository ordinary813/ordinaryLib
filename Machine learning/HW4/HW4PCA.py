import numpy as np
from matplotlib import pyplot as plt
import os
import cv2

import warnings
warnings.filterwarnings('ignore')

def proccess_data(folder):
  image_arrays = []
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_arrays.append(gray_image)
  return np.array(image_arrays)

print(os.listdir())

smile = proccess_data('./Machine learning/HW4/kaggle/data/smile')
non_smile = proccess_data('./Machine learning/HW4/kaggle/data/non_smile')

dataset = np.vstack((smile,non_smile))
labels = np.concatenate((np.ones(smile.shape[0]),np.zeros(non_smile.shape[0])))

# plt.subplot(121)
# plt.title("Smile")
# plt.imshow(smile[0], cmap='gray')

# plt.subplot(122)
# plt.title("Not smile")
# plt.imshow(non_smile[0], cmap='gray')

# plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size = 0.2, stratify=labels, random_state=42)

# print(f'train size is {x_train.shape} and labels size is {y_train.shape}')
# print(f'test size is {x_test.shape} and labels size is {y_test.shape}')
# print()

x_train_flatten = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test_flatten = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

# print(f'flattened train size is {x_train_flatten.shape} ')
# print(f'flattened test size is {x_test_flatten.shape}')

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# shrinks the data matrix to have k features
def PCA_train(data, k):
    flatten_data = data.reshape(np.shape(data)[0],-1)
    # array of feature means
    flattened_mean = np.mean(flatten_data, axis=0)
    mu = np.mean(data, axis=0)
    # center data
    Z = flatten_data - flattened_mean

    # calculate the scatter matrix
    scatter_matrix = np.dot(Z.transpose(), Z)
    eigenvals, eigenvecs = np.linalg.eig(scatter_matrix)

    # find the best k components
    best_k_indices = np.argsort(eigenvals)[::-1][:k ** 2]
    E = eigenvecs[:, best_k_indices].transpose()
    y = np.dot(E, Z.transpose()).transpose()
    y = y.reshape(np.shape(y)[0], k, k)
    return y.astype(float), mu, E
    
def PCA_test(test, mu, E):
    flatten_test = test.reshape(np.shape(test)[0],-1)
    # array of feature means
    flattened_mean = np.mean(flatten_test, axis=0)
    Z = flatten_test - flattened_mean
    y = np.dot(E, Z.transpose()).transpose()
    y = y.reshape(np.shape(y)[0], int(np.sqrt(np.shape(E)[0])), int(np.sqrt(np.shape(E)[0])))           # basically |samples|-k-k
    return y.astype(float)

def recover_PCA(data, mu, E):
    recovered = np.dot(E.transpose(), data.transpose()).transpose() + mu
    return recovered.astype(float)

x_train_new, mu, eig = PCA_train(x_train, k=9)
x_test_new = PCA_test(x_test, mu, eig)

plt.subplot(131)
plt.title("Original Image")
plt.imshow(x_train[1], cmap='gray')

plt.subplot(132)
plt.title("Image in lower dimension")
plt.imshow(x_train_new[1], cmap='gray')

# plt.subplot(133)
# plt.title("Recovered Image")
# plt.imshow(None, cmap='gray')

plt.show()