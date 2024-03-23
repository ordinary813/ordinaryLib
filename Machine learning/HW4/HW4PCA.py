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
  mu = np.mean(data, axis=0)
  Z = data - mu

  # flatten the centered array to 962x4096 instead of 962x64x64
  # flatten the mean array
  Z = flatten(Z)
  mu = flatten(mu)

  # the scatter matrix is essentialy the covariance matrix
  scatter_matrix = np.cov(Z, rowvar=False)
  # since cov matrix is symettric I'll use eigh instead of eig
  eigenvals, eigenvecs = np.linalg.eigh(scatter_matrix)

  best_k_indices = np.argsort(eigenvals)[::-1][:k]
  # E is the best k eigen vectors (k = 81)
  E = eigenvecs[:, best_k_indices]
  best_k_eigenvals = eigenvals[best_k_indices]

  y = np.matmul(Z, E)
  # return an array of size |samples| x (k), flattened mean array, k best eigen vectors
  return y, mu, E , best_k_eigenvals

def PCA_test(test, mu, E):
  flatten_test = flatten(test)
  Z = flatten_test - mu
  y = np.matmul(Z, E)
  return y

def recover_PCA(data, mu, E):
  recovered = np.matmul(E, data) + mu
  recovered = unflatten(recovered)
  return recovered.astype(float)

# gets a cubic array and flattens it so we 
# can work  with features instead of pixels
def flatten(data):
  if(data.ndim == 2):
    return data.reshape(-1)
  if(data.ndim == 3):
    return data.reshape(np.shape(data)[0], -1)
    
# gets a flat array and makes it a cubic array
def unflatten(data):
  if(data.ndim == 2):
    return data.reshape(np.shape(data)[0], np.sqrt(np.shape(data)[1]).astype(int), np.sqrt(np.shape(data)[1]).astype(int))
  if(data.ndim == 1):
    return data.reshape(np.sqrt(np.shape(data)[0]).astype(int), np.sqrt(np.shape(data)[0]).astype(int))


x_train_new, mu, eig, eig_val = PCA_train(x_train, k=81)
x_test_new = PCA_test(x_test, mu, eig)

plt.subplot(131)
plt.title("Original Image")
plt.imshow(x_train[3], cmap='gray')

plt.subplot(132)
plt.title("Image in lower dimension")
plt.imshow(unflatten(x_train_new[3]), cmap='gray')

plt.subplot(133)
plt.title("Recovered Image")
plt.imshow(recover_PCA(x_train_new[3], mu, eig), cmap='gray')

plt.show()

def EIG_CDF(eig_list):
  sorted_eigenvalues = np.sort(eig_list)[::-1]
  eigenvalues_cumsum = np.cumsum(sorted_eigenvalues)
  eigenvalues_cumsum_normalized = eigenvalues_cumsum / np.sum(sorted_eigenvalues) # eigenvalues_cumsum[-1]

  amount = np.argmax(eigenvalues_cumsum_normalized >= 0.95) + 1

  plt.plot(np.arange(1, len(sorted_eigenvalues)+1), eigenvalues_cumsum_normalized)
  plt.xlabel('Principal Component')
  plt.ylabel('Cumulative Proportion of Variance')
  plt.title(f'CDF of Eigenvalues - {amount} eigs preserves 95% of enetry')
  plt.show()

EIG_CDF(eig_val)


x_train_new, mu, eig, eig_val = PCA_train(x_train, k=49)
x_test_new = PCA_test(x_test, mu, eig)

plt.subplot(131)
plt.title("Original Image")
plt.imshow(x_train[3], cmap='gray')

plt.subplot(132)
plt.title("Image in lower dimension")
plt.imshow(unflatten(x_train_new[3]), cmap='gray')

plt.subplot(133)
plt.title("Recovered Image")
plt.imshow(recover_PCA(x_train_new[3], mu, eig), cmap='gray')

plt.show()

from sklearn.neighbors import KNeighborsClassifier

accs = []
# ks that are x^2 and are up until 52 (49)
ks = [i ** 2 for i in range(1,8)]
for k in ks:
    x_train_new, _, _, _ = PCA_train(x_train, k)
    knn = KNeighborsClassifier(n_neighbors=k)
    avg_accuracy = cross_val_score(knn, x_train_new, y_train, cv=5).mean()
    accs.append(avg_accuracy)

plt.figure(figsize=(14,5))
plt.plot(ks, accs)
plt.xlabel('k')
plt.xticks(ks)
plt.ylabel('avg accuracy')
plt.show()


knn = KNeighborsClassifier(n_neighbors=49)
knn.fit(x_train_new, y_train)
y_pred = knn.predict(x_test_new)
acc = np.mean(y_test == y_pred)
print(f'acc on test is {acc}')