import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D
import requests
from io import BytesIO

def load_npy_file(url):
    response = requests.get(url)
    if response.status_code == 200:
        npy_data = np.load(BytesIO(response.content), allow_pickle=True).item()
        return npy_data
    else:
        return None

# Load the dictionary
data_dict = load_npy_file('https://sharon.srworkspace.com/ml/datasets/hw3/linreg_data_2d.npy')

# Access the data as needed
x_train = data_dict['x_train']
y_train = data_dict['y_train']
x_test = data_dict['x_test']
y_test = data_dict['y_test']


def kernel(xi, xj, sigma):
  return np.exp(-(np.linalg.norm(xi-xj) ** 2)/ (2 * (sigma ** 2)))

def prepear_kernel_matrix(train, sigma):
  K = np.zeros((len(train), len(train)))
  for i in range(len(train)):
    for j in range(len(train)):
      K[i,j] = kernel(train[i], train[j], sigma)
  return K

def get_alphas(kernel, target, lamda=0.01):
  return (np.linalg.inv(kernel + lamda* np.identity(np.shape(kernel)[0])).dot(target))

def predict(alphas, train, test, sigma):
  pred = 0
  for i, rowTrain in enumerate(train):
    pred += alphas[i] * kernel(train[i], test, sigma)
  return pred

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=20, random_state=42)

best_acc = 0
best_sigma = None
best_lamda = None

# perform 100 random searches of pairs of lambda and sigma
for i in range(100):
    lamda = np.random.choice(np.linspace(0, 20, 1000))
    sigma = np.random.choice(np.linspace(0, 20, 1000))

    # this is equivalent of saying "if sigma is 0 - regenerate the pair of hyperparameters"
    if(sigma == 0):
        i -= 1
        continue

    kernalMat = prepear_kernel_matrix(x_train, sigma)
    alphas = get_alphas(kernalMat, y_train, lamda)
    predictions = [predict(alphas, x_train, valSample, sigma) for valSample in x_val]

    # get accuracy
    acc = np.mean(np.array(predictions) == np.array(y_val))

    if(acc > best_acc):
        best_acc = acc
        best_lamda = lamda
        best_sigma = sigmax_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=20, random_state=42)

best_acc = 0
best_sigma = None
best_lamda = None

# perform 100 random searches of pairs of lambda and sigma
for i in range(100):
    lamda = np.random.choice(np.linspace(0, 20, 1000))
    sigma = np.random.choice(np.linspace(0, 20, 1000))

    # this is equivalent of saying "if sigma is 0 - regenerate the pair of hyperparameters"
    if(sigma == 0):
        i -= 1
        continue

    kernalMat = prepear_kernel_matrix(x_train, sigma)
    alphas = get_alphas(kernalMat, y_train, lamda)
    predictions = [predict(alphas, x_train, valSample, sigma) for valSample in x_val]

    # get accuracy
    acc = np.mean(np.array(predictions) == np.array(y_val))

    if(acc > best_acc):
        best_acc = acc
        best_lamda = lamda
        best_sigma = sigma

x_train =  np.concatenate((x_train, x_val), axis=0)
y_train =  np.concatenate((y_train, y_val), axis=0)

kernalMat = prepear_kernel_matrix(x_train, best_sigma)
alphas = get_alphas(kernalMat, y_train, best_lamda)

mse = 0
for idx, samp in enumerate(x_train):
  mse += (predict(alphas, x_train, samp, sigma=4) - y_train[idx]) ** 2
mse = mse / len(x_train)
print(f'train mse is {mse}')

mse = 0
for idx, samp in enumerate(x_test):
  mse += (predict(alphas, x_train, samp, sigma=4) - y_test[idx]) ** 2
mse = mse / len(x_test)
print(f'test mse is {mse}')