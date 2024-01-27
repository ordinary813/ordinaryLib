import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# This function gets a test set and a training set (both are dataframes with the same features)
# it returns a matrix of dimensions (sizeof_test x sizeof_train) which contains the distance
# between samples in TRAIN and TEST.
# In euclidean distance a distance between two samples is the following:
# SQUARE ROOT of the SUM of all square SUBTRACTION between corresponding
# X's (features), meaning we subtract color with color, spectral class with spectral class and
# so on, we sum them up then square root, and then assign the value to the right cell in the
# distances matrix.

def Euclidean(test, data):
    distances = np.zeros((test.shape[0], data.shape[0]))
    # in these loops we iterate through all samples in train and test.
    for rowTrain in range(data.shape[0]):
        for rowTest in range(test.shape[0]):
            # for each 2 samples, we define the final product
            product = 0
            # we sum up all the squared subtraction of corresponding features
            for inRow in range(data.shape[1]):
                product += (data.iloc[rowTrain][inRow] - test.iloc[rowTest][inRow]) ** 2
            # we apply square root to the entire sum and assign it to the correct cell in distances
            product = np.sqrt(product)
            distances[rowTest][rowTrain] = product
    return distances

def Manhattan(test, data):
    distances = np.zeros((test.shape[0], data.shape[0]))

    for rowTrain in range(data.shape[0]):
        for rowTest in range(test.shape[0]):
            # for each 2 samples, we define the final product
            product = 0
            # we sum up all the absolute distance between each pair of coressponding features
            for inRow in range(data.shape[1]):
                product += np.abs(data.iloc[rowTrain][inRow] - test.iloc[rowTest][inRow])
            # assign the final sumnation of all absolute distances of features
            distances[rowTest][rowTrain] = product
    return distances

def Mahalanobis(test, data):
  distances = np.zeros((test.shape[0], data.shape[0]))
  covariance_matrix_data = np.cov(data, rowvar=False)
  # Calculate the Mahalanobis distances
  for i in range(test.shape[0]):
      for j in range(data.shape[0]):
          diff =  test[i] - data[j]
          distances[i, j] = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(covariance_matrix_data)), diff.T))
  return distances

df = pd.read_csv('Machine learning\SCRAPPED\CSGO-Weapons-Data.csv')

X = df.drop(['Type','Name'], axis=1)         # Feature matrix
y = df['Type']                      # Output vector     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

print(X_train.head())
print(y_train.head())

print(X_test.head())
print(y_test.head())


corr_mat = df.drop('Name', axis=1).corr()
print(corr_mat)

# print(Manhattan(X_test, X_train))

def kNN_classify(data, labels, test, k, metric='Euclidian'):
    arguments = (test, data)
    distances = eval(f'{metric}(*arguments)')   #returns np[][] |test| X |data| by the given metric.
    w = 1

metric='Euclidian'
k = 3
kNN_classify(X_train, y_train, X_test, k, metric)