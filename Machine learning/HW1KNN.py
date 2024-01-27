import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw1/Stars.csv')
df.head(3)

colors = df['Color'].unique()
for idx, color in enumerate(colors):
  df['Color'] = df['Color'].replace({color: idx})

spec_class = df['Spectral_Class'].unique()
for idx, spec in enumerate(spec_class):
  df['Spectral_Class'] = df['Spectral_Class'].replace({spec: idx})
# df.head(3)

# -------------------------- IMPLEMENTATION OF CORRELATION MATRIX ---------------------------- #
corrMat = df.corr()

# ------------------------------- WHICH DISTANCE METRIC -------------------------------------- #
# I think mahalanobis distance will be better since the dataset is built with a lot of features
# -------------------------------------------------------------------------------------------- #

X = df.drop('Type', axis=1)         # Feature matrix
y = df['Type']                      # Output vector     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

def Euclidean(test, data):
  distances = np.zeroes(test.shape[0],data.shape[0])

def Manhattan(test, data):
  distances = np.zeroes(test.shape[0],data.shape[0])

def Mahalanobis(test, data):
  distances = np.zeros((test.shape[0], data.shape[0]))
  covariance_matrix_data = np.cov(data, rowvar=False)

  # Calculate the Mahalanobis distances
  for i in range(test.shape[0]):
      for j in range(data.shape[0]):
          diff =  test[i] - data[j]
          distances[i, j] = np.sqrt(np.dot(np.dot(diff, np.linalg.inv(covariance_matrix_data)), diff.T))
  return distances