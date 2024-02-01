import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def readTrainData(file_name):
  df = pd.read_csv(file_name, header=None)

  #labels i.e. first column of the dataframe
  lbAll = df.iloc[:,0].to_numpy()
  
  #2d list where each row is an array of words 
  texAll = df.iloc[:,1:].to_numpy()
  texAll = [row[0].split() for row in texAll]
  
  # Flatten the list of lists into a 1D list of words
  allWords_1d = [word for row in texAll for word in row]

  # Find unique words using np.unique
  voc = np.unique(allWords_1d)

  # enumerate {"not bullying", "gender", "age", "religion", "ethnicity"} in an array
  cat = {
    0:"not bullying",
    1:"gender",
    2:"age",
    3:"religion",
    4:"ethnicity"
  }
  return texAll, lbAll, voc, cat

# def classify_point_gaussian_bayes(x):
#     # I'm copying the data so the original data will not change for any reason
#     scores = []
#     samplesTrain = X_train
#     classesTrain = y_train

#     classes = np.unique(y_train)        # number of classes

#     for c in classes:
#         X_c = X_train[y_train == c]         # array of samples that are from class c

#         mean_c = np.mean(X_c, axis=0)       # mean vector of the current class (mean of each feature)
#         cov_c = np.cov(X_c, rowvar=False)   # compute covariance matrix for current class (cov matrix of all features for X_c)
#         prior_c = len(X_c) / len(X_train)   # compute prior for the current class (amount of appearances of class out of the entire data)

#         likelihood = estimateLikelihood(x, mean_c, cov_c)
#         score = likelihood * prior_c
#         scores.append(score)
    
#     predicted_class = classes[np.argmax(scores)]
#     return predicted_class    

# Pw matrix of class conditional probs
# P vector of priors
def learn_NB_text():
  P = []
  Pw = []
  # set an array of all classes
  classes = lblAll_train
  featureSamples = texAll_train

  for c in classes:
    samples_c = featureSamples[classes == c]

  return Pw, P

# def ClassifyNB_text(Pw, P):
# 	# Implement here
     
TRAIN_FILE = 'https://sharon.srworkspace.com/ml/datasets/hw1/cyber_train.csv'
TEST_FILE = 'https://sharon.srworkspace.com/ml/datasets/hw1/cyber_test.csv'

texAll_train, lblAll_train, voc, cat = readTrainData(TRAIN_FILE)

# cats must be the same at train and test
# voc of test is irrelevant - we already trained on other voc.
# texAll_test, lblAll_test, _, __ = readTrainData(TEST_FILE)

# Pw, P = learn_NB_text()
# sum_right = ClassifyNB_text(Pw, P)
# print(sum_right)