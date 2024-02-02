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


# need to use count vectorizer to vectorize this data, cant work with words.
# Pw matrix of class conditional probs
# P vector of priors
def learn_NB_text():
  P = []
  Pw = []

  # read the train dataset so we could make a vectorizer
  df = pd.read_csv(TRAIN_FILE, header=None)
  
  # insert each sentence into a 1d array of strings, for vectorizer
  strArr = [df.iloc[:,1:].to_numpy()[i][0] for i in range(np.shape(df.iloc[:,1:].to_numpy())[0])]
  vectorizer = CountVectorizer()
  vectorizer.fit(strArr)

  print(f'Vocabulary:\n{vectorizer.vocabulary_}')
  voc = vectorizer.vocabulary_
  
  vector = vectorizer.transform(strArr)

  print(vector.toarray())
  # set an array of all classes
  classes = lblAll_train
  tweets = texAll_train

  for c in classes:
    samples_c = tweets[classes == c]

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