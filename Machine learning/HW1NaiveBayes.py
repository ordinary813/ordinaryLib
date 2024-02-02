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

  # find all unique classes in lbAll
  cat = np.unique(lbAll)
  cat = np.reshape(cat,[len(cat),1])

  return texAll, lbAll, voc, cat


# need to use count vectorizer to vectorize this data, cant work with words.
# Pw matrix of class conditional probs, i.e |class| 
# P vector of priors
def learn_NB_text():

  # this object is used for vectorizing words in a string (just counts them), 
  # it usually takes strings but since textAll is words I disabled the analyzer
  # https://stackoverflow.com/questions/47711515/apply-countvectorizer-to-column-with-list-of-words-in-rows-in-python
  vectorizer = CountVectorizer(analyzer=lambda x: x)
  vectorizer.fit(texAll_train)
  voc = vectorizer.vocabulary_
  
  # declare a word vector, a 2d counting array with |rows| x |unique words in all text|
  wordVector = vectorizer.transform(texAll_train).toarray()
  print(wordVector)

  # set an array of all classes
  classes = lblAll_train
  tweets = texAll_train

  lblDict = {}
  count = 0
  
  # go over each possible label
  for label in range(len(cat)):
    lblDict[cat[label][0]] = []

    # sum appearances of current label out of all training samples
    for j in range(len(classes)):
      if(classes[j] == cat[label]):
        count = count + 1
    
    # calculate the prior and push it to the array
    prior = count / len(classes)
    lblDict[cat[label][0]].append(prior)

    count = 0
  P = lblDict

  # create a matrix the size of |labels| x |unique words|
  Pw = np.zeros((np.shape(cat)[0],np.shape(wordVector)[1]))

  for rowIndex in range(len(np.shape(cat)[0])):
    for wordIndex in range(len(np.shape(wordVector)[1])):
      i = 1 #CCHANGE HERE, SHOULD BE #WORD APPEARS IN CLASS/# WORD APPEARS IN ALL DATA

  return Pw, P

# COMPARE THE ALGORITHM'S PREDICITON TO THE REAL LABELS OF TEST

# def ClassifyNB_text(Pw, P):
# 	# Implement here
     
TRAIN_FILE = 'https://sharon.srworkspace.com/ml/datasets/hw1/cyber_train.csv'
TEST_FILE = 'https://sharon.srworkspace.com/ml/datasets/hw1/cyber_test.csv'

texAll_train, lblAll_train, voc, cat = readTrainData(TRAIN_FILE)

# cats must be the same at train and test
# voc of test is irrelevant - we already trained on other voc.
texAll_test, lblAll_test, _, __ = readTrainData(TEST_FILE)

Pw, P = learn_NB_text()
# sum_right = ClassifyNB_text(Pw, P)
# print(sum_right)