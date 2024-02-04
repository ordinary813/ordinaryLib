import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def readTrainData(file_name):
  df = pd.read_csv(file_name, header=None)

  #labels i.e. first column of the dataframe, length = #samples
  lbAll = df.iloc[:,0].to_numpy()
  
  #2d list where each row is an array of words , |samples| x |amount of words in each row|
  texAll = df.iloc[:,1:].to_numpy()
  texAll = [row[0].split() for row in texAll]
  
  # Flatten the list of lists into a 1D list of words
  allWords_1d = [word for row in texAll for word in row]

  # Find unique words using np.unique
  voc = np.unique(allWords_1d)

  # find all unique classes in lbAll
  cat = np.unique(lbAll)

  return texAll, lbAll, voc, cat


# Pw matrix of class conditional probs, i.e |class| x ||
# P vector of priors
def learn_NB_text():

  # don't want to work with global variables, 
  # copying the features to 'tweets', and the labels to 'classes'
  classes = lblAll_train
  tweets = texAll_train
  
  # labelVectorizer = CountVectorizer()
  # labelVector = labelVectorizer.fit_transform(classes)

  lblDict = {}
  count = 0
  # go over each possible label to find priors
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

  # this object is used for vectorizing words in a string (just counts them), 
  # it usually takes strings but since textAll is words I disabled the analyzer
  # https://stackoverflow.com/questions/47711515/apply-countvectorizer-to-column-with-list-of-words-in-rows-in-python
  vectorizer = CountVectorizer(analyzer=lambda x: x)
  vectorizer.fit(texAll_train)
  voc = vectorizer.vocabulary_
  
  # declare a word vector, a 2d counting array with |rows| x |unique words in all text|
  wordVector = vectorizer.transform(texAll_train).toarray()
  
  # create a matrix the size of |labels| x |unique words|
  # each cell in the place [row][col] holds the probability P(tweet|CLASS)
  # meaning, the probability in class, that a word is going to appear
  # basically sum of #WORD in current class/sum of all #words from class tweets
  Pw = np.zeros((np.shape(cat)[0],np.shape(wordVector)[1]))

  vectorizer = CountVectorizer(analyzer=lambda x: x)
  data = vectorizer.fit_transform(tweets)

  uniqueWords = vectorizer.get_feature_names_out()

  Pw = np.zeros((len(cat),len(uniqueWords)))

  for i,label in enumerate(cat):
      # rows where label appears in the data
      label_indices = [idx for idx, cat in enumerate(classes) if cat == label]
      label_word_counts = data[label_indices, :].sum(axis=0)
      total_words_in_label = label_word_counts.sum()
      Pw[i, :] = label_word_counts / total_words_in_label

  return Pw, P
  # classTweets = np.array([])

  # for rowIndex in range(len(cat)):
  #   totalWordsClass = 0
  #   sumOfWordInClass = 0
  #   classTweetArray = np.array([])

  #   # get the total amount of words from current class (label)
  #   # iterate over the training data, for every row that has the current class
  #   # add the amonut of the row's words
  #   for dataRow in range(len(tweets)):
  #     if(cat[rowIndex][0] == classes[dataRow]):
  #       # totalWordsClass = totalWordsClass + len(tweets[rowIndex])

  #       # add current tweet to the current array of tweets in class
  #       classTweetArray = np.append(classTweetArray,tweets[dataRow])

  #   classTweets = np.append(classTweets,classTweetArray)



  # --------------------------- CHECK ------------------------ #
  # for i, category in enumerate(cat):

  #   catIndex = [index for index,category2 in]
  #   # an array that each index holds how much a word appears for a certain class
  #   wordCountsInC = wordVector[classTweets, :].sum(axis=0)
  #   # a value that is the total amount of words with the appearance of a certain class in the data
  #   totalWordsInC = wordCountsInC.sum()
  #   # computation of a word in the i'th cell probability of being in a certain class
  #   Pw[i,:] = wordCountsInC/totalWordsInC
  
    # for wordIndex in range(np.shape(wordVector)[1]):
    #   # CHANGE HERE, SHOULD BE #WORD APPEARS IN CLASS/# WORD APPEARS IN ALL DATA
    #   Pw[rowIndex][wordIndex] = sumOfWordInClass/totalWordsClass

  

# COMPARE THE ALGORITHM'S PREDICITON TO THE REAL LABELS OF TEST

# def ClassifyNB_text(Pw, P):

   
#   tweets = texAll_test
#   labels = lblAll_test

#   correct = 0
# 	for dataRow in range(len(tweets)):

#   return correct/len(tweets)
     
      
     
TRAIN_FILE = 'https://sharon.srworkspace.com/ml/datasets/hw1/cyber_train.csv'
TEST_FILE = 'https://sharon.srworkspace.com/ml/datasets/hw1/cyber_test.csv'

texAll_train, lblAll_train, voc, cat = readTrainData(TRAIN_FILE)

# cats must be the same at train and test
# voc of test is irrelevant - we already trained on other voc.
texAll_test, lblAll_test, _, __ = readTrainData(TEST_FILE)

Pw, P = learn_NB_text()
# sum_right = ClassifyNB_text(Pw, P)
# print(sum_right)