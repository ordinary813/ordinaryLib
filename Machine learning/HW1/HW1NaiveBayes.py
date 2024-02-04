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

def learn_NB_text():
  # don't want to work with global variables, 
  # copying the features to 'tweets', and the labels to 'classes'
  classes = lblAll_train
  tweets = texAll_train

  # declare a dictionary that will hold (category: it's prior)
  labelDict = {}

  # go over each possible label to find priors
  for label in cat:
    # create a new cell for each unique label
    labelDict[label] = []
    count = 0
    # iterate over samples
    for rowIndex in range(len(classes)):
      # if the current row has the right label, increment count
      if(classes[rowIndex] == label):
        count = count + 1
    
    # calculate the prior and push it to the array
    prior = count / len(classes)
    labelDict[label].append(prior)

    count = 0
  P = labelDict
  
  # this object is used for vectorizing words in a string (just counts them), 
  # it usually takes strings but since textAll is words I disabled the analyzer
  # https://stackoverflow.com/questions/47711515/apply-countvectorizer-to-column-with-list-of-words-in-rows-in-python
  vectorizer = CountVectorizer(analyzer=lambda x: x)
  tweetsVector = vectorizer.fit_transform(tweets)

  uniqueWords = vectorizer.get_feature_names_out()
  
  # create a matrix the size of |labels| x |unique words|
  # each cell in the place [row][col] holds the probability P(tweet|CLASS)
  Pw = np.zeros((len(cat),len(uniqueWords)))

  for i,label in enumerate(cat):
      # row Indices where label appears in the samples
      labelIndices = [idx for idx, cat in enumerate(classes) if cat == label]

      # creates an array that for every row of the current label - sums how many of each word there is
      wordCountsLabel = np.array(tweetsVector[labelIndices, :].sum(axis=0))
      # total amount of words of the current label
      sumWordsLabel = wordCountsLabel.sum()

      # for every word, calculate |curent word in class| / |words in class| with laplace smoothing
      for wordIndex in range(len(uniqueWords)):
        Pw[i][wordIndex] = (wordCountsLabel[0][wordIndex] + 1) / (sumWordsLabel + len(uniqueWords))

  return Pw, P

  

# COMPARE THE ALGORITHM'S PREDICITON TO THE REAL LABELS OF TEST
# Implement fhe function that classifies all tweets from the test set and computes the success rate.
# Iterate over all tweets of test and for each tweet find the most probable category.
def ClassifyNB_text(Pw, P):

  tweets = texAll_test
  labels = lblAll_test
  laplaceSmooth = 1

  vectorizer = CountVectorizer(analyzer=lambda x: x)
  tweetsVector = vectorizer.fit_transform(texAll_train)

  correct = 0
  # iterate over all rows of test
  for rowIndex in range(len(tweets)):
    # calculate probabilities for each label
    labelProbs = P
  
    # iterate over all words in the current tweet
    for word in tweets[rowIndex]:
      # get the index of the word from vectorizer,
      # this is crucial because Pw's word indices are built with those indices!
      wordIndex = vectorizer.vocabulary_.get(word, -1)

      if wordIndex != -1:
        # calculate the log probability of the word given each label and update label_probabilities
        for i, label in enumerate(cat):
          labelProbs[i] += np.log(Pw[i][wordIndex])

    # choose the label with the highest probability
    predicted_label = cat[np.argmax(labelProbs)]

    # check if the predicted label matches the true label
    if predicted_label == labels[rowIndex]:
      correct += 1

  return correct/len(tweets)


TRAIN_FILE = 'https://sharon.srworkspace.com/ml/datasets/hw1/cyber_train.csv'
TEST_FILE = 'https://sharon.srworkspace.com/ml/datasets/hw1/cyber_test.csv'

texAll_train, lblAll_train, voc, cat = readTrainData(TRAIN_FILE)

# cats must be the same at train and test
# voc of test is irrelevant - we already trained on other voc.
texAll_test, lblAll_test, _, __ = readTrainData(TEST_FILE)

Pw, P = learn_NB_text()
sum_right = ClassifyNB_text(Pw, P)
print(sum_right)