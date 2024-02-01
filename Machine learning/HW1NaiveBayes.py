import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def readTrainData(file_name):
  df = pd.read_csv(file_name)
  
  #labels i.e. first column of the dataframe
  lbAll = df.iloc[:,0].to_numpy()
  
  #2d list where each row is an array of words 
  texAll = df.iloc[:,1:].to_numpy()
  # all unique words in the second column
  voc = np.unique(texAll)
  # enumerate {"not bullying", "gender", "age", "religion", "ethnicity"} in an array
  cat = {
    0:"not bullying",
    1:"gender",
    2:"age",
    3:"religion",
    4:"ethnicity"
  }
  return texAll, lbAll, voc, cat

def learn_NB_text():
  # Implement here
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