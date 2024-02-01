import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

def readTrainData(file_name):
  df = pd.read_csv(file_name)
  # Implement here
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
texAll_test, lblAll_test, _, __ = readTrainData(TEST_FILE)

Pw, P = learn_NB_text()
sum_right = ClassifyNB_text(Pw, P)
print(sum_right)