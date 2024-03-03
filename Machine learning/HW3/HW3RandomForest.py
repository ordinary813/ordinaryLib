import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sklearn.model_selection import KFold
from HW3DecisionTrees import DecisionTree

data = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw3/banknote_authentication.csv')
train, test = train_test_split(data, test_size=0.2, stratify=data['class'], random_state=42)

# Define the ID3 decision tree class
class RandomForest:
	def __init__(self, n_estimators=3, method='simple', criterion='entropy'):
		self.forest = []
		self.criterion = criterion
		self.n_estimators = n_estimators
		self.method = method
    
    # this method selects a random n feature subset of the data and returns it
	def select_features(self, data):
		np.random.seed(40+len(self.forest))

		if self.method == 'sqrt':
			m = int(np.sqrt(len(data.columns)-1))
		elif self.method == 'log':
			m = int(np.log2(len(data.columns)-1))
		else:
			m = np.random.randint(0, len(data.columns))

        # select random m features for current classifier
		incidies = np.random.choice(np.arange(0, len(data.columns)-1), size=m, replace=False)
		features = list(data.columns[incidies])
		
        # returns the data only with the selected m features, and their class
		return data[features + ['class']]

	def sample_data(self, data):
		# This method samples len(data) with repitition from data.
		# You can use numpy to select random incidies.
		indices = np.random.choice(np.arange(0, len(data.columns)-1), size=len(data.columns)-1, replace=False)
		return data.iloc[indices, :]

	def fit(self, data):
		self.forest = []
		for _ in range(self.n_estimators):
			samp_data = data.iloc[self.sample_data(data)]
			# Implement here
			
    # def fit(self, data):
    #     self.forest = []
    #     for _ in range(self.n_estimators):
    #         # Sample data with replacement
    #         sampled_data = self.sample_data(data)
            
    #         # Select features for the current classifier
    #         if self.method == 'simple':
    #             features_data = sampled_data.copy()  # Use all features
    #         else:
    #             features_data = self.select_features(sampled_data)
            
    #         # Create and fit a decision tree using the selected features
    #         tree = DecisionTree(criterion=self.criterion)
    #         tree.fit(features_data)
            
    #         # Append the trained decision tree to the forest
    #         self.forest.append(tree)


	def _predict(self, X):
		# Predict the labels for new data points
		predictions = []

		preds = [tree.predict(X) for tree in self.forest]
		preds = list(zip(*preds))
		predictions = [Counter(est).most_common(1)[0][0] for est in preds]

		return predictions

	def score(self, X):
		pred = self._predict(X)
		return (pred == X.iloc[:,-1]).sum() / len(X)