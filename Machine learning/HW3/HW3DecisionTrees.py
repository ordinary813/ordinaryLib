import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw3/banknote_authentication.csv')
# print(data.head(3))

# Define the ID3 decision tree class
class DecisionTree:
	def __init__(self):
		self.tree = {}

	def calculate_entropy(self, data):
		labels = data.iloc[:, -1]
		unique_labels, labels_counts = np.unique(labels, return_counts=True)
		probs = labels_counts/ len(labels)
		entropy =  -np.sum(probs * np.log2(probs))
		return entropy
	
	def calculate_information_gain(self, data, feature):
		total_entropy = self.calculate_entropy(data)
		information_gain = total_entropy

		values = np.linspace(np.min(data[feature]), np.max(data[feature]), 10)
		best_treshold = None
		best_gain = 0
		
		for value in values:
			left_split = self.filter_data(data, feature, value, left=True)
			right_split = self.filter_data(data, feature, value, left=False)
			if(len(left_split) == 0 or len(right_split) == 0):
				continue
			left_entropy = self.calculate_entropy(left_split)
			right_entropy = self.calculate_entropy(right_split)
			current_entropy = (len(left_split)/data) * left_entropy + (len(right_split)/data) * right_entropy
			gain = total_entropy - current_entropy
			if(gain > best_gain):
				gain = best_gain
				best_treshold = value

		return best_gain, best_treshold

	def filter_data(self, data, feature, value, left=True):
		if left:
			return data[data[feature] <= value].drop(feature, axis=1)
		else:
			return data[data[feature] > value].drop(feature, axis=1)

	def create_tree(self, data, depth=0):
		# Recursive function to create the decision tree
		labels = data.iloc[:, -1]

		# Base case: if all labels are the same, return the label
		if len(np.unique(labels)) == 1:
			return list(labels)[0]

		features = data.columns.tolist()[:-1]
		# Base case: if there are no features left to split on, return the majority label
		if len(features) == 0:
			unique_labels, label_counts = np.unique(labels, return_counts=True)
			majority_label = unique_labels[label_counts.argmax()]
			return majority_label

		selected_feature = None
		best_gain = 0
		best_treshold = None

		for feature in features:
			gain, treshold = self.calculate_information_gain(data, feature)
			if gain >= best_gain:
				selected_feature = feature
				best_treshold = treshold
				best_gain = gain

		# Create the tree node
		tree_node = {}
		tree_node[(selected_feature, f"<={best_treshold}")] = self.create_tree(self.filter_data(data, selected_feature, best_treshold, left=True), depth+1)
		tree_node[(selected_feature, f">{best_treshold}")] = self.create_tree(self.filter_data(data, selected_feature, best_treshold, left=False), depth+1)

		# check if can unite them.
		if not isinstance(tree_node[(selected_feature, f"<={best_treshold}")], dict) and \
				not isinstance(tree_node[(selected_feature, f">{best_treshold}")], dict):
			if tree_node[(selected_feature, f"<={best_treshold}")] == tree_node[(selected_feature, f">{best_treshold}")]:
				return tree_node[(selected_feature, f"<={best_treshold}")]

		return tree_node

	def fit(self, data):
		self.tree = self.create_tree(data)

	def predict(self, X):
		X = [row[1] for row in X.iterrows()]

		# Predict the labels for new data points
		predictions = []

		for row in X:
			current_node = self.tree
			while isinstance(current_node, dict):
				split_condition = next(iter(current_node))
				feature, value = split_condition
				treshold = float(value[2:])
				if row[feature] <= treshold:
					current_node = current_node[feature, f"<={treshold}"]
				else:
					current_node = current_node[feature, f">{treshold}"]
			predictions.append(current_node)

		return predictions

	def _plot(self, tree, indent):
		depth = 1
		for key, value in tree.items():
			if isinstance(value, dict):
				print(" " * indent + str(key) + ":")
				depth = max(depth, 1 + self._plot(value, indent + 2))
			else:
				print(" " * indent + str(key) + ": " + str(value))
		return depth

	def plot(self):
		depth = self._plot(self.tree, 0)
		print(f'depth is {depth}')