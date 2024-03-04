import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw3/banknote_authentication.csv')
# print(data.head(3))

# Define the ID3 decision tree class
class DecisionTree:
	def __init__(self, criterion='entropy', thresholds=10):
		self.tree = {}
		self.criterion = criterion
		self.thresholds = thresholds

	# calculate total entropy of data
	def calculate_entropy(self, data):
		labels = data.iloc[:, -1]
		unique_labels, labels_counts = np.unique(labels, return_counts=True)
		probs = labels_counts/ len(labels)
		entropy =  -np.sum(probs * np.log2(probs))
		return entropy
	
	# calculate gini of data
	def calculate_gini(self, data):
		labels = data.iloc[:, -1]
		unique_labels, labels_counts = np.unique(labels, return_counts=True)
		probs = labels_counts/ len(labels)
		gini =  1 - np.sum(probs ** 2)
		return gini
	
	# calculate the information gain for a certain node, using a specific feature
	def calculate_information_gain(self, data, feature):
		# calculate the impurity of the current node by running calculate_'criterion'(data)
		total_impurity = eval(f"self.calculate_{self.criterion}(data)")

		# values interval
		values = np.linspace(np.min(data[feature]), np.max(data[feature]), self.thresholds)
		best_treshold = None
		best_gain = 0
		
		# iterate over each threshold value and decide which one gives the best information gain for the data over a feature
		for value in values:
			# split the current node's children into 2 sub-trees
			# features that are greater than value (right split) 
			# features that are lower than value (left split)
			left_split = self.filter_data(data, feature, value, left=True)
			right_split = self.filter_data(data, feature, value, left=False)

			# calculate impurity of each sub-tree, and add the weighted sum of them to 'current_entropy'
			left_impurity = eval(f"self.calculate_{self.criterion}(left_split)")
			right_impurity = eval(f"self.calculate_{self.criterion}(right_split)")

			# this is the sum of all weighted impurities out of a node
			sub_impurity = (len(left_split)/len(data)) * left_impurity + (len(right_split)/len(data)) * right_impurity

			# calculate information gain for the current node
			gain = total_impurity - sub_impurity
			
			# get the max gain and the coressponding value
			if(gain > best_gain):
				best_gain = gain
				best_treshold = value
		return best_gain, best_treshold

	def filter_data(self, data, feature, value, left=True):
		if left:
			# return the data where the value of 'feature' is less than value, without feature column
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
  
# tree = DecisionTree()
# tree.fit(data)
# tree.plot()

# train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['class'])

# for criterion in ["entropy", "gini"]:
# 	print(f"------------ {criterion} ------------")
# 	# define a decision tree and fit the training data on it
# 	tree_train = DecisionTree(criterion)
# 	tree_train.fit(train)

# 	# compare true labels with predicted labels
# 	trainLabels = train.iloc[:, -1]
# 	trainPreds = tree_train.predict(train)
# 	acc = np.mean( trainPreds == trainLabels)
# 	print(f'Training accuracy is {acc}')

# 	# test the trained decision tree with the test set
# 	testLabels = test.iloc[:, -1]
# 	testPreds = tree_train.predict(test)
# 	acc = np.mean( testPreds == testLabels)
# 	print(f'Test accuracy is {acc}')
# 	print()