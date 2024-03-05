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

	# returns row indices of |selected features| to account
	def sample_data(self, data):
		indices = np.random.choice(np.arange(0, len(data)-1), size=len(data)-1)
		return indices

	def fit(self, data):
		self.forest = []
		for _ in range(self.n_estimators):
			samp_data = data.iloc[self.sample_data(data)]
			
			# only if the method is simple we copy the entire data
			# otherwise we select m random features to take into account
			if(self.method == 'simple'):
				features = samp_data.copy()
			else:
				features = self.select_features(samp_data)

			# define a decision tree to add to the forest
			tree = DecisionTree(criterion=self.criterion)
			tree.fit(features)
			self.forest.append(tree)


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


def KFold2(data, model, cv=5):
  kf = KFold(n_splits=cv)
  scores = []

  for train_index, test_index in kf.split(data):
    train_data, test_data = data.iloc[train_index], data.iloc[test_index]
    model.fit(train_data)
    score = model.score(test_data)
    scores.append(score)

  return np.mean(scores)

# dict1 = {'entropy': [], 'gini': []}

# criterions = ['entropy', 'gini']
# for crt in criterions:
#   forest = RandomForest(n_estimators=3, method='simple', criterion=crt)
#   forest.fit(train)

#   acc = forest.score(train)
#   dict1[crt].append(acc)

#   acc = forest.score(test)
#   dict1[crt].append(acc)

# print('using 3 estimators')
# df = pd.DataFrame(dict1, columns=criterions, index=['train', 'test'])
# print(df)

correct_entropy = []
correct_gini = []

# I made the range a variable for ease of use
myRange = range(3,13,2)

for i in tqdm(myRange):
	forest = RandomForest(n_estimators=i, method='simple', criterion='gini')
	correct_gini.append(KFold2(data=train, model=forest, cv=5))
	forest = RandomForest(n_estimators=i, method='simple', criterion='entropy')
	correct_entropy.append(KFold2(data=train, model=forest, cv=5))

plt.plot(myRange, np.array(correct_entropy), label='entropy')
plt.plot(myRange, np.array(correct_gini), label='gini')

plt.legend(loc='upper left')
plt.xlabel('trees num')
plt.ylabel('avg accuracy')
plt.show()


dict1 = {'entropy': [], 'gini': []}

best_n_entropy = np.argmax(np.array(correct_entropy)) + myRange.start
best_n_gini = np.argmax(np.array(correct_gini)) + myRange.start

# let's say I want to use the best_n to be the best for gini, then:
best_n = best_n_gini

criterions = ['entropy', 'gini']
for crt in criterions:
  forest = RandomForest(n_estimators=best_n, method='simple', criterion=crt)
  forest.fit(train)

  acc = forest.score(train)
  dict1[crt].append(acc)

  acc = forest.score(test)
  dict1[crt].append(acc)

print(f'using {best_n} estimators')
df = pd.DataFrame(dict1, columns=criterions, index=['train', 'test'])
print(df)