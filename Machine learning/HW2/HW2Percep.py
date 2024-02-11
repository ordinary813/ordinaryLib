import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import seaborn as sns

# -------------------------------- Implementation ------------------------------------------ #
# reading data and giving names to the columns, for the sake of readability
# the dataset is a 3xn array, where:
# column 1 = exam 1,
# column 2 = exam 2
# column 3 = admitted = 1, not admitted = 0
# for the ith sample (row)
df = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw2/exams.csv', header=None)
df = df.rename(columns={0: "Exam 1", 1: "Exam 2", 2: "Admission"})

features = df.columns.values[:-1]

# df.plot(kind='density', subplots=True, layout=(1,3), figsize=(12, 6), sharex=False)
# plt.show()
# ----------------------------------------------------------------------------------------- #

# plt.figure(figsize=(12, 6))
# sns.set_style('whitegrid')
# for i, feature in enumerate(features, 1):
#     plt.subplot(1, 2, i)
#     plt.hist(df[feature], density=True, bins=25, alpha=0.7, label=feature)
    
#     sns.kdeplot(np.array(df[feature]), bw=0.5, color='RoyalBlue')
#     plt.title(f'Density of {feature}')
#     plt.xlabel(feature)
#     plt.ylabel('Density')
# plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
onlyFeatures = df.drop("Admission",axis=1)

scaledFeatures = scaler.fit_transform(onlyFeatures)

# create scaled dataframe
scaledDF = pd.DataFrame(scaledFeatures, columns=onlyFeatures.columns)

# concatenate the scaled marks with the original Admission column
scaledDF = pd.concat([scaledDF, df["Admission"]], axis=1)

def plot(train, labels, w, bias, show=True):
	# Create a figure and axis object
	fig, ax = plt.subplots()

	c0 = train[labels == -1]
	c1 = train[labels == 1]

	# Plot the data
	ax.scatter(c0[:,0], c0[:,1], c='red')
	ax.scatter(c1[:,0], c1[:,1], c='blue')

	a, b, c = w[0], w[1], bias

	# Compute the slope and y-intercept of the line
	m = -a / b
	b = -c / b

	# Generate some x values for the plot
	x = np.arange(np.min(train[:,0]), np.max(train[:,0]), 0.1)

	# Compute the corresponding y values using the equation of the line
	y = m * x + b

	# Plot the line
	plt.plot(x, y)

	# Add axis labels and title
	ax.set_xlabel('X-axis')
	ax.set_ylabel('Y-axis')

	preds = np.sign(np.dot(train, w)+bias)
	acc = np.count_nonzero(labels == preds) / len(labels)

	ax.set_title(f'Train accuracy is {acc}')
	ax.set_xlim(-0.1, 1.1)
	ax.set_ylim(-0.1, 1.1)

	if show:
		plt.show()

def perceptron(data, labels, lr = 1):
    # initialize w to be all 1's
    weights = np.ones(data.shape[1])
    bias = 1
    
    # # preprocess the reduction from a^t*t < 0 and > 0 to just < 0
    # for i in range(data.shape[0]):
    #     if(labels.iloc[i] == 0):
    #         # you cant multiply 0 by (-1), change it to 1
    #         labels.iloc[i] == 1
            
    #         # multiply the samples feature values by (-1)
    #         data.iloc[i][0] = -data.iloc[i][0]
    #         data.iloc[i][1] = -data.iloc[i][1]
    
    for _ in range(100):
        # mini batch gradient descent
        # for every w in each step, calculate the loss by  identifying wrong predictions and summing
        # it to a value, add to the current point lr*sum
        for i in range(data.shape[0]):
            # compute (a^t)*y for the current sample
            prediction = weights.T @ data.iloc[i] + bias
            if prediction < 0:
                # the gradient of every Xi is just the value of the Xi (for perceptron)
                weights += lr * data.iloc[i].iloc[0] + lr * data.iloc[i].iloc[1]
                bias += lr * data.iloc[i].iloc[0] + lr * data.iloc[i].iloc[1]
    return weights, bias

    # preprocess the reduction from a^t*t < 0 and > 0 to just < 0
for i in range(scaledDF.shape[0]):
    if(scaledDF.iloc[i].iloc[2] == 0):
    # you cant multiply 0 by (-1), change it to 1
        scaledDF.iat[i,2] = -1
        
        # multiply the samples feature values by (-1)
        # scaledDF.iat[i,0] = -scaledDF.iat[i,0]
        # scaledDF.iat[i,1] = -scaledDF.iat[i,1]
            
w, w0 = perceptron(scaledDF.drop("Admission",axis=1),scaledDF["Admission"],0.5)

plot(scaledDF.drop("Admission",axis=1).to_numpy(), scaledDF["Admission"].to_numpy(), w, w0, show=True)