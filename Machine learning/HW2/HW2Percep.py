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

def preprocessNonNegative(data, labels):
    for i in range(data.shape[0]):
        # for every label where class is 0, negate the features
        if(labels.iloc[i] == 0):
            data.iat[i,0] = -data.iat[i,0]
            data.iat[i,1] = -data.iat[i,1]
    
    return data, labels
    
def perceptron(data, labels, lr = 1):
    # initialize w to be all 1's, weights is included with w0 which is why it is the dimensions + 1
    a = np.ones(data.shape[1] + 1)
    data, labels = preprocessNonNegative(data, labels)

    for _ in range(1000):
        # mini batch gradient descent
        # for every w in each step, calculate the loss by identifying wrong predictions and summing
        # it to a value, add to the current point lr*lossDerivative
        # lossDerivative is the derivative of the loss function, so basically sum of all feature vectors that are miss-classified
        lossDerivative = np.zeros(data.shape[1]+1)
        for i in range(data.shape[0]):
            # compute (a^t)*y for the current sample
            prediction = a[:-1].T @ data.iloc[i] + a[-1]
            if prediction < 0:
                # the gradient of every Xi is just the value of the Xi (for perceptron) 
                yi = data.iloc[i].to_numpy()
                # (and 1)
                yi = np.append(yi,1)
                lossDerivative += yi
        # update the weights according to the sum of all mis-classified samples
        a += lr * lossDerivative
    return a


def perceptronUpgrade(data, labels, lr = 1):
    # initialize w to be all 1's, weights is included with w0 which is why it is the dimensions + 1
    a = np.ones(data.shape[1] + 1)
    data, labels = preprocessNonNegative(data, labels)
    ws = np.ones(data.shape[1] + 1)
    for _ in range(1000):
        # mini batch gradient descent
        # for every w in each step, calculate the loss by identifying wrong predictions and summing
        # it to a value, add to the current point lr*lossDerivative
        # lossDerivative is the derivative of the loss function, so basically sum of all feature vectors that are miss-classified
        lossDerivative = np.zeros(data.shape[1]+1)
        for i in range(data.shape[0]):
            # compute (a^t)*y for the current sample
            prediction = a[:-1].T @ data.iloc[i] + a[-1]
            if prediction < 0:
                # the gradient of every Xi is just the value of the Xi (for perceptron) 
                yi = data.iloc[i].to_numpy()
                # (and 1)
                yi = np.append(yi,1)
                lossDerivative += yi
        # update the weights according to the sum of all mis-classified samples
        a += lr * lossDerivative
        ws = np.vstack([ws,a])
    return ws

a = perceptron(scaledDF.drop("Admission",axis=1),scaledDF["Admission"])

plot(scaledDF.drop("Admission",axis=1).to_numpy(), scaledDF["Admission"].to_numpy(), a[:-1], a[-1], show=True)