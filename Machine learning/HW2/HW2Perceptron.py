import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# reading data and giving names to the columns, for the sake of readability
# the dataset is a 3xn array, where:
# column 1 = exam 1,
# column 2 = exam 2
# column 3 = admitted = 1, not admitted = 0
# for the ith sample (row)
df = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw2/exams.csv', header=None)
df = df.rename(columns={0: "Exam 1", 1: "Exam 2", 2: "Admission"})

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
onlyFeatures = df.drop("Admission",axis=1).to_numpy()

# data is the samples, it is also scaled after the following line of code
data = scaler.fit_transform(onlyFeatures)
labels = df["Admission"].replace(0, -1)

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

# this function precprocesses the data so every label == 0 is going to "flip" it's features
# it also appends an extra variable 1 or -1 which is w0
def transformFeatures(data, labels):
    y = np.c_[data, np.ones(data.shape[0])]
    for i in range(data.shape[0]):
        # for every label where class is 0, negate the features
        if(labels[i] == -1):
            y[i] = -y[i]
    return y

def perceptron(data, labels, lr = 1, num_iterations = 10000,upgrade = False):
    # n samples, d features
    n, d = data.shape
    # initialize a = [w w0] to be all 1's, dim = |w| + 1
    a = np.ones(d + 1)
    ws = []
    ws.append(a)
    # initialize y = [x 1] to the data, it is already preprocessed after the following line
    y = transformFeatures(data, labels)

    for _ in range(num_iterations):
        lossDerivative = 0
        for i in range(n):
            # compute (a^t)*y for the current sample
            prediction = a @ y[i]
            if prediction < 0:
                lossDerivative += y[i]
        # update the weights according to the sum of all mis-classified samples
        a += lr * lossDerivative
        ws.append(a)   
    if upgrade:
         return ws
    return a

a = perceptron(data, labels, lr=0.01)
print(f'a = {a}')

# plot(data, labels, a[:-1], a[-1])



# ws = perceptron(data, labels, 0.01, True)    # Implement here

# def plot_anim(ws):
#   for ww in ws:
#     plt.clf()
#     plot(data, labels, ww[:-1], ww[-1], False)
#     display(plt.gcf())
#     clear_output(wait=True)  # Clear the previous plot

# plot_anim(ws)

def findR(data):
    R = 0
    for i in range(data.shape[0]):
        R = max(R, np.linalg.norm(data[i]))
    return R

def deviationVector(data, labels, w):
    dev = []
    y = transformFeatures(data, labels)

    for i in range(data.shape[0]):
        di = max(0, 1 - labels[i] * w @ y[i])
        dev.append(di)
    return dev

dev = deviationVector(data,labels,a)
D = np.linalg.norm(dev)
R = findR(data)
upperBound = 2 * ((R + D) ** 2)
print(f'Upper bound of mistakes is: {upperBound}')

a = perceptron(data, labels, num_iterations=int(np.ceil(upperBound)))

plot(data, labels, a[:-1], a[-1])