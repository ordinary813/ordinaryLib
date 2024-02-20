import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from mpl_toolkits.mplot3d import Axes3D

def plot(data, labels, w, bias):

  a, b, c = w[0], w[1], w[2]
  d = bias

  # create a 3D scatter plot
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='coolwarm')

  xx, yy = np.meshgrid(range(-2, 2), range(-2, 2))
  z = (-a * xx - b * yy - d) * 1.0 / c

  ax.plot_surface(xx, yy, z, alpha=0.4)
  ax.azim += 30
  ax.elev += 10
  #ax.view_init(elev=0, azim=90, roll=45)

  # customize the plot
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  plt.title('3D Scatter Plot with 2D Labels')
  plt.show()
  
df = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw2/suv_data.csv')

# remove userID, it has no value in the classification
df = df.drop(['User ID'], axis=1)
# replace categorical values with discrete binary values for gender
genders = ['Male','Female']
for idx, gender in enumerate(genders):
  df['Gender'] = df['Gender'].replace({gender: idx})

# preprocess the data to fit with the gradient later
df['Purchased'] = df['Purchased'].replace(0, -1)

# df.drop(['Purchased'],axis=1).plot(kind='density', subplots=True, layout=(1,3), figsize=(8, 4), sharex=False)
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(df[['Age', 'EstimatedSalary']])
df[['Age', 'EstimatedSalary']] = scaler.transform(df[['Age', 'EstimatedSalary']])

# First, split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(df.drop('Purchased', axis=1), df['Purchased'], test_size=0.2, random_state=42)

# Then, split the remaining data into training and validation sets (70% train, 30% validation)
X_real_train, X_val, y_real_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# convert the dataframe to numpy array
X_train = X_train.values
X_test = X_test.values
X_val = X_val.values
X_real_train = X_real_train.values

y_train = y_train.values
y_test = y_test.values
y_val = y_val.values
y_real_train = y_real_train.values

def sigmoid(z):
  return 1/(1+np.exp(-z))

def Logistic_Regression_via_GD(P,y,lr,lamda = 0):
  # n samples, d features
  n, d = P.shape
  # it was easier to comprehend w and w0 seperately
  w = np.ones(d)
  w0 = 0

  for _ in range(1000):
    gradient = 0
    gradientBias = 0
    # for every sample, calculate z. 
    # then if true label == 1: compute yi*xi*P(C=-1|xi), and for bias yi*1*P(C=-1|xi)
    # if true label == -1: compute yi*xi*P(C=1|xi), and for bias yi*1*P(C=1|xi)
    for i in range(n):
      z = w @ P[i] + w0
      # calcualte the current expression to add to the sum of the gradient
      if y[i] == 1:
        gradient += y[i] * P[i] * (1-sigmoid(z))
        gradientBias += y[i] * 1 * (1-sigmoid(z))
      elif y[i] == -1:
        gradient += y[i] * P[i] * sigmoid(z)
        gradientBias += y[i] * 1 * sigmoid(z)

    # regularization functionality
    gradient += 2 * lamda * w
    gradientBias += 2 * lamda * w0

    # update the weights using gradient descent
    w += lr  * gradient
    w0 += lr *  gradientBias
  return w, w0

# w vector and bias
# x is a feature vector of a sample
def predict(x,w,b):
  z = w @ x + b
  if z >= 0.5:
    return 1
  elif z < 0.5:
    return -1

w, b =Logistic_Regression_via_GD(X_train,y_train,0.1)
print("w =" ,w, "\nb = " ,b)

count = 0
for i in range(X_test.shape[0]):
  if y_test[i] == predict(X_test[i],w,b):
    count += 1
print("Accuracy = ", 100 * count/X_test.shape[0], "%")

plot(X_test, y_test, w, b)

lamads = np.arange(0, 5, 0.1)
maxLamda = 0
maxAccuracy = 0
for lamda in lamads:
  w, b =Logistic_Regression_via_GD(X_train,y_train,0.1, lamda)

  count = 0
  for i in range(X_val.shape[0]):
    if y_val[i] == predict(X_val[i],w,b):
      count += 1
  accuracy = count/X_val.shape[0]

  if accuracy > maxAccuracy:
    maxLamda = lamda
    maxAccuracy = accuracy
  print(f"Valdation accuracy for lamda={lamda:.2f}: {accuracy * 100}%")

w, b = Logistic_Regression_via_GD(X_train, y_train, 0.1, maxLamda)
count = 0
for i in range(X_test.shape[0]):
  if y_test[i] == predict(X_test[i],w,b):
    count += 1
print("Accuracy = ", 100 * count/X_test.shape[0], "%")