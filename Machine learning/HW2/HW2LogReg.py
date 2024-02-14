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
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

# convert the dataframe to numpy array
X_train = X_train.values
X_test = X_test.values
X_val = X_val.values

y_train = y_train.values
y_test = y_test.values
y_val = y_val.values

def sigmoid(z):
  return 1/(1+np.exp(-z))

def Logistic_Regression_via_GD(P,y,lr,lamda = 0):
  m, n = P.shape
  w = np.random.randn(n, 1)  # Initialize weights with random values
  b = 0  # Initialize bias

  for _ in range(1000):
    # Compute the hypothesis function
    z = np.dot(P, w) + b
    phi = sigmoid(z)

    # Compute the gradient of the loss function
    dw = (1 / m) * np.dot(P.T, (phi - y))
    db = (1 / m) * np.sum(phi - y)

    # Update the weights using gradient descent
    w -= lr * dw
    b -= lr * db

  return w, b

def predict(x, w, b):
    z = np.dot(x, w) + b
    h = sigmoid(z)
    return h


w, b =Logistic_Regression_via_GD(X_train,y_train,1)

print("w =" +w+ "\nb = " +b)