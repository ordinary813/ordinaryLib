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
  
  features = df.columns.values[:-1]
  rows = (len(features) // 2) + (len(features) % 2)
  plt.figure(figsize=(12, 6))

for i, feature in enumerate(features, 1):
    plt.subplot(rows, 2, i)
    plt.hist(df[feature], density=True, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Density Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split