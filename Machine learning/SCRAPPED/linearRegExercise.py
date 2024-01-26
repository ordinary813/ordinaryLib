import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression

#implement for and size of X, meaning i need to find w = (X^t*X)^-1*X^t*y

# Get dataset
df_sal = pd.read_csv('content/Salary_Data.csv')
df_xi = df_sal.iloc[:, :1]
df_yi = df_sal.iloc[:, -1:]

# Sample data
# xi = np.array([6,4,2,6,4,6,1,2,33,4,6,19,43,23,43,65,23,54,87,34,12,32])
# yi = np.array([3,5,2,7,9,3,1,11,25,2,3,66,23,54,65,87,45,65,34,65,23,12])

xi = df_xi.to_numpy()
yi = df_yi.to_numpy()

# Calculate the mean of x and y
mean_x = df_xi.mean()
mean_y = df_yi.mean()

# # Calculate the slope (m) and y-intercept (b) using the formulas
# m doesnt return a value but a dataframe - check it
# m_check = np.sum((df_xi - mean_x) * (df_yi - mean_y)) / np.sum((df_xi - mean_x)**2)
m = ((df_xi - mean_x) * (df_yi - mean_y)).sum()
b = mean_y - m * mean_x

# Predict the y values using the calculated slope and y-intercept
predictions = m * xi + b

# Print the coefficients
print('Slope (m):', m)
print('Y-intercept (b):', b)

# Plot the data and the regression line
plt.scatter(xi, yi, label='Actual data')
plt.plot(xi, predictions, color='red', label='Linear regression')
plt.xlabel('xi')
plt.ylabel('yi')
plt.legend()
plt.show()
