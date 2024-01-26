import numpy as np
import matplotlib.pyplot as plt

import numpy as np

class LinearRegression:
    
    def fit(self, X, y):
        n = X.shape[0]
        #calculate w= (X^t*X)^(-1)*X^t*y, since it is the average of distances we calculate the mean of x and mean of y
        x_mean = np.mean(X)                 #E[X]
        y_mean = np.mean(y)                 #E[Y]
        xy_mean = np.mean(X * y)            #E[XY]
        x_squared_mean = np.mean(X ** 2)    #E[X^2]
        
        #calculate w (m) and b0 (b)
        #(looks like m = E[XY]- E[X]E[Y]/E[X^2]-E[X]^2)
        self.m = (xy_mean - x_mean * y_mean) / (x_squared_mean - x_mean ** 2)
        #looks like E[Y]-m*E[m]
        self.b = y_mean - self.m * x_mean
    
    def predict(self, X):
        return self.m * X + self.b
	

data = np.array([[0.4, 3.4], [0.95, 5.8], [0.16, 2.9], [0.7, 3.6], [0.59, 3.27], [0.11, 1.89], [0.05, 4.5]])
data_X = data[:,0]  #array in the size of each column in data
data_y = data[:,1]  #array in the size of each column in data

model = LinearRegression()
model.fit(data_X,data_y)

print(model.m)
plt.scatter(data[:,0], data[:,1], color='blue', label='Data')
plt.plot(data_X, model.predict(data_X), '-r')  # solid
plt.show()