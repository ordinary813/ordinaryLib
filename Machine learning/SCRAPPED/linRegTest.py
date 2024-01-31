import numpy as np
import matplotlib.pyplot as plt

def Linreg_sol(X, y):
    product = X.T @ X
    productInv = np.linalg.inv(product)
    
    w = productInv @ X.T @ y
    return w[0][0]


data = np.array([[0.4, 3.4], [0.95, 5.8], [0.16, 2.9], [0.7, 3.6], [0.59, 3.27], [0.11, 1.89], [0.05, 4.5]])

# get X and y
X = (data[:,0]).reshape((len(data[:,0]), 1))
y = (data[:,1]).reshape((len(data[:,1]), 1))

w = Linreg_sol(X, y)
#calculate mean array, mean[0]= mean of X, mean[1] = mean of y
mean = np.array([np.mean(X),np.mean(y)])
plt.scatter(X, y, color='royalblue', label='Data')
plt.scatter(mean[0], mean[1], color='black', label='mean')
x = np.arange(-1, 1, 0.01)
y0 = w * (x - mean[0]) + mean[1]
plt.plot(x,y0, color='royalblue')

#centered data
X = X - mean[0]
y = y - mean[1]
plt.scatter(X, y, color='salmon', label='Data')
plt.scatter(0, 0, color='black', label='mean')

w = Linreg_sol(X, y)
x = np.arange(-1, 1, 0.01)
y0 = w * x
plt.plot(x,y0, color='salmon')
plt.grid()
plt.show()
