import numpy as np
import matplotlib.pyplot as plt

# n = length of each row in data
# d = 1, since there is only one column which isn't the output (y)
# X = first column of data
# y = second column of data
# w = vector of weights for each feature Xi, for a singular feature it is a scalar which is the slope of the linear regression
  
def Linreg_sol(X, y):
    product = X.T @ X
    productInv = np.linalg.inv(product)
    
    w = productInv @ X.T @ y
    return w[0][0]

data = np.array([[0.4, 3.4], [0.95, 5.8], [0.16, 2.9], [0.7, 3.6], [0.59, 3.27], [0.11, 1.89], [0.05, 4.5]])

# ------------------------- IMPLEMENTATION ------------------------------- #
X = (data[:,0]).reshape((len(data[:,0]), 1))
y = (data[:,1]).reshape((len(data[:,1]), 1))

#calculate mean array, mean[0]= mean of X, mean[1] = mean of y
mean = np.array([np.mean(X),np.mean(y)])

#center data
X = X - mean[0]
y = y - mean[1]
# --------------------------------$--------------------------------------- #

w = Linreg_sol(X, y)

# Restore the original line. if y'=wx' (after removing bias) than y-u_y = w(x-u_x), isolate y.
print(f'The linear line is y={w:.2f}*(x-{mean[0]:.2f})+{mean[1]:.2f}')

#--------------------------------------------------------------------------#
# x = np.arange(-0.01, 1, 0.01)
# y = w * (x - mean[0]) + mean[1]
# plt.plot(x,y)

# plt.scatter(data[:,0], data[:,1], color='blue', label='Data')
# plt.show()
#--------------------------------------------------------------------------#


# ------------------------ DOES THE LINE FIT THE DATA? ------------------------ #
# The line does not fit the data precisely but you could see a trend in the     #
# linear regression solution to try and approximate the output given an input   #
# ----------------------------------------------------------------------------- #



# ----------------------------------------------------------------------------- #
# Calculate mean array, mean[0]= mean of X, mean[1] = mean of y
mean = np.array([np.mean(data[:,0]),np.mean(data[:,1])])
# Calculate standrad deviation array, std[0] = std of X, std[1] = std of y
std = np.array([np.std(data[:,0]),np.std(data[:,1])])

# Implement the standardization scaling on the data
newData = (data - mean) / std

newX = (newData[:,0]).reshape((len(newData[:,0]), 1))
newy = (newData[:,1]).reshape((len(newData[:,1]), 1))
w = Linreg_sol(newX, newy)

# Restore the original line. if y'=wx' (after standardization) than (y-u_y)/std_y = w(x-u_x)/std_x, isolate y.
print(f'The linear line is y=({w:.2f}*((x-{mean[0]:.2f})/{std[0]:.2f})*{std[1]:.2f}+{mean[1]:.2f})')
# ----------------------------------------------------------------------------- #



# ----------------------------------------------------------------------------- #
x1 = np.arange(-0.01, 1, 0.01)
y1 = w * (x1 - mean[0]) + mean[1]
plt.plot(x1,y1, color = 'b', label='Regular')

plt.scatter(data[:,0], data[:,1], color='blue', label='Data')

x = np.arange(-0.01, 1, 0.01)
y = w * (x - mean[0]) * std[1] / std[0] + mean[1]
plt.plot(x,y, color = 'r', label = 'Standardized')
plt.show()
# ----------------------------------------------------------------------------- #




# --------------------------- IS THE RESULT BETTER ---------------------------- #
# The result is not that different for the better in my opinon, probably due to
# the data being really small.
# ----------------------------------------------------------------------------- #


# ---------------------------- PRINTING OUTLIERS ------------------------------ #
data = np.array([[0.4, 3.4], [0.95, 5.8], [0.16, 2.9], [0.7, 3.6], [0.59, 3.27], [0.11, 1.89], [0.05, 4.5]])








# ---------------------------- REMOVAL OF OUTLIERS ----------------------------- #
# NEED TO GET THE STANDARD DERIVATION OF THE Y AXIS AND REMOVE POINTS THAT 
# EXCCEED 1 STANDARD DEVIATION
# need to find how to refer to f(x) - STILL NOT DONE


# set y to be the vector for yi
y = (data[:,0]).reshape((len(data[:,0]), 1))

#threshold is 1 standard deviation of the y-axis
threshold = np.std(y)
distancesIndices = np.array([])
data1 = w * (data[:,0] - mean[0]) * std[1] / std[0] + mean[1]
#find all indices of outliers and store them, also print corresponding data 
for i in range(len(data)):
    if (np.abs( data1[i] - data[i][1])) > threshold:
        distancesIndices = np.append(distancesIndices,i)
        print(data[i])

#converting to int array
distancesIndices = distancesIndices.astype(int)
    
# np.delete(data, i, axis = 0)
# ------------------------------------------------------------------------------ #