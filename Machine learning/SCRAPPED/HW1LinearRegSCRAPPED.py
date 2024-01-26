import numpy as np
import matplotlib.pyplot as plt

# n = length of each row in data
# d = 1, since there is only one column which isn't the output (y)
# X = first column of data
# y = second column of data
# w = vector of weights for each feature Xi, for a singular feature it is a scalar which is the slope of the linear regression
  
def Linreg_sol(X, y):
   
    # ------------------------------------------- TECHINCALITY --------------------------------------------------------#
    # X is an array in the code (in the real calculation for w it is a COLUMN (1xn and not nx1)
    # it means that when calculating (X^T*X) the product is going to be of dimensions nxn and not dxd (1x1 in that case)
    # because X^T is a nxd array, X is a dxn array
    # therefore in the code we will switch between them (X.TRANSPOSE <=> X)
    #------------------------------------------------------------------------------------------------------------------#
    
    # _______________________________________________________________________________________________________ #
    X_T = np.reshape(X,(len(X),1))              # X_T is of dimensions nx1, X is of dimensions 1xn
    y_for_calc = np.reshape(y,(len(y),1))       # I want y to be of dimensions 1xn for the calculation
    product = X @ X_T                           # product of X and X transpose
    
    if(product.shape != (1,)):                  # Checking if the product is of dimensions 1x1
        multiInv = np.linalg.inv(X @ X_T)       # since we can't invert a 1x1 matrix
        w = multiInv @ X @ y_for_calc
        return w
    else:
        multiInv = product[0]                   # disclaimer: for this exercise it should always end up here
        w = multiInv * X @ y_for_calc           # since there is only 1 feature
        return w[0]                             # 
    # _______________________________________________________________________________________________________ #
    # need to implement w = (X^T*X)-1(X^T)*y
    
    # _______________________________________________________________________________________________________ #
    # product = X.T @ X                               # this results in a 1x1 array

    # if(product.shape != (1,)):                      # the built-in function for matrix inverse does not manage a 1x1 matrix
    #     multiInv = multiInv = np.linalg.inv(product)
    #     w = multiInv @ X.T @ y
    # else:
    #     w = 1
    # return w
    # _______________________________________________________________________________________________________ #
data = np.array([[0.4, 3.4], [0.95, 5.8], [0.16, 2.9], [0.7, 3.6], [0.59, 3.27], [0.11, 1.89], [0.05, 4.5]])

# ------------------------- IMPLEMENTATION ------------------------------- #
X = data[:,0]
y = data[:,1]


w=[0,3,4]
w.reshape(w.shape,1)
#test#
# X = np.array([1,2,3,4,5,6])
# y = np.array([15,16,17,18,19,20])

# print(((X.T @ X)))

#calculate mean array, mean[0]= mean of X, mean[1] = mean of y
mean = np.array([np.mean(X),np.mean(y)])

#center data
X = X - mean[0]
y = y - mean[1]
# --------------------------------$--------------------------------------- #

w = Linreg_sol(X, y)

# Restore the original line. if y'=wx' (after removing bias) than y-u_y = w(x-u_x), isolate y.
print(f'The linear line is y={w:.2f}*(x-{mean[0]:.2f})+{mean[1]:.2f}')

x = np.arange(-0.01, 1, 0.01)
y = w * (x - mean[0]) + mean[1]
plt.plot(x,y)

plt.scatter(data[:,0], data[:,1], color='blue', label='Data')
plt.show()

# ------------------------ DOES THE LINE FIT THE DATA? ------------------------ #
# The line does not fit the data precisely but you could see a trend in the     #
# linear regression solution to try and approximate the output give an input    #
# ----------------------------------------------------------------------------- #

# Calculate mean array, mean[0]= mean of X, mean[1] = mean of y
mean = np.array([np.mean(data[:,0]),np.mean(data[:,1])])
# Calculate standrad deviation array, std[0] = std of X, std[1] = std of y
std = np.array([np.std(data[:,0]),np.std(data[:,1])])

# Implement the standardization scaling on the data
newData = (data - mean) / std

newX = newData[:,0]
newy = newData[:,1]
w = Linreg_sol(newX, newy)

# Restore the original line. if y'=wx' (after standardization) than (y-u_y)/std_y = w(x-u_x)/std_x, isolate y.
print(f'The linear line is y=({w:.2f}*((x-{mean[0]:.2f})/{std[0]:.2f})*{std[1]:.2f}+{mean[1]:.2f})')

x = np.arange(-0.01, 1, 0.01)
y = w * (x - mean[0]) * std[1] / std[0] + mean[1]
plt.plot(x,y)

plt.scatter(data[:,0], data[:,1], color='blue', label='Data')
plt.show()