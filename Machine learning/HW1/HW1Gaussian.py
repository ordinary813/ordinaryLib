import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# ------------------------------------ 1 ------------------------------------ #
df = pd.read_csv("https://sharon.srworkspace.com/ml/datasets/hw1/wine.data.csv")

# 178 samples, 14 features each
print(df.shape)
print(df.head(5))
# ____________________________________ $ ____________________________________ #


# ------------------------------------ 2 ------------------------------------ #
# df.plot(kind='density', subplots=True, layout=(4,4), figsize=(18, 15), sharex=False)
# plt.show()
# ____________________________________ $ ____________________________________ #



# -------------------------------- QUESTION --------------------------------- #
# there is potential for gaussian bayes since all features distribute like normal
# distribution more or less
# ____________________________________ $ ____________________________________ #



# ------------------------------------ 3 ------------------------------------ #
X = df.drop('Class', axis=1)         # Feature matrix
y = df['Class']                      # Output vector     

X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25, stratify=y)
# ____________________________________ $ ____________________________________ #


# ------------------------------------ 4 ------------------------------------ #

# compute the likelihood of the sample x being distributed with a normal distribution,
# mean vector 'mean', and covariance matrix 'covariance'
def estimateLikelihood(x, mean, covariance):
    d = len(mean)                           # amount of features (14)
    
    #compute the multivariate gaussian pdf
    detCov = np.linalg.det(covariance)      # determinant of covariance matrix
    invCov = np.linalg.inv(covariance)      # inverse matrix of covariance matrix
    constant = 1 / ((2 * np.pi) * np.sqrt(detCov))

    exponent = -0.5 * np.dot(np.dot((x - mean).T, invCov), (x - mean))
    return constant * np.exp(exponent)

# Get a test sample x and return it's predicted Class
def classify_point_gaussian_bayes(x):
    # I'm copying the data so the original data will not change for any reason
    scores = []
    samplesTrain = X_train
    classesTrain = y_train

    classes = np.unique(y_train)        # number of classes

    for c in classes:
        X_c = X_train[y_train == c]         # array of samples that are from class c

        mean_c = np.mean(X_c, axis=0)       # mean vector of the current class (mean of each feature)
        cov_c = np.cov(X_c, rowvar=False)   # compute covariance matrix for current class (cov matrix of all features for X_c)
        prior_c = len(X_c) / len(X_train)   # compute prior for the current class (amount of appearances of class out of the entire data)

        likelihood = estimateLikelihood(x, mean_c, cov_c)
        score = likelihood * prior_c
        # score = np.log(likelihood) + np.log(prior_c)
        scores.append(score)
    
    predicted_class = classes[np.argmax(scores)]
    return predicted_class    
        
def classify_point_gaussian_naive_bayes(x):

    # I'm copying the data so the original data will not change for any reason
    samplesTrain = X_train
    classesTrain = y_train
    
    scores = []
    classes = np.unique(y_train)        # number of classes

    for c in classes:
        X_c = X_train[y_train == c]         # array of samples that are from class c

        mean_c = np.mean(X_c, axis=0)       # mean vector of the current class (mean of each feature)
        cov_c = np.cov(X_c, rowvar=False)   # compute covariance matrix for current class (cov matrix of all features for X_c)
        prior_c = len(X_c) / len(X_train)   # compute prior for the current class (amount of appearances of class out of the entire data)

        likelihood = estimateLikelihood(x, mean_c, np.diag(np.diag(cov_c)))    # finding pdf, for naive bayes I'm sending the diagonal matrix of cov_c, 
                                                                    # it should give us the wanted result
        score = likelihood * prior_c
        scores.append(score)
        
    
    predicted_class = classes[np.argmax(scores)]
    return predicted_class
# ____________________________________ $ ____________________________________ #

res = []
for idx, test_point in enumerate(X_test):
  print(f'current test point:\n {pd.DataFrame(test_point.reshape(1,13),columns=df.drop("Class", axis=1).columns.values)}')
  print(f'Class is: {y_test[idx]}')
  res.append(classify_point_gaussian_bayes(test_point) == y_test[idx])
print(f'Test accuracy for gaussian bayes is {res.count(True)/len(res)}')

res = []
for idx, test_point in enumerate(X_test):
  res.append(classify_point_gaussian_naive_bayes(test_point) == y_test[idx])
print(f'Test accuracy for gaussian naive bayes is {res.count(True)/len(res)}')

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()

X_train_scaled = standard_scaler.fit_transform(X_train)
X_test_scaled = standard_scaler.transform(X_test)

res = []
for idx, test_point in enumerate(X_test_scaled):
  res.append(classify_point_gaussian_bayes(test_point) == y_test[idx])
print(f'Test accuracy for gaussian bayes is {res.count(True)/len(res)}')

res = []
for idx, test_point in enumerate(X_test_scaled):
  res.append(classify_point_gaussian_naive_bayes(test_point) == y_test[idx])
print(f'Test accuracy for gaussian naive bayes is {res.count(True)/len(res)}')