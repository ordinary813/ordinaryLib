import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# splitting data into 80/20, I'm assuming I can't use sklearn

# shuffles the dataframe for randomness of the split
shuffled_df = df.sample(frac=1, random_state=21)

# choose the index to split on 80% of the data
split_index = int(0.8 * len(shuffled_df))

# split the data into train and test samples
train_df = shuffled_df[:split_index]
test_df = shuffled_df[split_index:]

# Separate features and the labels (Class)
# X_train and test are of dimensions |samples| * 13
# y_train and test are of dimensions |samples|
X_train = train_df.drop('Class', axis=1).values
y_train = train_df['Class'].values

X_test = test_df.drop('Class', axis=1).values
y_test = test_df['Class'].values
# ____________________________________ $ ____________________________________ #


# ------------------------------------ 4 ------------------------------------ #

# compute the likelihood of the sample x being distributed for a normal distribution
# with mean vector mean, and covariance matrix covariance
def findPDF(x, mean, covariance):
    d = len(mean)                           # amount of features (14)
    
    #compute the multivariate gaussian pdf
    detCov = np.linalg.det(covariance)      # determinant of covariance matrix
    invCov = np.linalg.inv(covariance)      # inverse matrix of covariance matrix
    constant = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(detCov))

    exponent = -0.5 * np.dot(np.dot((x - mean).T, invCov), (x - mean))
    return constant * np.exp(exponent)

# Get a test sample x and return it's predicted Class
def classify_point_gaussian_bayes(x):
    # I'm copying the data so the original data will not change for any reason
    scores = []
    samplesTrain = X_train
    classesTrain = y_train
    
    classes = np.unique(y_train)        # number of classes
    class_means = []                    # mean vector for each class
    class_covariances = []              # covariance matrices for each class
    class_priors = []                   # the probability of a class out of the entire data

    for c in classes:
        X_c = X_train[y_train == c]         # array of samples that are from class c
        mean_c = np.mean(X_c, axis=0)       # mean vector of the current class
        cov_c = np.cov(X_c, rowvar=False)   # compute covariance matrix for current class
        prior_c = len(X_c) / len(X_train)   # compute prior for the current class

        likelihood = findPDF(x, mean_c, cov_c)
        score = np.log(likelihood) + np.log(prior_c)
        scores.append(score)
        
        # class_means.append(mean_c)          # add mean vector of the current class
        # class_covariances.append(cov_c)     # add covariance matrix for current class
        # class_priors.append(prior_c)        # add prior of current class
    
    predicted_class = classes[np.argmax(scores)]
    return predicted_class    
        
def classify_point_gaussian_naive_bayes(x):

    # I'm copying the data so the original data will not change for any reason
    samplesTrain = X_train
    classesTrain = y_train
    
    scores = []
    classes = np.unique(y_train)        # number of classes
    class_means = []                    # mean vector for each class
    class_covariances = []              # covariance matrices for each class
    class_priors = []                   # the probability of a class out of the entire data

    for c in classes:
        X_c = X_train[y_train == c]         # array of samples that are from class c
        mean_c = np.mean(X_c, axis=0)       # mean vector of the current class
        cov_c = np.cov(X_c, rowvar=False)   # compute covariance matrix for current class
        prior_c = len(X_c) / len(X_train)   # compute prior for the current class

        likelihood = likelihood(x,mean_c, np.diag(np.diag(cov_c)))
        likelihood = np.prod(1 / np.sqrt(2 * np.pi * cov_c) * np.exp(-(x - mean_c)**2 / (2 * cov_c)))
        score = np.log(likelihood) + np.log(prior_c)
        scores.append(score)
        
    
    predicted_class = classes[np.argmax(scores)]
    return predicted_class
# ____________________________________ $ ____________________________________ #

res = []
for idx, test_point in enumerate(X_test):
  res.append(classify_point_gaussian_bayes(test_point) == y_test[idx])
print(f'Test accuracy for gaussian bayes is {res.count(True)/len(res)}')

res = []
for idx, test_point in enumerate(X_test):
  res.append(classify_point_gaussian_naive_bayes(test_point) == y_test[idx])
print(f'Test accuracy for gaussian naive bayes is {res.count(True)/len(res)}')