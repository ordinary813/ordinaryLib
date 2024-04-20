import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
df = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/dataset2.csv')

x = df['ear_length'].to_numpy()
y = df['animal_type'].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

def find_prior(array, class_name):
    number_of_dogs = np.count_nonzero(array == class_name)
    return (number_of_dogs / len(array))

def find_mle_mean(features, lables, class_name):
    sum = 0; size = 0
    for i in range(len(features)):
      if (lables[i] == class_name):
        sum += features[i]
        size += 1
    return (1/size)*sum

dogs_prior = find_prior(y_train, "dog")
cats_prior = find_prior(y_train, "cat")
dogs_mle = find_mle_mean(x_train, y_train, "dog")
cats_mle = find_mle_mean(x_train, y_train, "cat")

def calc_map(x, mu, sigma, prior = 0.5):
    coefficient = 1/(sigma*(np.sqrt(2*np.pi)))
    exponent = -0.5 * (((x - mu)/sigma) ** 2)
    likelihood = (coefficient)*((np.exp(exponent)))
    return likelihood * prior

def test(test_data, dogs_mle, dogs_prior, cats_mle, cats_prior):
    predictions = np.empty(len(test_data), dtype="object")
    for i in range(len(test_data)):
      if (calc_map(test_data[i], dogs_mle, 0.5, dogs_prior) > calc_map(i, cats_mle, 0.5, cats_prior)):
        predictions[i] = "dog"
      else:
        predictions[i] = "cat"
    return predictions


count = 0
predictions = test(x_test, dogs_mle, dogs_prior, cats_mle, cats_prior)


for i in range(len(predictions)):
  if (predictions[i] == y_test[i]):
    count += 1

acc = count / len(predictions)
print(acc)