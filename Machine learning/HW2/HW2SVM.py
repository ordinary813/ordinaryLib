import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import requests
from io import BytesIO

def load_npy_file(url):
  response = requests.get(url)
  if response.status_code == 200:
    npy_data = np.load(BytesIO(response.content), allow_pickle=True).item()
    return npy_data
  else:
    return None

# 2 features, 1 label
data_dict = load_npy_file('https://sharon.srworkspace.com/ml/datasets/hw2/svm_data_2d.npy')

# Access the data as needed
X_train = data_dict['X_train']
y_train = data_dict['y_train']
X_val = data_dict['X_val']
y_val = data_dict['y_val']

# this function gets a feature array and returns its ellipsoid kernel
# features = [x1,x2]
# new_features = [x1^2, x2^2, x1, x2]
# ----------------------------------------------- #
def ellipsify(features):
  new_features = []

  for row in features:
    squaredValues = [val ** 2 for val in row]
    squaredValues.extend(row)
    new_features.append(squaredValues)

  # convert to numpy
  new_features = np.asarray(new_features)
  return new_features

# ----------------------------------------------- #
new_features = ellipsify(X_train)

model = SVC(kernel='linear', C=10)
model.fit(new_features, y_train)

# Get the hyperplane equation coefficients and intercept
coefficients = model.coef_[0]
intercept = model.intercept_

# Print the hyperplane equation
equation_parts = []
for i in range(len(coefficients)):
    equation_parts.append(f"({coefficients[i]:.3f} * X{i+1})")
equation = " + ".join(equation_parts) + f" + ({intercept[0]:.3f})"

print("Hyperplane equation:")
print(f"  {equation}")

train_features = ellipsify(X_train)
train_preds = model.predict(train_features)
train_acc = np.mean(train_preds == y_train)

val_features = ellipsify(X_val)
val_preds = model.predict(val_features)
val_acc = np.mean(val_preds == y_val)

xx, yy = np.meshgrid(np.arange(-2, 2.2, 0.1), np.arange(-2, 2.2, 0.1))
data = np.c_[xx.ravel(), yy.ravel()]

# have to see what's wrong here
new_features = ellipsify(data)
Z = model.predict(new_features)
Z = Z.reshape(xx.shape)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Plot the training data on the first subplot
axs[0].contourf(xx, yy, Z, alpha=0.8)
scatter1 = axs[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title(f'Train dataset - {train_acc:.4f} accuracy')

# Plot the validation data on the second subplot
axs[1].contourf(xx, yy, Z, alpha=0.8)
scatter2 = axs[1].scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='bwr')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_title(f'Validation dataset - {val_acc:.4f} accuracy')

plt.show()