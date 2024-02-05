import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import seaborn as sns

# reading data and giving names to the columns, for the sake of readability
df = pd.read_csv('https://sharon.srworkspace.com/ml/datasets/hw2/exams.csv', header=None)
df = df.rename(columns={0: "Test 1", 1: "Test 2", 2: "Admission"})

features = df.columns.values[:-1]
plt.figure(figsize=(12, 6))
sns.set_style('whitegrid')

for i, feature in enumerate(features, 1):
    plt.subplot(1, 2, i)
    plt.hist(df[feature], density=True, bins=25, alpha=0.7, label=feature)
    
    sns.kdeplot(np.array(df[feature]), bw=0.5, color='RoyalBlue')
    plt.title(f'Density of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
plt.show()