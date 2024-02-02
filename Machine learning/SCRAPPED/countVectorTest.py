from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

# doc = [['hi','hello','hi','what'],
#        ['no','yes','is','hi','what'],
#        ['how','yes','hi']]

doc = [
    "hi hello hi what",
    "no yes is hi what",
    "how yes hi"
]
vectorizer = CountVectorizer()
vectorizer.fit(doc)

print(f'Vocabulary:\n{vectorizer.vocabulary_}')

vector = vectorizer.transform(doc)

print(vector.toarray())