import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def train(tweets, labels):
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(tweets)

    uniqueWords = vectorizer.get_feature_names_out()

    Pw = np.zeros((len(set(labels)),len(uniqueWords)))

    for i,label in enumerate(set(labels)):
        # rows where label appears in the data
        label_indices = [idx for idx, cat in enumerate(labels) if cat == label]
        label_word_counts = data[label_indices, :].sum(axis=0)
        total_words_in_label = label_word_counts.sum()
        Pw[i, :] = label_word_counts / total_words_in_label

    samplesSize =  len(labels)
    cat = cat
    P = dict(zip())

    return Pw, P


labels = ['bad',
          'bad',
          'good',
          'bad',
          'ok',
          'good']

tweets = ['last night i was this',
          'hey twitter how are you doing',
          'how was i supposed to know',
          'this is how you do it',
          'if i had a twitter i would have to know',
          'man this was suppsoed to be the last']
Pw, P = train()
print(Pw)