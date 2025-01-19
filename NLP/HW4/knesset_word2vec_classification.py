import json
import re
import argparse, os

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_corpus",
        type=str,
        help="Path to the corpus jsonl file."
    )

    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model."
    )

    args = parser.parse_args()

    corpus_path = args.path_to_corpus
    model_path = args.model_path

    # ------------- code from HW 3 -------------
    df = pd.read_json(corpus_path, lines=True)
    speaker_counts = df['speaker_name'].value_counts()

    # Get the two speakers with the most sentences
    top_speakers = speaker_counts.head(2)


    def _matches_speaker_name(speaker_name, name_in_data):
        if not speaker_name or not name_in_data:
            return False

        name_parts = speaker_name.strip().split()
        data_parts = name_in_data.strip().split()

        if len(name_parts) > 4 or len(data_parts) > 4:
            return False

        # Iterate over all parts of the name
        for i, part in enumerate(name_parts):
            if i >= len(data_parts):
                return False

            # Match initials with full names
            if len(part) == 1 and data_parts[i].startswith(part):
                continue
            if len(data_parts[i]) == 1 and part.startswith(data_parts[i]):
                continue

            # Match quoted or abbreviated initials
            if re.fullmatch(rf"{re.escape(part[0])}['\"׳`]?", data_parts[i]):
                continue
            if re.fullmatch(rf"{re.escape(data_parts[i][0])}['\"׳`]?", part):
                continue

            # Full name matching
            if part != data_parts[i]:
                return False

        return True


    # Function to find all variations of a speaker's name
    def get_speaker_variations(df, target_speaker):
        variations = []
        for name_in_data in df['speaker_name'].unique():
            if _matches_speaker_name(target_speaker, name_in_data):
                variations.append(name_in_data)
        return variations


    # Find variations for each speaker
    first_speaker_variations = get_speaker_variations(df, top_speakers.index[0])
    second_speaker_variations = get_speaker_variations(df, top_speakers.index[1])

    # Normalize the speaker names to the original two speakers
    normalization_dict = {
        **{variation: top_speakers.index[0] for variation in first_speaker_variations},
        **{variation: top_speakers.index[1] for variation in second_speaker_variations}
    }

    # replace the speaker variation to the main name in the DataFrame
    df['speaker_name'] = df['speaker_name'].map(normalization_dict).fillna(df['speaker_name'])

    # Map normalized speaker names to numeric labels
    df['label'] = df['speaker_name'].apply(lambda x: 0 if x == top_speakers.index[0] else
    (1 if x == top_speakers.index[1] else 2))

    # Filter the DataFrame for the top two speakers (binary classification)
    binary_df = df[df['speaker_name'].isin(top_speakers.index)].copy()

    # Binary Classification - Down-Sampling
    binary_class_counts = binary_df['label'].value_counts()
    min_binary_class_size = binary_class_counts.min()  # get the smaller class

    # Down-sample each class to the size of the smallest class
    binary_df_downsampled = pd.concat([
        group.sample(min_binary_class_size, random_state=42)
        for _, group in binary_df.groupby('label')
    ]).reset_index(drop=True)

    #------------------------------------------#

    model = Word2Vec.load(model_path)
    sentence_embeddings = []
    #calc sentence embedding only for the down sampled data sentences
    for _, row in binary_df_downsampled.iterrows():
        sentence = row['sentence_text']

        # Tokenize the sentence (remove numbers and punctuation)
        tokens = re.findall(r'["\u0590-\u05FF]+|"\u0590-\u05FF+"', sentence)

        # Filter tokens with less than 2 characters and keep only known words
        clean_tokens = [token for token in tokens if len(token) > 1 and token in model.wv]

        # Compute the sentence embedding as the mean of word vectors in the sentence
        if clean_tokens:  # Avoid division by zero for empty lists
            word_vectors = [model.wv[token] for token in clean_tokens]
            sentence_embedding = np.mean(word_vectors, axis=0)
        else:
            sentence_embedding = np.zeros(model.vector_size)

        # Append the result to the list
        sentence_embeddings.append(sentence_embedding)

    #train KN classifier

    X_binary_downsampled = np.array(sentence_embeddings)
    y_binary_downsampled = binary_df_downsampled['label']

    # KNN classifier
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric='minkowski',
        weights='uniform'
    )


    # Evaluate using 5-fold cross-validation and generate classification reports
    def evaluate_classifier(classifier, X, y, classifier_name, task_name):
        # Get predictions using cross-validation
        y_pred = cross_val_predict(classifier, X, y, cv=5, method='predict')
        # Determine the number of classes
        num_classes = len(set(y))

        # Set target names based on the number of classes
        if num_classes == 2:
            target_names = ["first", "second"]
        else:
            raise ValueError("Unexpected number of classes!")

        print(classification_report(y, y_pred, target_names=target_names, zero_division=0))


    evaluate_classifier(knn, X_binary_downsampled, y_binary_downsampled, 'KNN with sentence embeddings', 'Binary Classification')