import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import re, os
import random
import numpy as np
import argparse

random.seed(42)
np.random.seed(42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_corpus",
        type=str,
        help="Path to the corpus jsonl file."
    )

    parser.add_argument(
        "sentences_file",
        type=str,
        help="Path to the sentences file."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="Path to the directory for the output files. Default is local directory."
    )

    args = parser.parse_args()

    corpus_path = args.path_to_corpus
    sentences_file = args.sentences_file
    out_dir = args.output_dir

    # ------------- part 1 -------------
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

    # Create a new DataFrame for multi-class classification (with "Other")
    multi_class_df = df.copy()

    # ------------- part 2 -------------
    # Binary Classification - Down-Sampling
    binary_class_counts = binary_df['label'].value_counts()
    min_binary_class_size = binary_class_counts.min()  # get the smaller class

    # Down-sample each class to the size of the smallest class
    binary_df_downsampled = pd.concat([
        group.sample(min_binary_class_size, random_state=42)
        for _, group in binary_df.groupby('label')
    ]).reset_index(drop=True)

    # Multi-Class Classification - Down-Sampling
    multi_class_counts = multi_class_df['label'].value_counts()
    min_multi_class_size = multi_class_counts.min()  # get the smaller class

    # Down-sample each class to the size of the smallest class
    multi_class_df_downsampled = pd.concat([
        group.sample(min_multi_class_size, random_state=42)
        for _, group in multi_class_df.groupby('label')
    ]).reset_index(drop=True)

    # ------------- part 3 -------------

    # BOW with TFIDF
    vectorizer = TfidfVectorizer(max_features=1000)

    X_tfidf_binary = vectorizer.fit_transform(binary_df_downsampled['sentence_text'])
    X_tfidf_multi = vectorizer.fit_transform(multi_class_df_downsampled['sentence_text'])


    # Vector of sentence qualities
    def get_sentence_length(sentence):
        return len(sentence.split())


    def punctuation_count(sentence):
        punctuation_marks = ['.', ',', '?', '!', ':', ';']
        return sum(1 for char in sentence if char in punctuation_marks)


    def create_feature_vector(df):
        # Create features: sentence length, punctuation count, sentiment score
        df['sentence_length'] = df['sentence_text'].apply(get_sentence_length)
        df['punctuation_count'] = df['sentence_text'].apply(punctuation_count)

        # Add protocol type information
        df['protocol_type'] = df['protocol_type'].apply(lambda x: 1 if x == 'plenary' else 0)  # Convert 'plenary' to 1

        # Knesset number
        df['knesset_number'] = df['knesset_number']

        # Create the feature vector
        features = df[['sentence_length', 'punctuation_count', 'protocol_type', 'knesset_number']]

        return features


    # Use the data to create the feature vector
    binary_features = create_feature_vector(binary_df_downsampled)
    multi_class_features = create_feature_vector(multi_class_df_downsampled)

    # ------------- part 4 -------------

    # KNN classifier
    knn = KNeighborsClassifier(
        n_neighbors=5,
        metric='minkowski',
        weights='uniform'
    )

    # Logistic Regression classifier
    log_reg = LogisticRegression(max_iter=1000, solver='lbfgs')

    # Standardize custom features
    scaler = StandardScaler()
    binary_features_scaled = scaler.fit_transform(binary_features)
    multi_class_features_scaled = scaler.fit_transform(multi_class_features)


    # Evaluate using 5-fold cross-validation and generate classification reports
    def evaluate_classifier(classifier, X, y, classifier_name, task_name):
        # Get predictions using cross-validation
        y_pred = cross_val_predict(classifier, X, y, cv=5, method='predict')
        # Determine the number of classes
        num_classes = len(set(y))

        # Set target names based on the number of classes
        if num_classes == 2:
            target_names = ["first", "second"]
        elif num_classes == 3:
            target_names = ["first", "second", "other"]
        else:
            raise ValueError("Unexpected number of classes!")

        # We added the prints to the report
        print(f"\nClassification Report for {task_name} - {classifier_name}:\n")
        print(classification_report(y, y_pred, target_names=target_names, zero_division=0))


    # Binary Classification
    y_binary = binary_df_downsampled['label']

    evaluate_classifier(knn, X_tfidf_binary, y_binary, 'KNN with TF-IDF', 'Binary Classification')
    evaluate_classifier(log_reg, X_tfidf_binary, y_binary, 'Logistic Regression with TF-IDF', 'Binary Classification')

    evaluate_classifier(knn, binary_features_scaled, y_binary, 'KNN with Custom Features', 'Binary Classification')
    evaluate_classifier(log_reg, binary_features_scaled, y_binary, 'Logistic Regression with Custom Features',
                        'Binary Classification')

    # Multi-Class Classification
    y_multi = multi_class_df_downsampled['label']

    evaluate_classifier(knn, X_tfidf_multi, y_multi, 'KNN with TF-IDF', 'Multi-Class Classification')
    evaluate_classifier(log_reg, X_tfidf_multi, y_multi, 'Logistic Regression with TF-IDF', 'Multi-Class Classification')

    evaluate_classifier(knn, multi_class_features_scaled, y_multi, 'KNN with Custom Features', 'Multi-Class Classification')
    evaluate_classifier(log_reg, multi_class_features_scaled, y_multi, 'Logistic Regression with Custom Features',
                        'Multi-Class Classification')

    # ------------- part 5 -------------
    # Read the sentences
    with open(sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    # Transform the sentences using the TF-IDF vectorizer
    X_tfidf_sentences = vectorizer.transform(sentences)

    # Use the Logistic Regression model trained on TF-IDF for multi-class classification
    log_reg.fit(X_tfidf_multi, y_multi)

    # Predict the labels for the new sentences
    predictions = log_reg.predict(X_tfidf_sentences)

    # Map numeric labels to the required text labels
    label_mapping = {0: "first", 1: "second", 2: "other"}
    predictions_text = [label_mapping[label] for label in predictions]

    # Write the predictions to the output file
    with open(os.path.join(out_dir,'classification_results.txt'), 'w', encoding='utf-8') as f:
        for prediction in predictions_text:
            f.write(prediction + '\n')