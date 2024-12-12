import pandas as pd
import random
import os
import re
import json
import math
from math import log


class Trigram_LM:
    def __init__(self, protocol_type, file_path):
        self.protocol_type = protocol_type
        self.corpus = None  # DataFrame to store the relvant sentences from corpus
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}
        self.load_corpus(file_path) # load the corpus
        self.calculate_counts() #calc bigrams, trigrams and unigrams

    def load_corpus(self, file_path):
        try:
            df = pd.read_json(file_path, lines=True)

            # Filter the DataFrame based on the protocol type
            self.corpus = df[df['protocol_type'] == self.protocol_type]

        except Exception as e:
            print(f"Error loading corpus: {e}")

    # Counts appearances of every collocation of size 1,2,3 in the corpus
    def calculate_counts(self):
        # Iterate over each row in the corpus
        for index, row in self.corpus.iterrows():
            sentence = row['sentence_text']
            tokens = sentence.split()  # Tokenize the sentence

            # Calculate unigram count frequencies
            for token in tokens:
                if token not in self.unigrams:
                    self.unigrams[token] = 0
                self.unigrams[token] += 1

            # Calculate bigram count frequencies
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                if bigram not in self.bigrams:
                    self.bigrams[bigram] = 0
                self.bigrams[bigram] += 1

            # Calculate trigram count frequencies
            for i in range(len(tokens) - 2):
                trigram = (tokens[i], tokens[i + 1], tokens[i + 2])
                if trigram not in self.trigrams:
                    self.trigrams[trigram] = 0
                self.trigrams[trigram] += 1

    # Input: sentence
    # Output: log probability of the sentence to appear in the corpus
    def calculate_prob_of_sentence(self, sentence):
        tokens = sentence.split()
        tokens = ["<s_0>", "<s_1>"] + tokens  # Add start tokens
        log_prob = 0

        # weights for the interpolation (give more value to trigram that has more context)
        lambda_1 = 0.7
        lambda_2 = 0.2
        lambda_3 = 0.1

        V = len(self.unigrams)  # number of unique words in the corpus

        # Start from the third token
        for i in range(2, len(tokens)):
            unigram = tokens[i]  # the current token
            bigram = (tokens[i - 1], tokens[i])  # current and previous tokens
            trigram = (tokens[i - 2], tokens[i - 1], tokens[i])  # current and two previous tokens

            # Add 1 for Laplace smoothing
            trigram_count = self.trigrams.get(trigram, 0) + 1
            bigram_count = self.bigrams.get(bigram, 0) + 1
            unigram_count = self.unigrams.get(unigram, 0) + 1

            # Add V for smoothing
            trigram_probability = trigram_count / (self.bigrams.get((tokens[i - 2], tokens[i - 1]), 0) + V)
            bigram_probability = bigram_count / (self.unigrams.get(tokens[i - 1], 0) + V)
            unigram_probability = unigram_count / (sum(self.unigrams.values()) + V)

            # Apply linear interpolation
            prob = lambda_1 * trigram_probability + lambda_2 * bigram_probability + lambda_3 * unigram_probability

            # Log probability (avoid log(0))
            log_prob += (0 if prob == 0 else math.log(prob))

        return log_prob

    def generate_next_token(self, sentence):

        # Split the input sentence into tokens
        tokens = sentence.split()
        tokens = ["<s_0>", "<s_1>"] + tokens  # Add start tokens

        highest_prob = -float('inf')
        best_token = None

        # weights for the interpolation
        lambda_1 = 0.7
        lambda_2 = 0.2
        lambda_3 = 0.1

        V = len(self.unigrams)  # number of unique words in the corpus

        # Loop through all potential next tokens
        for token in self.unigrams:
            unigram = token
            bigram = (tokens[-1], token)
            trigram = (tokens[-2], tokens[-1], token)

            # Add 1 for Laplace smoothing
            trigram_count = self.trigrams.get(trigram, 0) + 1
            bigram_count = self.bigrams.get(bigram, 0) + 1
            unigram_count = self.unigrams.get(unigram, 0) + 1

            # Add V for smoothing
            trigram_probability = trigram_count / (self.bigrams.get((tokens[-2], tokens[-1]), 0) + V)
            bigram_probability = bigram_count / (self.unigrams.get(tokens[-1], 0) + V)
            unigram_probability = unigram_count / (sum(self.unigrams.values()) + V)

            # Linear interpolation of trigram, bigram, and unigram probabilities
            prob = lambda_1 * trigram_probability + lambda_2 * bigram_probability + lambda_3 * unigram_probability

            # Update the best token
            if prob > highest_prob:
                highest_prob = prob
                best_token = token

        return best_token, highest_prob
    
    

def generate_ngrams(text, n):
        words = re.findall(r'\b\w+\b', text.lower())
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

def compute_idf(protocol_collocations, total_docs):
    doc_counts = {}

    # for every collocation in all documents we increment when we saw the collocation
    # by the end of this loop each collocation should have its frequency in all documents
    for doc_collocations in protocol_collocations:
        for collocation in doc_collocations:
            doc_counts[collocation] = doc_counts.get(collocation, 0) + 1

    idf_scores = {}

    # compute IDF for each collocation
    for collocation, doc_count in doc_counts.items():
        idf_scores[collocation] = log(total_docs / (doc_count))

    return idf_scores


def compute_tf(doc_collocations):
    total_count = len(doc_collocations)
    collocation_counts = {}
    
    # Count occurrences of each collocation in doc_collocations
    for coll in doc_collocations:
        collocation_counts[coll] = collocation_counts.get(coll, 0) + 1
    
    # Compute TF scores
    tf_scores = {coll: count / total_count for coll, count in collocation_counts.items()}

    return tf_scores

def compute_tfidf(tf_scores, idf_scores):
    tfidf_scores = {}
    for coll, tf in tf_scores.items():
        tfidf_scores[coll] = tf * idf_scores.get(coll, 0)
    return tfidf_scores

# Input: 
#   corpus_df: a dataframe containing the corpus' data
#   k: number of top collocations
#   n: length of collocations
#   t: min threshold for the amount of collocations
# Output:
#   collocation:grade list from the corpus
def get_k_n_t_collocations(corpus_df, k, n, t, type):

    #produce all collocations of length n
    corpus_df['collocations'] = corpus_df['sentence_text'].apply(lambda x: generate_ngrams(x, n))

    # place all collocations in a dictionary of structure <Collocation>: <Count>
    collocation_counts = {}
    for coll_list in corpus_df['collocations']:
        for coll in coll_list:
            collocation_counts[coll] = collocation_counts.get(coll, 0) + 1

    if type == "frequency":
        # only include collcations that appear more than <t>
        filtered_collocations = {coll: count for coll, count in collocation_counts.items() if count >= t}
    elif type == "tfidf":

        total_docs = len(corpus_df['protocol_name'].unique())

        # group by protocol docs
        grouped = corpus_df.groupby('protocol_name')['collocations']
        protocol_collocations = grouped.apply(lambda x: sum(x, []))

        idf_scores = compute_idf(protocol_collocations, total_docs)

        tfidf_scores = {}
        collocation_counts = {}

        print("\nProcessing protocols...")

        for protocol_name, collocations in grouped:
            print(f"\rProcessing {protocol_name:<30}", end='', flush=True)
            # list of collocations for the current document/protocol
            doc_collocations = sum(collocations, [])

            for coll in doc_collocations:
                collocation_counts[coll] = collocation_counts.get(coll, 0) + 1

            tf_scores = compute_tf(doc_collocations)

            # Compute TF-IDF for collocations in this document
            tfidf = compute_tfidf(tf_scores, idf_scores)

            for coll, score in tfidf.items():
                tfidf_scores[coll] = tfidf_scores.get(coll, 0) + score

        # Only include collocations that have a score >= t
        filtered_collocations = {coll: score for coll, score in tfidf_scores.items() if collocation_counts.get(coll, 0) >= t}

    sorted_collocations = sorted(filtered_collocations.items(), key=lambda x: x[1], reverse=True)[:k]
    return sorted_collocations

# Input: list of strings and a percentage x
# Output: list of strings after masking x% of the tokens
def mask_tokens_in_sentences(sentences, x):
    masked_sentences = []

    for sentence in sentences:
        tokens = sentence.split()
        num_tokens_to_mask = int(len(tokens) * (x / 100))
        if num_tokens_to_mask == 0:
            num_tokens_to_mask = 1
        tokens_to_mask = random.sample(range(len(tokens)), num_tokens_to_mask)

        masked_tokens = ["[*]" if i in tokens_to_mask else token for i, token in enumerate(tokens)]
        masked_sentences.append(" ".join(masked_tokens))
    return masked_sentences

# Input: a dataframe, amount of entries to mask with [*], and a percentage x
# Output: the dataframe after applying the mask
def mask_corpus(corpus_df, amount_to_mask, x, original_path ='', masked_path=''):
    if amount_to_mask > len(corpus_df):
        amount_to_mask = len(corpus_df)

    # copy dataframe to ensure integrity of the original dataframe
    df_copy = corpus_df.copy()

    more_than_5 = [i for i in range(len(df_copy)) if len(re.findall(r'\b\w+\b', df_copy.iloc[i]['sentence_text'])) >= 5]

    # pick senetences with more than 5 tokens
    mask_indices = random.sample(more_than_5, amount_to_mask)

    sentences_to_mask = [df_copy.iloc[i]['sentence_text'] for i in mask_indices]

    masked_sentences = mask_tokens_in_sentences(sentences_to_mask, x)

    for idx, i in enumerate(mask_indices):
        df_copy.loc[df_copy.index[i], 'sentence_text'] = masked_sentences[idx]
    
    # print only when necessary
    if original_path != '' and masked_path != '':
        with open(original_path, 'w', encoding='utf-8') as original_file, \
            open(masked_path, 'w', encoding='utf-8') as masked_file:
            
            for index in mask_indices:
                original_file.write(f"{corpus_df['sentence_text'].iloc[index]}\n")
                masked_file.write(f"{df_copy['sentence_text'].iloc[index]}\n")

    return sentences_to_mask, masked_sentences

def to_word(n):
    match n:
        case 2:
            return "Two"
        case 3:
            return "Three"
        case 4:
            return "Four"

def calculate_perplexity(trigram_model, original_sentence, masked_sentence, masked_indices):
    total_log_prob = 0
    count = 0

    for index in masked_indices:
        # split to corresponding tokens
        masked_tokens = masked_sentence.split()
        original_tokens = original_sentence.split()

        token = original_tokens[index]
        context = " ".join(masked_tokens[max(0, index - 2):index])

        prob = trigram_model.calculate_prob_of_sentence(f"{context} {token}") / (trigram_model.calculate_prob_of_sentence(context) + 1)

        if prob > 0:
            total_log_prob += -log(prob)
            count += 1
        else:
            print(f"masked sentence prob is 0 for index {index}")

    return math.exp(total_log_prob / count) if count > 0 else float('inf')


def compute_average_perplexity(masked_sentences, original_sentences, trigram_model):
    total_perplexity = 0
    sentence_count = 0

    for masked, original in zip(masked_sentences, original_sentences):
        tokens = masked.split()
        masked_indices = [i for i, token in enumerate(tokens) if token == "[*]"]

        if masked_indices:
            perplexity = calculate_perplexity(trigram_model, original, masked, masked_indices)
            total_perplexity += perplexity
            sentence_count += 1

    return total_perplexity / sentence_count if sentence_count > 0 else float('inf')

def main():
    corpus_path = 'knesset_corpus.jsonl'
    output_file = 'top_collocations.txt'


    #############################################
    ############### Exercise 2 ##################
    #############################################
    print("----- Top collocations section -----")

    k = 10          # top 10 collocations
    ns = [2,3,4]    # n-grams
    t = 5           # minimum of <t> counts for an n-gram
    types = ['frequency','tfidf']

    trigram_model_committee = Trigram_LM("committee", corpus_path)
    trigram_model_plenary = Trigram_LM("plenary", corpus_path)

    #save the output into a file
    with open(output_file, 'w', encoding='utf-8') as f:
        for n in ns:
            f.write(f"{to_word(n)}-gram collocations:\n")
            for score_type in types:
                if score_type == 'frequency':
                    f.write(f"Frequency:\n")
                elif score_type == 'tfidf':
                    f.write(f"TF-IDF:\n")
                
                for model_name, model_corpus in [
                    ("Committee corpus", trigram_model_committee.corpus),
                    ("Plenary corpus", trigram_model_plenary.corpus)
                ]:

                    f.write(f"{model_name}:\n")
                    # Get top collocations for the current configuration
                    result = get_k_n_t_collocations(model_corpus,k, n, t, score_type)
                    for collocation in result:
                        f.write(f"{collocation[0]}\n")
                    f.write("\n")  # Empty line between sections
            f.write("\n")  # Empty line between n-gram categories

        print(f"\nComplete, Output file: {output_file}", flush=True)

    #############################################
    ############### Exercise 3 ##################
    #############################################
    print("----- Token masks section -----")
    original_ouput_path = 'original_sampled_sents.txt'
    masked_ouput_path = 'masked_sampled_sents.txt'

    # print to file and get original + masked sentences
    original_sens, masked_sens = mask_corpus(trigram_model_committee.corpus, 10, 10, original_ouput_path, masked_ouput_path)
    print("Corpus masked...")
    print("Original + Masked sentences written to file")

    print("----- Token prediciton section -----")
    predicted_sentences = []

    for original, masked in zip(original_sens, masked_sens):
        masked_tokens = masked.split()
        
        for i in range(len(masked_tokens)):
            token = masked_tokens[i]
            
            if token == "[*]":  # If the token is a masked token
                # generate the toekn using the plenary model
                generated_token = trigram_model_plenary.generate_next_token(" ".join(masked_tokens[:i]))
                
                # replace [*] with the predicted token
                masked_tokens[i] = generated_token[0]
        
        # Join the tokens back into a full sentence
        predicted_sentence = " ".join(masked_tokens)
        predicted_sentences.append(predicted_sentence)

    results_path = 'sampled_sents_results.txt'
    with open(results_path, 'w', encoding='utf-8') as f:
        for index, predicted_sentence in enumerate(predicted_sentences):
            # calculate probabilities
            commitee_prob = trigram_model_committee.calculate_prob_of_sentence(predicted_sentence)
            plenary_prob = trigram_model_plenary.calculate_prob_of_sentence(predicted_sentence)

            tokens = predicted_sentence.split()
            f.write(f'original_sentence: {original_sens[index]}\n')
            f.write(f'masked_sentence: {masked_sens[index]}\n')
            f.write(f'plenary_sentence: {predicted_sentence}\n')
            f.write(f'plenary_tokens: {tokens}\n')
            f.write(f'probability of plenary sentence in plenary corpus: {plenary_prob:.2f}\n')
            f.write(f'probability of plenary sentence in committee corpus: {commitee_prob:.2f}\n')
            f.write("\n")
    print("Prediction results written to file")

    print("----- Perplexity section -----")
    result_perplexity_path = "result_perplexity.txt"

    average_perplexity = compute_average_perplexity(masked_sens, original_sens, trigram_model_plenary)

    with open(result_perplexity_path, 'w', encoding='utf-8') as f:
        f.write(f"{average_perplexity:.2f}\n")

    print(f"average perplexity written to {result_perplexity_path}")

if __name__ == "__main__":
    main()