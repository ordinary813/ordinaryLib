from collections import defaultdict
import pandas as pd
import random
import os
import math
from math import log
import argparse
from collections import Counter


class Trigram_LM:

    def __init__(self, protocol_type, file_path):
        self.protocol_type = protocol_type
        self.corpus = None
        self.load_corpus(file_path)  # load the corpus
        self.unigrams = self.get_ngram_counts(1)
        self.bigrams = self.get_ngram_counts(2)
        self.trigrams = self.get_ngram_counts(3)

    def load_corpus(self, file_path):
        try:
            df = pd.read_json(file_path, lines=True)

            # Filter the DataFrame based on the protocol type
            self.corpus = df[df['protocol_type'] == self.protocol_type]

        except Exception as e:
            print(f"Error loading corpus: {e}")

    def get_ngram_counts(self, n):
        ngram_counter = Counter()

        # Loop over each sentence in the corpus
        for sentence in self.corpus['sentence_text']:
            ngrams = self.extract_ngrams_from_sentence(sentence, n)
            ngram_counter.update(ngrams)

        return ngram_counter

    def extract_ngrams_from_sentence(self, sentence, n):
        tokens = sentence.split()  # Tokenize the sentence
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]  # Generate n-grams

    def calculate_prob_of_sentence(self, sentence):
        tokens = sentence.split()
        tokens = ["<s_0>", "<s_1>"] + tokens  # Add start tokens
        log_prob = 0

        # weights for the interpolation
        lambda_1 = 0.7
        lambda_2 = 0.2
        lambda_3 = 0.1

        V = len(self.unigrams)  # number of unique words in the corpus
        unigram_frequencies = sum(self.unigrams.values())

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
            unigram_probability = unigram_count / (unigram_frequencies + V)

            # Apply linear interpolation
            prob = lambda_1 * trigram_probability + lambda_2 * bigram_probability + lambda_3 * unigram_probability

            # Log probability (avoid log(0))
            log_prob += (0 if prob == 0 else math.log(prob))

        return log_prob

    def generate_next_token(self, sentence):
        tokens = sentence.split()
        tokens = ["<s_0>", "<s_1>"] + tokens  # Add start tokens

        highest_log_prob = -float('inf')
        best_token = None

        # weights for the interpolation
        lambda_1 = 0.7
        lambda_2 = 0.2
        lambda_3 = 0.1

        V = len(self.unigrams)  # number of unique words in the corpus
        unigram_frequencies = sum(self.unigrams.values())

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
            unigram_probability = unigram_count / (unigram_frequencies + V)

            # Linear interpolation of trigram, bigram, and unigram probabilities
            prob = lambda_1 * trigram_probability + lambda_2 * bigram_probability + lambda_3 * unigram_probability

            # Update the best token
            if prob > 0:  # Avoid log of zero
                log_prob = math.log2(prob)
                if log_prob > highest_log_prob:
                    highest_log_prob = log_prob
                    best_token = token

        return best_token, highest_log_prob

    def generate_collocations_n(self, text, n):
        words = text.split()
        return [' '.join(words[i:i + n]) for i in range(len(words) - n + 1)]

    def compute_idf(self, protocol_collocations, total_docs):
        doc_counts = Counter()

        # for every collocation in all documents we increment when we saw the collocation
        # by the end of this loop each collocation should have its frequency in all documents
        for doc_collocations in protocol_collocations:
            for collocation in doc_collocations:
                doc_counts[collocation] += 1

        idf_scores = {collocation: log(total_docs / (doc_count)) for collocation, doc_count in doc_counts.items()}

        return idf_scores

    def compute_tf(self, doc_collocations):
        total_count = len(doc_collocations)
        collocation_counts = Counter(doc_collocations)

        # Compute TF scores
        tf_scores = {coll: count / total_count for coll, count in collocation_counts.items()}

        return tf_scores

    def compute_tfidf(self, tf_scores, idf_scores):
        tfidf_scores = {coll: tf * idf_scores.get(coll, 0) for coll, tf in tf_scores.items()}
        return tfidf_scores

    def get_k_n_t_collocations(self, k, n, t, type):
        corpus_df = self.corpus
        # produce all collocations of length n
        corpus_df['collocations'] = corpus_df['sentence_text'].apply(lambda x: self.generate_collocations_n(x, n))

        # place all collocations in a Counter
        collocation_counts = Counter()
        for coll_list in corpus_df['collocations']:
            collocation_counts.update(coll_list)

        if type == "frequency":
            # only include collocations that appear more than t
            filtered_collocations = {coll: count for coll, count in collocation_counts.items() if count >= t}
        elif type == "tfidf":

            total_docs = len(corpus_df['protocol_name'].unique())

            # group by protocol number
            grouped = corpus_df.groupby('protocol_name')['collocations']
            protocol_collocations = grouped.apply(lambda x: sum(x, []))

            idf_scores = self.compute_idf(protocol_collocations, total_docs)

            tfidf_scores = Counter()

            for protocol_name, collocations in grouped:
                doc_collocations = sum(collocations, [])

                tf_scores = self.compute_tf(doc_collocations)

                # Compute TF-IDF for collocations in the protocol
                tfidf = self.compute_tfidf(tf_scores, idf_scores)
                tfidf_scores.update(tfidf)

            filtered_collocations = {coll: tfidf_scores.get(coll, 0) for coll in collocation_counts.keys() if tfidf_scores.get(coll, 0) >= t}

        return filtered_collocations

def mask_tokens_in_sentences(sentences, x):
    masked_sentences = []

    for sentence in sentences:
        tokens = sentence.split()
        num_tokens_to_mask = max(1, int(len(tokens) * x / 100))  # Ensure at least 1 token is masked
        if num_tokens_to_mask > 0:
            tokens_to_mask = random.sample(range(len(tokens)), num_tokens_to_mask) # choose the sentences to mask

            # change the tokens to *
            for index in tokens_to_mask:
                tokens[index] = '[*]'

        masked_sentences.append(' '.join(tokens))  # create the sentence again

    return masked_sentences


def get_random_sentences(corpus_df, num_sentences=10, min_tokens=5):
    # Filter sentences that have at least `min_tokens` tokens
    filtered_sentences = corpus_df[corpus_df['sentence_text'].str.split().str.len() >= min_tokens]

    # Convert the filtered DataFrame column to a list
    filtered_sentences_list = filtered_sentences['sentence_text'].tolist()

    # Randomly select `num_sentences` from the valid sentences
    return random.sample(filtered_sentences_list, num_sentences)


def write_ngram_collocations(trigram_model_committee, trigram_model_plenary, n, output_file):
    # Get n-gram counts using the appropriate method (get_ngram_counts)
    top_collocations_freq_committee = sorted(trigram_model_committee.get_ngram_counts(n).items(), key=lambda x: x[1],
                                             reverse=True)[:10]
    top_collocations_freq_plenary = sorted(trigram_model_plenary.get_ngram_counts(n).items(), key=lambda x: x[1],
                                           reverse=True)[:10]

    # TF-IDF calculation (assuming it's done somewhere in your code)
    top_collocations_tfidf_committee = sorted(trigram_model_committee.get_ngram_counts(n).items(), key=lambda x: x[1],
                                              reverse=True)[:10]  # Placeholder
    top_collocations_tfidf_plenary = sorted(trigram_model_plenary.get_ngram_counts(n).items(), key=lambda x: x[1],
                                            reverse=True)[:10]  # Placeholder

    # Open the output file and write the collocations
    with open(output_file, 'a', encoding='utf-8') as f:
        if n == 2:
            f.write("Two-gram collocations:\n")
        elif n == 3:
            f.write("Three-gram collocations:\n")
        elif n == 4:
            f.write("Four-gram collocations:\n")

        f.write("Frequency:\n")

        # Write "Committee corpus" section for Frequency
        f.write("Committee corpus:\n")
        for ngram, count in top_collocations_freq_committee:
            ngram_str = ' '.join(ngram)
            f.write(f"{ngram_str}\n")
        f.write("\n")

        # Write "Plenary corpus" section for Frequency
        f.write("Plenary corpus:\n")
        for ngram, count in top_collocations_freq_plenary:
            ngram_str = ' '.join(ngram)
            f.write(f"{ngram_str}\n")
        f.write("\n")

        f.write("TF-IDF:\n")

        # Write "Committee corpus" section for TF-IDF
        f.write("Committee corpus:\n")
        for ngram, count in top_collocations_tfidf_committee:
            ngram_str = ' '.join(ngram)
            f.write(f"{ngram_str}\n")
        f.write("\n")

        # Write "Plenary corpus" section for TF-IDF
        f.write("Plenary corpus:\n")
        for ngram, count in top_collocations_tfidf_plenary:
            ngram_str = ' '.join(ngram)
            f.write(f"{ngram_str}\n")
        f.write("\n")


def calculate_perplexity(masked_sentences, trigram_model_plenary):

    sentence_perplexities = []

    for sentence in masked_sentences:
        # positions of the masked tokens
        masked_token_positions = [i for i, token in enumerate(sentence.split()) if token == "[*]"]

        log_prob_sum = 0
        masked_token_count = len(masked_token_positions)

        # For each masked token in the sentence, predict and calculate its log probability
        for idx in masked_token_positions:
            # Extract the context for the trigram (previous 2 tokens before the masked token)
            context = sentence.split()[idx - 2:idx]
            # Predict the best token and its log probability
            best_token, highest_log_prob = trigram_model_plenary.generate_next_token(" ".join(context))

            # Add the log probability of the predicted token
            log_prob_sum += highest_log_prob

        # Calculate the perplexity for the current sentence
        if masked_token_count > 0:
            sentence_perplexity = math.exp(-log_prob_sum / masked_token_count)
            sentence_perplexities.append(sentence_perplexity)

    # Calculate the average perplexity across all sentences
    average_perplexity = sum(sentence_perplexities) / len(sentence_perplexities) if sentence_perplexities else 0

    return average_perplexity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_corpus",
        type=str,
        help="Path to the corpus jsonl file."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the directory for the output files."
    )

    args = parser.parse_args()

    corpus_path = args.path_to_corpus
    out_dir = args.output_dir

    # Create output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)

    # Create an instance of Trigram_LM for both committee and plenary corpora
    trigram_model_committee = Trigram_LM("committee", corpus_path)
    trigram_model_plenary = Trigram_LM("plenary", corpus_path)

    # Output file for collocations
    output_file = r"knesset_collocations.txt"
    output_file = os.path.join(out_dir, output_file)

    # Clear the file before writing
    with open(output_file, 'w', encoding='utf-8') as f:
        pass

    for n in [2, 3, 4]:
        write_ngram_collocations(trigram_model_committee, trigram_model_plenary, n, output_file)

    # --------part 3----------

    # Get 10 random sentences with at least 5 tokens
    selected_sentences = get_random_sentences(trigram_model_committee.corpus, num_sentences=10, min_tokens=5)

    # Mask 10% of tokens in the selected sentences
    masked_sentences = mask_tokens_in_sentences(selected_sentences, 10)

    # Output the original sentences
    with open(os.path.join(out_dir, r"original_sampled_sents.txt"), 'w', encoding='utf-8') as f:
        for sentence in selected_sentences:
            f.write(sentence + '\n')

    # Output the masked sentences
    with open(os.path.join(out_dir, r"masked_sampled_sents.txt"), 'w', encoding='utf-8') as f:
        for sentence in masked_sentences:
            f.write(sentence + '\n')

    # Open the sampled_sents_results
    with open(os.path.join(out_dir, r"sampled_sents_results.txt"), 'w', encoding='utf-8') as result_file:
        for sentence in masked_sentences:
            # Split the sentence by the masked token
            parts = sentence.split("[*]")

            # Initialize updated sentence
            updated_sentence = parts[0]

            # List of predicted tokens for each mask
            predicted_tokens = []

            # Iterate through all the parts after splitting by [*]
            for i in range(1, len(parts)):
                # Predict the token for the current mask using the updated context
                best_token, highest_log_prob = trigram_model_plenary.generate_next_token(updated_sentence)

                # Add the predicted token to the updated sentence

                updated_sentence += best_token[0] + parts[i]
                # Store the predicted token
                predicted_tokens.append(best_token[0])

            # Calculate the probability of the sentence in both plenary and committee models
            plenary_prob = trigram_model_plenary.calculate_prob_of_sentence(updated_sentence)
            committee_prob = trigram_model_committee.calculate_prob_of_sentence(updated_sentence)

            # Write the results to the file
            result_file.write(f"original_sentence: {selected_sentences[masked_sentences.index(sentence)]}\n")
            result_file.write(f"masked_sentence: {sentence}\n")
            result_file.write(f"plenary_sentence: {updated_sentence}\n")
            result_file.write(f"plenary_tokens: {', '.join(predicted_tokens)}\n")
            result_file.write(f"probability of plenary sentence in plenary corpus: {plenary_prob:.2f}\n")
            result_file.write(f"probability of plenary sentence in committee corpus: {committee_prob:.2f}\n")
            result_file.write("\n")

    # Open the file to write perplexity results
    with open(os.path.join(out_dir, r"perplexity_result.txt"), 'w', encoding='utf-8') as result_file:

        average_perplexity = calculate_perplexity(masked_sentences, trigram_model_plenary)
        # Write the average perplexity to the result file
        result_file.write(f"{average_perplexity:.2f}")