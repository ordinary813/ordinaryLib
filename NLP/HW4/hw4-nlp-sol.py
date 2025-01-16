# %%
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import json
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
import re
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# %%
corpus_file = 'knesset_corpus.jsonl'
punctuations = '",./<>?;:\'[]{}\\|`~!@#$%^&*()-_=+'
tokenized_sentences = []

# Load the corpus into a list of tokenized sentences (list of lists)
with open(corpus_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        sentence_text = data.get('sentence_text', '')
        if sentence_text:
            tokenized_sentence = [
                ''.join(char for char in word if char not in punctuations)      # Remove punctuation
                for word in sentence_text.split() if not word.isdigit()         # Split by spaces + token is not a number
            ]
            tokenized_sentences.append([word for word in tokenized_sentence if word])

# %%
# Fit a model using the sentences we tokenized and save it
model = Word2Vec(sentences=tokenized_sentences, vector_size=75, window=5, min_count=1, workers=4)
model.save("knesset_word2vec.model")

# %%
word_vectors = model.wv
print(word_vectors['ישראל'])

# %% [markdown]
# 1. Increasing the vector size allows us to capture more information about each word as it has more dimensionality.
# 2. PROBLEMS IN THIS SPECIFIC CORPUS

# %%
words_to_check = ['ישראל', 'גברת', 'ממשלה', 'חבר', 'בוקר', 'מים', 'אסור', 'רשות', 'זכויות']

# We decided to utilize a dictionary to store the most similar words for each word
similar_words = {}

# Find the most similar words for each word
for word in words_to_check:
    if word in word_vectors:
        # Calculate similarity between the target word and all other words in the vocabulary
        similarity_scores = {other_word: word_vectors.similarity(word, other_word)
                             for other_word in word_vectors.index_to_key if other_word != word}
        
        # Sort by the scores and pick the top 5
        most_similar = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        similar_words[word] = most_similar

with open('knesset_similar_words.txt', 'w', encoding='utf-8') as f:
    for word, similar in similar_words.items():
        # Format the top 5 for the current word
        similar_str = ', '.join([f"({sim_word}, {score:.4f})" for sim_word, score in similar])
        f.write(f"{word}: {similar_str}\n")

# %%
def average_sentence_embedding(sentence):
    # Get the word embedding vectors for the sentence
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

sentence_embeddings = []
for sentence in tokenized_sentences:
    sentence_embeddings.append(average_sentence_embedding(sentence))

# %%
# The sentences we hand picked for this task, they seem to have a good structure
# and it is possible to infer their meanings, they were also not too long and not too short
selected_indices = [18,     22,     94,     3142,   3220, 
                    9277,   62100,  79100,  93123,  101293]

# Calculate cosine similarity between sentence embeddings
similar_sentences = {}
for idx in selected_indices:
    current_embedding = sentence_embeddings[idx].reshape(1, -1)
    similarities = cosine_similarity(current_embedding, sentence_embeddings)
    most_similar_idx = similarities.argsort()[0][-2]
    similar_sentences[idx] = most_similar_idx

# Load the original sentences with punctuations
original_sentences = []
with open(corpus_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        original_sentences.append(data.get('sentence_text', ''))

with open('knesset_similar_sentences.txt', 'w', encoding='utf-8') as f:
    for idx, similar_idx in similar_sentences.items():
        current_sentence = original_sentences[idx]
        similar_sentence = original_sentences[similar_idx]
        f.write(f"{current_sentence}: most similar sentence: {similar_sentence}\n")

# %%
sentences_to_check = {
    r'בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים .': ['דקות','הדיון'],
    r'בתור יושבת ראש הוועדה , אני מוכנה להאריך את ההסכם באותם תנאים .': ['הוועדה','אני', 'ההסכם'],
    r'בוקר טוב , אני פותח את הישיבה .': ['בוקר', 'פותח'],
    r'שלום , אנחנו שמחים להודיע שחברינו ה יקר קיבל קידום .': ['שלום', 'שמחים', 'היקר','קידום'],
    r'אין מניעה להמשיך לעסוק ב נושא .': ['מניעה']
}

def replace_with_similar_words(sentence_dict, model):
    replaced_sentences = []
    for sentence, words in sentence_dict.items():
        new_sentence = sentence
        for word in words:
            if word in model.wv:
                similar_word = model.wv.most_similar(word, topn=1)[0][0]
                new_sentence = new_sentence.replace(word, similar_word)
        replaced_sentences.append(new_sentence)
    return replaced_sentences

    # Replace words in the sentences
replaced_sentences_with_prompts = replace_with_similar_words(sentences_to_check, model)
for sentence in replaced_sentences_with_prompts:
    print(sentence)

# %% [markdown]
# ### KNN Classifier

# %%
class Speaker:
    def __init__(self, file_path, speaker_name=None):
        self.name = speaker_name
        self.df = pd.read_json(file_path, lines=True)
        if speaker_name:
            self.df = self.df[self.df['speaker_name'].apply(self._matches_speaker_name)]
    
    def _matches_speaker_name(self, name_in_data):
        if not self.name:
            return False
        
        name_parts = self.name.split()
        data_parts = name_in_data.split()
        
        # Handle case where name has more than 4 components
        if len(name_parts) > 4 or len(data_parts) > 4:
            return False
        
        # Iterate over all parts of the provided name
        for i, part in enumerate(name_parts):
            if i >= len(data_parts):  # Dataset name has fewer parts
                return False
            
            # If it's an initial, match with any name starting with the same letter
            if re.fullmatch(rf"{re.escape(part[0])}['\"׳`]?", part):
                if not data_parts[i].startswith(part[0]):
                    return False
            # If it's a full name, ensure it matches fully
            elif part != data_parts[i]:
                return False
        return True

corpus_path = 'knesset_corpus.jsonl'
df = pd.read_json(corpus_path, lines=True)

def get_most_frequent_speakers(df):
    speaker_counts = df['speaker_name'].value_counts()
    most_frequent_speaker = speaker_counts.idxmax()
    second_most_frequent_speaker = speaker_counts.index[1] if len(speaker_counts) > 1 else None
    return most_frequent_speaker, second_most_frequent_speaker

most_frequent_speaker, second_most_frequent_speaker = get_most_frequent_speakers(df)

# Binary classification task
speaker1Bin = Speaker(corpus_path, most_frequent_speaker)
speaker2Bin = Speaker(corpus_path, second_most_frequent_speaker)

def balance_dataframes(df1, df2):
    min_len = min(len(df1), len(df2))
    return df1.sample(min_len), df2.sample(min_len)

speaker1Bin.df, speaker2Bin.df = balance_dataframes(speaker1Bin.df, speaker2Bin.df)

# %%
def average_sentence_embedding(sentence):
    # Get the word embedding vectors for the sentence
    vectors = [model.wv[word] for word in sentence if word in model.wv]
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

tokenized_sentences = []
labels = []

for bin_data in [speaker1Bin.df, speaker2Bin.df]:
    for _, row in bin_data.iterrows():
        sentence_text = row['sentence_text']
        speaker_name = row['speaker_name']
        if sentence_text:
            tokenized_sentence = [
                ''.join(char for char in word if char not in punctuations)  # Remove punctuation
                for word in sentence_text.split() if not word.isdigit()     # Split by spaces + token is not a number
            ]
            tokenized_sentences.append([word for word in tokenized_sentence if word])
            labels.append(1 if speaker_name == most_frequent_speaker else 0)

sentence_embeddings = np.array([average_sentence_embedding(sentence) for sentence in tokenized_sentences])
labels = np.array(labels)

# Train the KNN classifier using cross-validation
classifier = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(classifier, sentence_embeddings, labels, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy scores: {scores}")
print(f"Mean accuracy: {scores.mean() * 100:.2f}%")

# %%
print(classification_report(labels, classifier.fit(sentence_embeddings, labels).predict(sentence_embeddings)))

# %% [markdown]
# ### Bert

# %%
original_sentences = []
with open('original_sampled_sents.txt', 'r', encoding='utf-8') as f:
    for line in f:
        original_sentences.append(line)

masked_sentences = []
with open('masked_sampled_sents.txt', 'r', encoding='utf-8') as f:
    for line in f:
        masked_sentences.append(line.replace('*', 'MASK').strip())

tokenized_masked_sentences = [sentence.split() for sentence in masked_sentences]

# %%
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert')
model = AutoModelForMaskedLM.from_pretrained('dicta-il/dictabert')
model.eval()

with open('dictabert_results.txt', 'w') as file:
    # Iterate over masked sentences and corresponding original sentences
    for original_sentence, masked_sentence in zip(original_sentences, masked_sentences):
        tokenized_sentence = masked_sentence.split()
        generated_tokens = []  # List to store generated tokens
        for i, token in enumerate(tokenized_sentence):
            if token == '[MASK]':
                inputs = tokenizer.encode(' '.join(tokenized_sentence), return_tensors='pt')
                outputs = model(inputs)
                predictions = outputs.logits[0, i, :]
                most_similar = torch.topk(predictions, 1)[1]
                generated_token = tokenizer.convert_ids_to_tokens(most_similar.item())
                tokenized_sentence[i] = generated_token
                generated_tokens.append(generated_token)
        
        dictaBERT_sentence = ' '.join(tokenized_sentence)
        dictaBERT_tokens = ', '.join(generated_tokens)
        
        # Write combined output to the file
        file.write(f"original_sentence: {original_sentence.replace('\n','')}\n")
        file.write(f"masked_sentence: {masked_sentence}\n")
        file.write(f"dictaBERT_sentence: {dictaBERT_sentence}\n")
        file.write(f"dictaBERT tokens: {dictaBERT_tokens}\n\n")