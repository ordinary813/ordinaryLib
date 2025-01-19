from gensim.models import Word2Vec
import json
import re, argparse, os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_to_corpus",
        type=str,
        help="Path to the corpus file."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the directory for the output files."
    )

    args = parser.parse_args()

    corpus_file = args.path_to_corpus
    out_dir = args.output_dir

    # Create output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)
    # -------- part 1-------------#
    tokenized_sentences = []

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentence = data.get("sentence_text", "")

            # Tokenize the sentence (remove numbers and punctuation, but keep words with quotation marks like "חו"ל")
            tokens = re.findall(r'["\u0590-\u05FF]+|"\u0590-\u05FF+"', sentence)

            # Filter out tokens with less than 2 characters
            clean_tokens = [token for token in tokens if len(token) > 1]

            # Add the tokens to the list
            if clean_tokens:
                tokenized_sentences.append(clean_tokens)

    # create word2vec model
    model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=5, min_count=1)

    # save model for further use
    model.save("knesset_word2vec.model")

    # -------- part 2-------------#

    # Load the saved Word2Vec model
    model = Word2Vec.load("knesset_word2vec.model")
    word_vectors = model.wv

    # List of words to analyze
    words_list = [
        "ישראל",
        "גברת",
        "ממשלה",
        "חבר",
        "בוקר",
        "מים",
        "אסור",
        "רשות",
        "זכויות"
    ]

    vocabulary = list(model.wv.key_to_index.keys())

    # write for each word in list the 5 closest words from corpus
    with open(os.path.join(out_dir, "knesset_similar_words.txt"), "w", encoding="utf-8") as f:
        for word in words_list:
            try:
                # Calculate similarity scores with all other words
                similarity_scores = []
                for other_word in vocabulary:
                    if other_word != word:  # Exclude the word itself
                        similarity_score = model.wv.similarity(word, other_word)
                        similarity_scores.append((other_word, similarity_score))

                # Sort by similarity score in descending order and take the top 5
                top_5_similar = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:5]

                result_line = f"{word}: "
                result_line += ", ".join([f"({sim_word}, {sim_score:.4f})" for sim_word, sim_score in top_5_similar])
                result_line += "\n"

                # Write to the file
                f.write(result_line)
            except KeyError:
                pass

    # -------- Check antonyms distances for question 2------------#

    # List of antonym pairs to check
    antonym_pairs = [("אהבה", "שנאה"), ("קל", "כבד"),("שקט", "רעש")]

    def cosine_distance(word1, word2, model):
        try:
            vec1 = model.wv[word1]
            vec2 = model.wv[word2]
            # Compute cosine similarity, and return the distance (1 - similarity)
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            distance = 1 - similarity
            return distance
        except KeyError:
            return None

    # #print antonym distances
    # for word1, word2 in antonym_pairs:
    #     distance = cosine_distance(word1, word2, model)
    #     if distance is not None:
    #         print(f"Distance between {word1} and {word2}: {distance:.4f}\n")
    #     else:
    #         print(f"One or both words '{word1}' and '{word2}' not in vocabulary\n")

    # -------------------------------------------------- #

    sentence_embeddings = []
    for sentence in tokenized_sentences:
        # Compute the sentence embedding as the mean of word vectors in the sentence
        if sentence:
            word_vectors = [model.wv[token] for token in sentence]
            sentence_embedding = np.mean(word_vectors, axis=0)
        else:
            sentence_embedding = np.zeros(model.vector_size)

        # Append the result to the list
        sentence_embeddings.append(sentence_embedding)

    # ----------------------------------------------------------- #
    selected_indices = [18, 22, 94, 3142, 3220,
                        9277, 62100, 79100, 93123, 101293]

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

    with open(os.path.join(out_dir, 'knesset_similar_sentences.txt'), 'w', encoding='utf-8') as f:
        for idx, similar_idx in similar_sentences.items():
            current_sentence = original_sentences[idx]
            similar_sentence = original_sentences[similar_idx]
            f.write(f"{current_sentence}: most similar sentence: {similar_sentence}\n")


    # Sentences with red-marked words
    marked_sentences = [
        "בעוד מספר דקות נתחיל את הדיון בנושא השבת החטופים.",
        "בתור יושבת ראש הוועדה , אני מוכנה להאריך את ההסכם באותם תנאים .",
        "בוקר טוב , אני פותח את הישיבה .",
        "שלום , אנחנו שמחים להודיע שחברינו היקר קיבל קידום .",
        "אין מניעה להמשיך לעסוק בנושא ."
    ]

    # Red words to replace
    red_words_by_sentence = {
        1: ["דקות", "הדיון"],
        2: ["הוועדה", "אני", "ההסכם"],
        3: ["בוקר", "פותח"],
        4: ["שלום", "שמחים", "היקר", "קידום"],
        5: ["מניעה"]
    }
    # Process the sentences
    replaced_sentences = []

    for i, sentence in enumerate(marked_sentences):
        sentence_number = i + 1
        red_words = red_words_by_sentence[sentence_number]

        original_sentence = sentence
        modified_sentence = sentence
        replaced_words = []  # List to store replaced words for this sentence

        # Replace all red words in the sentence
        for word_to_replace in red_words:
            try:
                # Find similar words using most_similar with positive and negative adjustments
                match word_to_replace:
                    case 'דקות':
                        similar_words = model.wv.most_similar(positive=[word_to_replace, "שניות"], topn=3)
                        new_word = similar_words[1][0]
                    case 'הדיון':
                        similar_words = model.wv.most_similar(word_to_replace, topn=3)
                        new_word = similar_words[1][0]
                    case 'הוועדה':
                        similar_words = model.wv.most_similar(positive=[word_to_replace, "הועדה"], topn=3)
                        new_word = similar_words[2][0]
                    case 'אני':
                        similar_words = model.wv.most_similar(word_to_replace, topn=3)
                        new_word = similar_words[0][0]
                    case 'ההסכם':
                        similar_words = model.wv.most_similar(positive=[word_to_replace, "הסכם", "הסכמה"], topn=3)
                        new_word = similar_words[2][0]
                    case 'בוקר':
                        similar_words = model.wv.most_similar(word_to_replace, topn=3)
                        new_word = similar_words[0][0]
                    case 'פותח':
                        similar_words = model.wv.most_similar(word_to_replace, topn=3)
                        new_word = similar_words[0][0]
                    case 'שלום':
                        similar_words = model.wv.most_similar(positive=[word_to_replace, "היי", "ברכות"], topn=3)
                        new_word = similar_words[0][0]
                    case 'שמחים':
                        similar_words = model.wv.most_similar(positive=[word_to_replace, "שמח","אנחנו"], topn=3)
                        new_word = similar_words[1][0]
                    case 'היקר':
                        similar_words = model.wv.most_similar(word_to_replace, topn=3)
                        new_word = similar_words[0][0]
                    case 'קידום':
                        similar_words = model.wv.most_similar(word_to_replace, topn=3)
                        new_word = similar_words[0][0]
                    case 'מניעה':
                        similar_words = model.wv.most_similar(positive=[word_to_replace, "הפרעה", "איסור", "הגבלה"], topn=3)
                        new_word = similar_words[1][0]
                    case _:
                        similar_words = model.wv.most_similar(word_to_replace, topn=3)
                        new_word = similar_words[0][0]

                # Replace the word in the sentence
                modified_sentence = modified_sentence.replace(word_to_replace, new_word)
                replaced_words.append(f"({word_to_replace} : {new_word})")  # Store the replacement for later

            except KeyError:
                pass

        # Append the modified sentence only once, after all replacements
        replaced_sentences.append(f"{sentence_number}: {original_sentence}: {modified_sentence}\n")
        replaced_sentences.append(f"replaced words: {', '.join(replaced_words)}\n")

    # Write the output to a file
    with open(os.path.join(out_dir, "red_words_sentences.txt"), "w", encoding="utf-8") as f:
        f.writelines(replaced_sentences)
