import os, torch, argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    "imdb_dir",
    type=str,
    help="Path to the directory of the imdb dataset."
)

parser.add_argument(
    "output_file_path",
    type=str,
    help="Path to the output txt file."
)

args = parser.parse_args()

imdb_dir = args.imdb_dir
output_file_path = args.output_file_path

# Initialize GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load the IMDB dataset ###
dataset_path = imdb_dir
# If path exists - load the dataset from disk, otherwise download it
if os.path.exists(dataset_path):
    subset = load_from_disk(dataset_path)
else:
    dataset = load_dataset('imdb')
    subset = dataset['train'].shuffle(seed=42).select(range(500))
    subset.save_to_disk(dataset_path)

dataset = subset
dataset = dataset.rename_column("label", "labels")

# Convert the dataset to a DataFrame to keep track of the labels (positive/negative)
dataset_df = pd.DataFrame(dataset)

# Create two subsets: one for positive reviews and one for negative reviews
positive_reviews = dataset_df[dataset_df['labels'] == 1]
negative_reviews = dataset_df[dataset_df['labels'] == 0]

# Perform stratified sampling by randomly selecting 25 positive reviews and 25 negative reviews
sampled_positive_reviews = positive_reviews.sample(n=25, random_state=42)
sampled_negative_reviews = negative_reviews.sample(n=25, random_state=42)

# Combine the two subsets into one dataframe (Stratified sampling)
sampled_reviews = pd.concat([sampled_positive_reviews, sampled_negative_reviews])

# Shuffle the combined dataset to mix the positive and negative reviews
sampled_reviews = sampled_reviews.sample(frac=1, random_state=42).reset_index(drop=True)

# Load the FLAN-T5 model and tokenizer
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
flan_model = flan_model.to(device)


# Zero-shot prompting function
def zero_shot_prompt(review_text):
    prompt = f"Classify the following review as positive or negative: \"{review_text}\""
    return prompt


few_shot_examples = """
Classify the following review as positive or negative, use the examples as reference

Example 1:
Review: "I was captivated by the film from start to finish. It was absolutely wonderful!"
Sentiment: positive

Example 2:
Review: "The film dragged on and lacked any real emotional depth. It was a disappointment."
Sentiment: negative

"""


def few_shot_prompt(review_text):
    prompt = few_shot_examples + f"\n\nReview: \"{review_text}\"\nSentiment:"
    return prompt


# Instruction-based prompting function
def instruction_prompt(review_text):
    prompt = (
        "Classify the overall sentiment of the following review as either 'positive' or 'negative'.\n"
        "Focus on the general tone of the review. If the review contains both positive and negative aspects, "
        "classify it based on the dominant sentiment expressed. If unclear, lean towards the overall feeling of the review.\n\n"
        f"Review: \"{review_text}\""
    )
    return prompt


# Generate model response function with truncation
def generate_response(prompt, model, tokenizer, max_length=20, max_input_length=512):
    # Tokenize the input and truncate if it exceeds the max input length
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_length).to(model.device)
    output = model.generate(**inputs, max_length=max_length, num_beams=5)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Initialize counters for accuracy
zero_shot_correct = 0
few_shot_correct = 0
instruction_based_correct = 0
total_reviews = len(sampled_reviews)

# Output file path
output_file = output_file_path

# Iterate through the sampled reviews and classify
with open(output_file, 'w') as f:
    for i, review in enumerate(sampled_reviews.itertuples()):
        review_text = review.text
        true_label = review.labels  # 0=negative, 1=positive
        true_label_text = "positive" if true_label == 1 else "negative"

        # Truncate review text to fit within model's token limit (if necessary)
        review_text_truncated = review_text[:1000]  # Optional truncation to limit review length

        # Zero-shot
        zero_shot = generate_response(zero_shot_prompt(review_text_truncated), flan_model,
                                      flan_tokenizer).strip().lower()

        # Few-shot
        few_shot = generate_response(few_shot_prompt(review_text_truncated), flan_model, flan_tokenizer).strip().lower()

        # Instruction-based
        instruction_based = generate_response(instruction_prompt(review_text_truncated), flan_model,
                                              flan_tokenizer).strip().lower()

        # Check if the model's prediction matches the true label
        zero_shot_correct += (zero_shot == true_label_text)
        few_shot_correct += (few_shot == true_label_text)
        instruction_based_correct += (instruction_based == true_label_text)

        # Writing results to file
        f.write(f"Review {i + 1}: {review_text}\n")
        f.write(f"Review {i + 1} true label: {true_label_text}\n")
        f.write(f"Review {i + 1} zero-shot: {zero_shot}\n")
        f.write(f"Review {i + 1} few-shot: {few_shot}\n")
        f.write(f"Review {i + 1} instruction-based: {instruction_based}\n\n")

# Calculate accuracy for each prompting strategy
zero_shot_accuracy = zero_shot_correct / total_reviews * 100
few_shot_accuracy = few_shot_correct / total_reviews * 100
instruction_based_accuracy = instruction_based_correct / total_reviews * 100

# # Print the accuracy
# print(f"Zero-shot accuracy: {zero_shot_accuracy:.2f}%")
# print(f"Few-shot accuracy: {few_shot_accuracy:.2f}%")
# print(f"Instruction-based accuracy: {instruction_based_accuracy:.2f}%")
