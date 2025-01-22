import os, torch, random, argparse
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

### Utilize GPU ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load the IMDB dataset ###
dataset_path = imdb_dir
# if the path exists - load the dataset from disk, otherwise download it
if os.path.exists(dataset_path):
    subset = load_from_disk(dataset_path)
else:
    dataset = load_dataset('imdb')
    subset = dataset['train'].shuffle(seed=42).select(range(500))
    subset.save_to_disk(dataset_path)

dataset = subset
dataset = dataset.rename_column("label", "labels")

flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
flan_model = flan_model.to(device)

sampled_reviews = random.sample(list(dataset), 50)

def zero_shot_prompt(review_text):
    prompt = f"Classify the following review as positive or negative: \"{review_text}\""
    return prompt

few_shot_examples = """
Review 1: "The movie was fantastic and full of unexpected turns." -> positive
Review 2: "The movie was boring and I didn't enjoy it." -> negative
"""

def few_shot_prompt(review_text):
    prompt = few_shot_examples + f"\nReview: \"{review_text}\" ->"
    return prompt

def instruction_prompt(review_text):
    prompt = f"Analyze the sentiment of the following review: \"{review_text}\". Respond with 'positive' or 'negative'."
    return prompt

def generate_response(prompt, model, tokenizer, max_length=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_length=max_length, num_beams=5)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

output_file = output_file_path

with open(output_file, 'w') as f:
    for i, review in enumerate(sampled_reviews):
        review_text = review['text']
        true_label = review['labels']  # Assuming label is in dataset, 0=negative, 1=positive
        true_label_text = "positive" if true_label == 1 else "negative"
        
        # Zero-shot
        zero_shot = generate_response(zero_shot_prompt(review_text), flan_model, flan_tokenizer)
        
        # Few-shot
        few_shot = generate_response(few_shot_prompt(review_text), flan_model, flan_tokenizer)
        
        # Instruction-based
        instruction_based = generate_response(instruction_prompt(review_text), flan_model, flan_tokenizer)
        
        # Writing to file
        f.write(f"Review {i+1}: {review_text}\n")
        f.write(f"Review {i+1} true label: {true_label_text}\n")
        f.write(f"Review {i+1} zero-shot: {zero_shot}\n")
        f.write(f"Review {i+1} few-shot: {few_shot}\n")
        f.write(f"Review {i+1} instruction-based: {instruction_based}\n\n")
