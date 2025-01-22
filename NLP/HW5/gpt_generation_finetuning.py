import os, torch
from datasets import load_dataset, load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

### Utilize GPU ##
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load the IMDB dataset ###
dataset_path = 'imdb_subset'
# if the path exists - load the dataset from disk, otherwise download it
if os.path.exists(dataset_path):
    subset = load_from_disk(dataset_path)
else:
    dataset = load_dataset('imdb')
    subset = dataset['train'].shuffle(seed=42).select(range(500))
    subset.save_to_disk('imdb_subset')

dataset = subset
dataset = dataset.rename_column("label", "labels")


# Select 100 reviews each (more than that and we ran out of VRAM)
positive_reviews = dataset.filter(lambda example: example["labels"] == 1).select(range(100))
negative_reviews = dataset.filter(lambda example: example["labels"] == 0).select(range(100))

def tokenize_reviews(dataset, tokenizer, max_length=150):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

### Model initialization ###
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

### Reviews tokenization ###
tokenized_positive_reviews = tokenize_reviews(positive_reviews, gpt2_tokenizer)
tokenized_negative_reviews = tokenize_reviews(negative_reviews, gpt2_tokenizer)

def train_save_GPT2(model, tokenizer, tokenized_dataset, prompt='The movie was'):
    # Check which dataset we are training on
    if tokenized_dataset["labels"][0] == 0:
        sentiment = "negative"
    elif tokenized_dataset["labels"][0] == 1:
        sentiment = "positive"

    training_args = TrainingArguments(
        output_dir=f"./gpt2-{sentiment}-results",
        num_train_epochs=5,
        learning_rate=2e-5,
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,   # batch size for evaluation
        weight_decay=0.01,              # strength of weight decay
        logging_dir='./logs',      # directory for storing logs
        logging_steps=10,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Train the model
    trainer.train()
    results = trainer.evaluate()
    print(f'Perplexity: {results["eval_loss"]}')

    # Save the model and tokenizer
    save_directory = f"./gpt2-{sentiment}-results"
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)

### Training each model ###
train_save_GPT2(gpt2_model, gpt2_tokenizer, tokenized_positive_reviews)
train_save_GPT2(gpt2_model, gpt2_tokenizer, tokenized_negative_reviews)

positive_model = GPT2LMHeadModel.from_pretrained("gpt2-positive-results")
positive_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-positive-results")

negative_model = GPT2LMHeadModel.from_pretrained("gpt2-negative-results")
negative_tokenizer = GPT2Tokenizer.from_pretrained("gpt2-negative-results")

# Parameters for text generation
max_length = 150
temperature = 0.7
top_k = 50
top_p = 0.9
repetition_penalty = 1.2
prompt = 'The movie was'

# Attention mask for each model
input_ids_pos = positive_tokenizer.encode(prompt, return_tensors="pt")
attention_mask = input_ids_pos.ne(positive_tokenizer.pad_token_id) 

input_ids_neg = negative_tokenizer.encode(prompt, return_tensors="pt")
attention_mask = input_ids_neg.ne(negative_tokenizer.pad_token_id)

# Review generator function
def generate_reviews(model, tokenizer, num_reviews=5):
    reviews = []
    for _ in range(num_reviews):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                num_return_sequences=1
            )
        review = tokenizer.decode(output[0], skip_special_tokens=True)
        reviews.append(review)
    return reviews

positive_reviews = generate_reviews(positive_model, positive_tokenizer)
negative_reviews = generate_reviews(negative_model, negative_tokenizer)

with open('generated_reviews.txt', 'w') as file:
    file.write("Reviews generated by positive model:\n")
    for i, review in enumerate(positive_reviews, start=1):
        file.write(f"{i}. {review}\n")
    file.write("\nReviews generated by negative model:\n")
    for i, review in enumerate(negative_reviews, start=1):
        file.write(f"{i}. {review}\n")