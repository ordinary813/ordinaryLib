import os, torch
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset, load_from_disk, disable_progress_bar
from transformers import Trainer, BertTokenizer, BertForSequenceClassification, TrainingArguments, logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
logging.set_verbosity_error()
disable_progress_bar()

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

### Bert model and tokenizer initialization ###
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

# Apply the tokenization function to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Train-test split
split_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
tokenized_train = split_dataset["train"]
tokenized_test = split_dataset["test"]

# Train-eval split
split_dataset = tokenized_train.train_test_split(test_size=0.4, seed=42)
tokenized_train = split_dataset["train"]
tokenized_eval = split_dataset["test"]


### Training configuration ###
training_args = TrainingArguments(
    output_dir="./results",               # Directory to save model checkpoints
    evaluation_strategy="epoch",          # Evaluate at the end of each epoch
    learning_rate=2e-5,                   # Learning rate
    per_device_train_batch_size=8,        # Batch size for training
    per_device_eval_batch_size=8,         # Batch size for evaluation
    num_train_epochs=10,                  # Number of epochs
    weight_decay=0.01,                    # Weight decay
    logging_dir="./logs",                 # Directory to save logs
    logging_steps=10000,                  # Log every 10000 steps
    save_total_limit=2,                   # Limit the number of saved checkpoints
    log_level = 'error',
    disable_tqdm=True,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)  # Get the index of the max logit
    return {"accuracy": accuracy_score(labels, preds)}

trainer = Trainer(
    model=model,                          # The pre-trained model to fine-tune
    args=training_args,                   # Training arguments
    train_dataset=tokenized_train,        # Training dataset
    eval_dataset=tokenized_eval,          # Evaluation dataset
    tokenizer=tokenizer,                  # Tokenizer for data preprocessing
    compute_metrics=compute_metrics       # Custom metrics function
)

### Training + deploying model to GPU ###
model = model.to(device)
trainer.train()

### Test ###
predictions = trainer.predict(tokenized_test)
logits = predictions.predictions                # Get the logits themselves
predicted_labels = np.argmax(logits, axis=1)    # Get the indices of the max logits
true_labels = predictions.label_ids             # Get the true labels

test_accuracy = accuracy_score(true_labels, predicted_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")