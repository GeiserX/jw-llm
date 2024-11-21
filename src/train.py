# train_llm.py

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Check GPU availability
print("Is CUDA available? ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of GPUs Available: ", torch.cuda.device_count())
    print("Current GPU Device: ", torch.cuda.current_device())
    print("GPU Name: ", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("CUDA is not available. PyTorch cannot use your GPU.")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load the texts from a file
def load_texts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# File paths to the extracted texts
text_file_paths = ["D:/jworg/vtts.json", "D:/jworg/jwpubs.json"]

# Load the texts from the files
all_texts = []

for file_path in text_file_paths:
    if os.path.exists(file_path):
        print(f"Loading extracted texts from {file_path}...")
        texts = load_texts_from_file(file_path)
        all_texts.extend(texts)
    else:
        print(f"Extracted texts file {file_path} not found. Please check the file path.")
        exit()

print(f"Loaded {len(all_texts)} texts for training.")

# Step 2: Preprocess Data
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')

# Set the pad_token to eos_token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Create a Dataset object from the list of texts
dataset = Dataset.from_dict({'text': all_texts})

# Define the tokenize function
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=2048)

# Tokenize the dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# Set the block size to the model's maximum input length for efficient training
block_size = 2048  # Adjust this value based on your model's max input length

# Group texts into blocks of block_size
def group_texts(examples):
    # Concatenate all the input_ids together
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples['input_ids'])
    # Drop the small remainder and create chunks of block_size
    total_length = (total_length // block_size) * block_size
    result = {
        k: [concatenated_examples[k][i:i + block_size] for i in range(0, total_length, block_size)]
        for k in concatenated_examples.keys()
    }
    # Set the labels to be the same as the input_ids for language modeling
    result['labels'] = result['input_ids'].copy()
    return result

print("Grouping texts into blocks...")
lm_dataset = tokenized_dataset.map(group_texts, batched=True)

# Optional: Split the dataset into training and evaluation datasets
split_datasets = lm_dataset.train_test_split(test_size=0.1)  # 10% evaluation data
train_dataset = split_datasets['train']
eval_dataset = split_datasets['test']

# Step 3: Fine-Tune the Model
try:
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Move model to GPU if available
model.to(device)

# Define the data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,  # Adjust this as needed
    per_device_train_batch_size=1,  # Adjust based on your GPU capacity
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    fp16=torch.cuda.is_available(),  # Enable mixed precision training if supported
    gradient_checkpointing=True,  # Useful for large models to save memory
    logging_dir='./logs',
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Provide evaluation dataset
    data_collator=data_collator,
)

# Start training
print("Starting training...")
trainer.train()

# Save the final model
trainer.save_model('fine_tuned_model')

# Step 4: Evaluation and Inference
def generate_text(prompt, model, tokenizer, max_length=50, num_return_sequences=1):
    model.eval()  # Set model to evaluation mode
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,  # To prevent repetition
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Load the fine-tuned model for inference
print("Loading fine-tuned model for inference...")
fine_tuned_model = AutoModelForCausalLM.from_pretrained('fine_tuned_model').to(device)

prompt = "¿Cuál es el nombre de Dios?"  # Replace with your actual prompt
print(f"Generating text for prompt: {prompt}")
generated_text = generate_text(prompt, fine_tuned_model, tokenizer)
print(f"Generated text: {generated_text}")

print("Script completed.")