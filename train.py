import os
import webvtt
import json
import torch
from transformers import MistralTokenizer, MistralForCausalLM, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Step 1: Extract Text from .vtt Files
def extract_text_from_vtt(file_path):
    text = ''
    for caption in webvtt.read(file_path):
        text += caption.text + ' '
    return text

def load_vtt_files(directory):
    vtt_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.vtt')]
    all_texts = [extract_text_from_vtt(file) for file in vtt_files]
    return all_texts

# Function to save the extracted texts to a file
def save_texts_to_file(texts, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)

# Function to load the texts from a file
def load_texts_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Directory and file paths
directory = "D:/jworg"
text_file_path = "extracted_texts.json"

# Check if the text file already exists. If it does, load the texts from the file.
# If it doesn't, extract the texts and save them to the file.
if os.path.exists(text_file_path):
    print("Loading extracted texts from file...")
    all_texts = load_texts_from_file(text_file_path)
else:
    print("Extracting texts from .vtt files...")
    all_texts = load_vtt_files(directory)
    save_texts_to_file(all_texts, text_file_path)

print(f"Processed text from {len(all_texts)} .vtt files")

# Rest of your script remains the same...
# Step 2: Preprocess Data
print("Loading tokenizer and model...")
tokenizer = MistralTokenizer.from_pretrained('mistral')

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return encodings.input_ids[0], encodings.attention_mask[0]

max_length = 512
dataset = TextDataset(all_texts, tokenizer, max_length)

# Step 3: Fine-Tune the Model
model = MistralForCausalLM.from_pretrained('mistral')

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 3.2 Training Setup
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 3.3 Start Training
print("Starting training...")
trainer.train()

# Step 4: Evaluation and Inference
def generate_text(prompt, model, tokenizer, max_length=50, num_return_sequences=1):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

prompt = "Your initial text"
generated_text = generate_text(prompt, model, tokenizer)
print(f"Generated text: {generated_text}")

print("Script completed.")
