import os
import json
import torch
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import random
from math import ceil

# Function to corrupt text by masking random spans using sentinel tokens
def corrupt_text(text, tokenizer, noise_density=0.15, mean_noise_span_length=3):
    tokens = tokenizer.encode(text, add_special_tokens=False, max_length=512, truncation=True)
    total_tokens = len(tokens)

    if total_tokens == 0:
        # If the text is empty, return it as is
        return text

    num_to_mask = max(1, int(ceil(total_tokens * noise_density)))
    num_spans = max(1, int(ceil(num_to_mask / mean_noise_span_length)))
    num_spans = min(total_tokens, num_spans)  # Ensure num_spans does not exceed total_tokens

    # Randomly select span start indices
    span_starts = sorted(random.sample(range(total_tokens), num_spans))

    corrupted_tokens = []
    previous_end = 0
    for i, start in enumerate(span_starts):
        # Randomly determine the span length
        span_length = random.randint(1, mean_noise_span_length)
        end = min(start + span_length, total_tokens)

        # Append tokens before the span
        corrupted_tokens.extend(tokens[previous_end:start])

        # Append sentinel token
        corrupted_tokens.append(tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>"))

        # Update previous_end
        previous_end = end

    # Append any remaining tokens after the last span
    corrupted_tokens.extend(tokens[previous_end:])

    # Decode the corrupted tokens to text
    corrupted_text = tokenizer.decode(corrupted_tokens, skip_special_tokens=False)
    return corrupted_text

# Define preprocessing function
def preprocess_function(examples):
    inputs = examples['text']
    # Corrupt the input texts
    corrupted_texts = [corrupt_text(text, tokenizer) for text in inputs]
    # Tokenize the corrupted texts
    model_inputs = tokenizer(corrupted_texts, truncation=True, padding='max_length', max_length=512)

    # Tokenize the targets (original texts)
    labels = tokenizer(inputs, truncation=True, padding='max_length', max_length=512)
    # Replace padding token ids in labels with -100 so they are ignored in loss computation
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

if __name__ == '__main__':
    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)

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

    # Function to load the texts from a JSON file
    def load_texts_from_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # File paths to the extracted texts
    text_file_paths = ["E:/jworg/vtts/vtts.json", "E:/jworg/jwpubs/jwpubs.json"]

    # Load the texts from the files
    all_texts = []

    for file_path in text_file_paths:
        if os.path.exists(file_path):
            print(f"Loading extracted texts from {file_path}...")
            texts = load_texts_from_file(file_path)
            # Extract only the 'text' field
            texts_only = [item['text'] for item in texts if 'text' in item]
            all_texts.extend(texts_only)
        else:
            print(f"Extracted texts file {file_path} not found. Please check the file path.")
            exit()

    print(f"Loaded {len(all_texts)} texts for training.")

    # Create a Dataset object from the list of texts
    dataset = Dataset.from_dict({'text': all_texts})

    # Initialize the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-xl')

    # Ensure tokenizer uses the correct special tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply the preprocessing to the dataset
    print("Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=['text'],
        num_proc=1  # Set to 1 to avoid multiprocessing issues (adjust as needed)
    )

    # Split the dataset into training and evaluation datasets
    split_datasets = tokenized_dataset.train_test_split(test_size=0.1)  # 10% evaluation data
    train_dataset = split_datasets['train']
    eval_dataset = split_datasets['test']

    # Load the pre-trained model
    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-xl')

    # Move model to GPU if available
    model.to(device)

    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,  # Adjust this as needed
        per_device_train_batch_size=1,  # Adjust based on your GPU capacity
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=500,  # Adjust based on your dataset size
        save_steps=500,  # Adjust based on your dataset size
        save_total_limit=2,
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # Enable mixed precision training if supported
        logging_dir='./logs',
        predict_with_generate=True,
    )

    # Define data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100, return_tensors='pt'
    )

    # Create Seq2SeqTrainer instance
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Provide evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model('xl')

    # Step 4: Evaluation and Inference
    def generate_text(prompt, model, tokenizer, max_length=50, num_return_sequences=1):
        model.eval()  # Set model to evaluation mode
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=512).to(device)
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
    fine_tuned_model = T5ForConditionalGeneration.from_pretrained('xl').to(device)

    prompt = "¿Cuál es el nombre de Dios?"  # Replace with your actual prompt
    print(f"Generating text for prompt: {prompt}")
    generated_text = generate_text(prompt, fine_tuned_model, tokenizer)
    print(f"Generated text: {generated_text}")

    print("Script completed.")