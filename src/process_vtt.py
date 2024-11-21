# extract_texts.py

import os
import json
import webvtt

# Step 1: Extract Text from .vtt Files
def extract_text_from_vtt(file_path):
    text = ''
    for caption in webvtt.read(file_path):
        # Remove lines that match the pattern of timestamps and metadata
        if '-->' not in caption.text and 'WEBVTT' not in caption.text:
            text += caption.text + ' '
    return text.replace('\n', ' ')

def load_vtt_files(directory):
    vtt_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.vtt')]
    all_texts = [extract_text_from_vtt(file) for file in vtt_files]
    return all_texts

# Function to save the extracted texts to a file
def save_texts_to_file(texts, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)

# Directory and file paths
directory = "D:/jworg"  # Update this path to your .vtt files directory
text_file_path = "D:/jworg/vtts.json"

# Check if the text file already exists. If it does, skip extraction.
if os.path.exists(text_file_path):
    print(f"{text_file_path} already exists. Skipping extraction.")
else:
    print("Extracting texts from .vtt files...")
    all_texts = load_vtt_files(directory)
    save_texts_to_file(all_texts, text_file_path)
    print(f"Processed text from {len(all_texts)} .vtt files")
    print(f"Extracted texts saved to {text_file_path}")