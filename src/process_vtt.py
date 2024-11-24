# extract_texts.py

import os
import json
import webvtt

# Function to extract text from .vtt files
def extract_text_from_vtt(file_path):
    text = ''
    for caption in webvtt.read(file_path):
        # Remove lines that match the pattern of timestamps and metadata
        if '-->' not in caption.text and 'WEBVTT' not in caption.text:
            text += caption.text + ' '
    return text.replace('\n', ' ')

# Function to load the metadata from S.json
def load_metadata(sjson_path):
    metadata_map = {}
    with open(sjson_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get('type') == 'media-item':
                o = data.get('o', {})
                naturalKey = o.get('naturalKey')
                title = o.get('title')
                primaryCategory = o.get('primaryCategory')
                if naturalKey:
                    metadata_map[naturalKey] = {
                        'title': title,
                        'primaryCategory': primaryCategory,
                    }
    return metadata_map

# Function to load and process .vtt files
def load_vtt_files(directory, metadata_map):
    vtt_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.vtt')]
    all_data = []
    for vtt_file in vtt_files:
        text = extract_text_from_vtt(vtt_file)
        filename = os.path.basename(vtt_file)
        # Get naturalKey from filename (without extension)
        naturalKey = os.path.splitext(filename)[0]
        # Get metadata
        metadata = metadata_map.get(naturalKey, {})
        data_entry = {
            'text': text,
            'naturalKey': naturalKey,
            'title': metadata.get('title', ''),
            'primaryCategory': metadata.get('primaryCategory', ''),
        }
        all_data.append(data_entry)
    return all_data

# Function to save the extracted data to a file
def save_data_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Directory and file paths
directory = "E:/jworg"  # Update this path to your .vtt files directory
text_file_path = os.path.join(directory, "vtts.json")
sjson_path = os.path.join(directory, "S.json")

print("Loading metadata from S.json...")
metadata_map = load_metadata(sjson_path)
print(f"Loaded metadata for {len(metadata_map)} media items.")

print("Extracting texts from .vtt files...")
all_data = load_vtt_files(directory, metadata_map)
save_data_to_file(all_data, text_file_path)
print(f"Processed text from {len(all_data)} .vtt files")
print(f"Extracted data saved to {text_file_path}")