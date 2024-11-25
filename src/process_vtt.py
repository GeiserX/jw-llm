import os
import json
import webvtt
import sqlite3
from urllib.parse import urlparse, unquote
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to extract text from .vtt files
def extract_text_from_vtt(file_path):
    text = ''
    for caption in webvtt.read(file_path):
        # Remove lines that match the pattern of timestamps and metadata
        if '-->' not in caption.text and 'WEBVTT' not in caption.text:
            text += caption.text + ' '
    return text.replace('\n', ' ').strip()

# Function to load the metadata from S.json
def load_metadata(sjson_path):
    metadata_map = {}
    all_keys = set()

    # First pass to collect all possible metadata keys
    with open(sjson_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get('type') == 'media-item':
                o = data.get('o', {})
                # Collect all keys except 'images', 'checksums', 'keyParts', and 'issueDate'
                keys = set(o.keys()) - {'images', 'checksums', 'keyParts', 'issueDate'}
                # Include keys from 'keyParts'
                keyParts = o.get('keyParts', {})
                keys.update(keyParts.keys())
                all_keys.update(keys)

    # Second pass to build metadata map
    with open(sjson_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get('type') == 'media-item':
                o = data.get('o', {})
                naturalKey = o.get('naturalKey')
                if naturalKey:
                    metadata = {}
                    # Initialize all keys with None
                    for key in all_keys:
                        metadata[key] = None

                    # Extract metadata fields excluding 'images', 'checksums', 'keyParts', and 'issueDate'
                    for key in o.keys():
                        if key not in {'images', 'checksums', 'keyParts', 'issueDate'}:
                            metadata[key] = o.get(key)

                    # Include fields from 'keyParts' at the top level
                    keyParts = o.get('keyParts', {})
                    for key, value in keyParts.items():
                        metadata[key] = value

                    metadata_map[naturalKey] = metadata
    return metadata_map

# Function to build filename to naturalKey mapping from jw_media.db
def build_filename_mapping(db_path):
    filename_map = {}
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Get column names
        cursor.execute("PRAGMA table_info(downloaded_vtts)")
        columns_info = cursor.fetchall()
        column_names = [col[1] for col in columns_info]

        # Determine which column to use as the identifier
        if 'identifier' in column_names:
            id_column = 'identifier'
        elif 'pubSymbol' in column_names:
            id_column = 'pubSymbol'
        else:
            logging.error("Neither 'identifier' nor 'pubSymbol' column found in downloaded_vtts table.")
            return filename_map  # Return empty mapping

        # Execute the query using the appropriate identifier column
        cursor.execute(f"SELECT {id_column}, track, formatCode, vtt_url FROM downloaded_vtts WHERE status = 'success'")
        rows = cursor.fetchall()
        conn.close()
        for identifier, track, formatCode, vtt_url in rows:
            # Extract the filename from vtt_url
            parsed_url = urlparse(vtt_url)
            filename = os.path.basename(parsed_url.path)
            filename = unquote(filename)  # Decode URL-encoded characters

            # Reconstruct the naturalKey without relying on JW_LANG
            # Extract languageCode from the filename or vtt_url
            parts = filename.split('_')
            languageCode = None
            if len(parts) > 1:
                languageCode = parts[1]
                if '-' in languageCode:
                    # Handle cases like 'en-US'
                    languageCode = languageCode.split('-')[0]
                languageCode = languageCode.upper()
            else:
                # If unable to parse, default to None
                languageCode = None

            # Handle formatCode and convert to uppercase
            formatCode_upper = formatCode.upper() if formatCode else ''

            # Reconstruct naturalKey
            if isinstance(identifier, str) and not identifier.isdigit():
                # Assuming 'pubSymbol' case
                if languageCode:
                    naturalKey = f"pub-{identifier}_{languageCode}_{track}_{formatCode_upper}"
                else:
                    naturalKey = f"pub-{identifier}_{track}_{formatCode_upper}"
            else:
                # Assuming 'docID' case
                if languageCode:
                    naturalKey = f"docid-{identifier}_{languageCode}_{track}_{formatCode_upper}"
                else:
                    naturalKey = f"docid-{identifier}_{track}_{formatCode_upper}"

            filename_map[filename] = naturalKey

    except Exception as e:
        logging.error(f"Error building filename mapping: {e}")

    return filename_map

# Function to load and process .vtt files
def load_vtt_files(directory, metadata_map, filename_map):
    vtt_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.vtt')]
    all_data = []
    for vtt_file in vtt_files:
        text = extract_text_from_vtt(vtt_file)
        filename = os.path.basename(vtt_file)

        # Get naturalKey from filename mapping
        naturalKey = filename_map.get(filename)

        if not naturalKey:
            logging.warning(f"No mapping found for file {filename}, skipping.")
            continue

        # Get metadata
        metadata = metadata_map.get(naturalKey, {})
        if not metadata:
            logging.warning(f"No metadata found for naturalKey {naturalKey}, skipping.")
            continue

        data_entry = {
            'text': text,
            'filename': filename,
            'naturalKey': naturalKey
        }
        # Include all metadata fields except those with null values
        data_entry.update({k: v for k, v in metadata.items() if v is not None})
        all_data.append(data_entry)
    return all_data

# Function to save the extracted data to a file
def save_data_to_file(data, file_path):
    # Use default separators to minimize JSON size
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # Directory and file paths
    directory = "E:/jworg/vtts"  # Update this path to your .vtt files directory
    db_path = "E:/jworg/vtts/jw_media.db"  # Path to the jw_media.db database
    text_file_path = os.path.join(directory, "vtts.json")
    sjson_path = os.path.join(directory, "S.json")

    print("Loading metadata from S.json...")
    metadata_map = load_metadata(sjson_path)
    print(f"Loaded metadata for {len(metadata_map)} media items.")

    print("Building filename to naturalKey mapping from jw_media.db...")
    filename_map = build_filename_mapping(db_path)
    print(f"Built filename mapping for {len(filename_map)} files.")

    print("Extracting texts from .vtt files...")
    all_data = load_vtt_files(directory, metadata_map, filename_map)
    save_data_to_file(all_data, text_file_path)
    print(f"Processed text from {len(all_data)} .vtt files")
    print(f"Extracted data saved to {text_file_path}")