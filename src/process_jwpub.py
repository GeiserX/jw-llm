# process_jwpubs.py

import os
import zipfile
import json
import sqlite3
import tempfile
import shutil
import hashlib
import zlib
from Crypto.Cipher import AES
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize
import hashlib

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def process_jwpub(file_path):
    texts = []
    
    # Create a temporary directory to extract files
    tmpdirname = tempfile.mkdtemp()
    try:
        # Unzip the .jwpub file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)
        
        # Locate the 'contents' file
        contents_path = os.path.join(tmpdirname, 'contents')
        
        if not os.path.exists(contents_path):
            print(f"'contents' file not found in {file_path}")
            return texts
        
        # Unzip the 'contents' file to get the SQLite database
        contents_tmpdirname = tempfile.mkdtemp()
        try:
            try:
                with zipfile.ZipFile(contents_path, 'r') as zip_ref:
                    zip_ref.extractall(contents_tmpdirname)
            except zipfile.BadZipFile:
                print(f"Error: {contents_path} is not a valid zip file.")
                return texts
            
            # Find the SQLite database file
            db_files = [os.path.join(contents_tmpdirname, file) for file in os.listdir(contents_tmpdirname) if file.endswith('.db')]

            if not db_files:
                print(f"No SQLite database file found in {contents_path}")
                return texts

            db_file_path = db_files[0]  # Assuming there's only one database file

            # Begin processing the database
            conn = sqlite3.connect(db_file_path)
            cursor = conn.cursor()

            # Step 1: Calculate the publication card hash
            try:
                # Query the required fields from the Publication table
                cursor.execute("""
                    SELECT MepsLanguageIndex, Symbol, Year, IssueTagNumber
                    FROM Publication
                    LIMIT 1;
                """)
                row = cursor.fetchone()
                if not row:
                    print("No data found in Publication table.")
                    conn.close()
                    return texts

                meps_language_index, symbol, year, issue_tag_number = row

                # Create the list of fields
                fields = [str(meps_language_index), symbol, str(year)]
                if issue_tag_number != '0' and issue_tag_number != '':
                    fields.append(issue_tag_number)

                # Join the list with underscores
                joined_string = '_'.join(fields)

                # Calculate SHA-256 hash of the string
                sha256_hash = hashlib.sha256(joined_string.encode('utf-8')).digest()

                # Known value to XOR with
                known_value_hex = '11cbb5587e32846d4c26790c633da289f66fe5842a3a585ce1bc3a294af5ada7'
                known_value = bytes.fromhex(known_value_hex)

                # XOR the hash with the known value
                final_hash = bytes(a ^ b for a, b in zip(sha256_hash, known_value))

                # Extract AES key and IV from the final hash
                aes_key = final_hash[:16]  # First 16 bytes
                aes_iv = final_hash[16:]   # Last 16 bytes

                # Step 2: Decrypt and decompress the content
                # Fetch the encrypted content from the Document table
                cursor.execute("""
                    SELECT Content
                    FROM Document
                    LIMIT 1;
                """)
                row = cursor.fetchone()
                if not row:
                    print("No content found in Document table.")
                    conn.close()
                    return texts

                encrypted_content = row[0]  # This is the BLOB field

                # Initialize the AES cipher
                cipher = AES.new(aes_key, AES.MODE_CBC, aes_iv)

                # Decrypt the content
                decrypted_data = cipher.decrypt(encrypted_content)

                # Remove PKCS7 padding
                padding_length = decrypted_data[-1]
                if isinstance(padding_length, str):
                    padding_length = ord(padding_length)
                decrypted_data = decrypted_data[:-padding_length]

                # Decompress using zlib
                decompressed_data = zlib.decompress(decrypted_data)

                # Decode to string
                text_content = decompressed_data.decode('utf-8')

                # Use DocumentParagraph to segment the content
                cursor.execute("""
                    SELECT BeginPosition, EndPosition 
                    FROM DocumentParagraph 
                    ORDER BY ParagraphIndex;
                """)
                paragraphs = cursor.fetchall()

                # Extract each paragraph using the positions
                extracted_paragraphs = []
                for begin_pos, end_pos in paragraphs:
                    paragraph_text = text_content[begin_pos:end_pos]
                    extracted_paragraphs.append(paragraph_text.strip())

                # Combine paragraphs into full text
                full_text = '\n\n'.join(extracted_paragraphs)

                # Process the text
                cleaned_text = process_text(full_text)

                texts.append(cleaned_text)

            except Exception as e:
                print(f"An error occurred while processing {file_path}: {e}")
                return texts

            finally:
                # Close the connection
                cursor.close()
                conn.close()

        finally:
            # Delete the contents temporary directory
            shutil.rmtree(contents_tmpdirname)
    finally:
        # Delete the main temporary directory
        shutil.rmtree(tmpdirname)

    return texts

# Text processing functions

def process_text(html_content):
    # Step 1: Strip HTML tags
    text = strip_html(html_content)
    # Step 2: Remove page numbers and references
    text = remove_page_numbers_and_references(text)
    # Step 3: Remove remaining HTML artifacts
    text = remove_html_artifacts(text)
    # Step 4: Normalize whitespace and remove non-printable characters
    text = normalize_whitespace(text)
    # Step 6: Split into sentences
    sentences = split_into_sentences(text)
    # Step 7: Remove repeated phrases within sentences
    sentences = [remove_repeated_phrases_in_sentence(s) for s in sentences]
    # Step 8: Remove duplicate sentences using hashing
    unique_sentences = remove_duplicate_sentences_hash(sentences)
    # Step 9: Compile cleaned text
    cleaned_text = compile_text(unique_sentences)
    return cleaned_text

def strip_html(html_content):
    # Use lxml parser for better handling of malformed HTML
    soup = BeautifulSoup(html_content, 'lxml')
    text = soup.get_text()
    return text

def remove_page_numbers_and_references(text):
    # Remove patterns like 'Página N' or 'Page N'
    text = re.sub(r'\b(Página|Page)\s+\d+\b', '', text)
    # Remove page numbers in text like 'p3' or 'id="p3"'
    text = re.sub(r'\b(p|id="p)\d+"?\b', '', text)
    # Remove references like '(Hebreos 3:4)'
    text = re.sub(r'\([^)]*\d+:\d+[^)]*\)', '', text)
    return text

def remove_html_artifacts(text):
    # Remove HTML attribute patterns
    pattern = r'\b(?:class|id|data-pid|data-bid|data-no|data-before-text|style|width|height|aria-hidden|href|data-xtid|data-bid|data-no|data-before-text)\s*=\s*"[^"]*"'
    text = re.sub(pattern, '', text)
    # Remove any remaining '<' or '>' characters
    text = text.replace('<', '').replace('>', '')
    # Remove escaped quotes
    text = text.replace('\\"', '')
    # Remove remaining HTML entities
    text = re.sub(r'&[^;\s]+;', '', text)
    return text

def remove_repeated_phrases(text):
    # Remove repeated phrases within the text
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text, flags=re.I)
    return text

def remove_repeated_phrases_in_sentence(sentence):
    words = sentence.split()
    seen = set()
    result = []
    for word in words:
        # Only add the word if it's not already in seen
        if word not in seen:
            seen.add(word)
            result.append(word)
    return ' '.join(result)

def remove_duplicate_sentences_hash(sentences):
    seen_hashes = set()
    unique_sentences = []
    for sentence in sentences:
        # Normalize sentence for hashing
        normalized_sentence = ''.join(e for e in sentence.lower() if e.isalnum())
        sentence_hash = hashlib.md5(normalized_sentence.encode()).hexdigest()
        if sentence_hash not in seen_hashes:
            seen_hashes.add(sentence_hash)
            unique_sentences.append(sentence)
    return unique_sentences

def normalize_whitespace(text):
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove any non-printable characters
    #text = re.sub(r'[^\x20-\x7E]', '', text)
    text = text.strip()
    return text

def split_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

from difflib import SequenceMatcher

def remove_similar_sentences(sentences, threshold=0.9):
    unique_sentences = []
    for sentence in sentences:
        is_duplicate = False
        for unique_sentence in unique_sentences:
            # Normalize sentences for comparison
            s1 = sentence.strip().lower()
            s2 = unique_sentence.strip().lower()
            # Skip very short sentences
            if len(s1) < 20 or len(s2) < 20:
                continue
            similarity = SequenceMatcher(None, s1, s2).ratio()
            if similarity > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_sentences.append(sentence)
    return unique_sentences

def compile_text(sentences):
    text = ' '.join(sentences)
    return text

# Main function to process all .jwpub files
def process_all_jwpubs(directory):
    all_texts = []

    # Find all .jwpub files in the directory
    jwpub_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.jwpub')]

    for file in jwpub_files:
        print(f"\nProcessing {file}...")
        texts = process_jwpub(file)
        if texts:
            all_texts.extend(texts)
            print(f"Extracted text from {file}")
        else:
            print(f"No texts extracted from {file}")

    return all_texts

# Function to save the extracted texts to a file
def save_texts_to_file(texts, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    # Directory and file paths
    directory = "D:/jworg"  # Update this path to your .jwpub files directory
    output_file_path = "D:/jworg/jwpubs.json"
    
    # Process all .jwpub files and extract texts
    all_texts = process_all_jwpubs(directory)
    
    if all_texts:
        # Save the extracted texts to a file
        save_texts_to_file(all_texts, output_file_path)
        print(f"\nExtracted texts saved to {output_file_path}")
    else:
        print("\nNo texts were extracted from the .jwpub files.")