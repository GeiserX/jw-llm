import os
import zipfile
import sqlite3
import tempfile
import shutil
import hashlib
import zlib
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad
import json
import traceback
from bs4 import BeautifulSoup  # For HTML parsing
import logging  # For better logging

def process_jwpub(file_path, catalog_db_conn, JW_LANG):
    texts = []

    logging.info(f"Starting processing of: {file_path}")

    tmpdirname = tempfile.mkdtemp()
    try:
        # Step 1: Extract contents of the jwpub file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdirname)

        contents_path = os.path.join(tmpdirname, 'contents')
        if not os.path.exists(contents_path):
            logging.warning("No 'contents' file found.")
            return texts

        contents_tmpdirname = tempfile.mkdtemp()
        try:
            try:
                with zipfile.ZipFile(contents_path, 'r') as zip_ref:
                    zip_ref.extractall(contents_tmpdirname)
            except zipfile.BadZipFile:
                logging.warning("Bad ZIP file in contents.")
                return texts

            # Find the .db file in the extracted contents
            db_files = [os.path.join(contents_tmpdirname, file) for file in os.listdir(contents_tmpdirname) if file.endswith('.db')]

            if not db_files:
                logging.warning("No database files found.")
                return texts

            db_file_path = db_files[0]

            conn = sqlite3.connect(db_file_path)
            cursor = conn.cursor()

            try:
                # Query the Publication table to get necessary fields
                cursor.execute("""
                    SELECT MepsLanguageIndex, Symbol, Year, IssueTagNumber
                    FROM Publication
                    LIMIT 1;
                """)
                row = cursor.fetchone()
                if not row:
                    logging.warning("No data in Publication table.")
                    conn.close()
                    return texts

                meps_language_index, symbol, year, issue_tag_number = row

                logging.debug(f"MepsLanguageIndex: {meps_language_index}")
                logging.debug(f"Symbol: '{symbol}'")
                logging.debug(f"Year: {year}")
                logging.debug(f"IssueTagNumber: {issue_tag_number}")

                # Build the hash input string
                fields = []
                try:
                    fields.append(str(meps_language_index).strip())
                except Exception as e:
                    logging.error(f"Error getting MepsLanguageIndex: {e}")
                    return texts
                try:
                    fields.append(symbol.strip())
                except Exception as e:
                    logging.error(f"Error getting Symbol: {e}")
                    return texts
                try:
                    fields.append(str(year).strip())
                except Exception as e:
                    logging.error(f"Error getting Year: {e}")
                    return texts

                issue_tag_number_str = str(issue_tag_number).strip()
                if issue_tag_number_str and issue_tag_number_str != '0':
                    fields.append(issue_tag_number_str)

                # Join the list with underscores
                joined_string = '_'.join(fields)
                logging.debug(f"Hash input string: '{joined_string}'")

                # Calculate SHA-256 hash of the string
                sha256_hash = hashlib.sha256(joined_string.encode('utf-8')).digest()
                logging.debug(f"SHA-256 hash: {sha256_hash.hex()}")

                # Known value to XOR with (ensure it's bytes)
                known_value_hex = '11cbb5587e32846d4c26790c633da289f66fe5842a3a585ce1bc3a294af5ada7'
                known_value = bytes.fromhex(known_value_hex)
                logging.debug(f"Known value for XOR: {known_value.hex()}")

                # XOR the hash with the known value
                final_hash = bytes(a ^ b for a, b in zip(sha256_hash, known_value))
                logging.debug(f"Final hash after XOR: {final_hash.hex()}")

                # Extract AES key and IV from the final hash
                aes_key = final_hash[:16]  # First 16 bytes
                aes_iv = final_hash[16:]   # Last 16 bytes
                logging.debug(f"AES Key: {aes_key.hex()}, Length: {len(aes_key)}")
                logging.debug(f"AES IV: {aes_iv.hex()}, Length: {len(aes_iv)}")

                # Step 2: Decrypt and decompress the content
                # Fetch all content entries from the Document table
                content_entries = []
                for table_name in ['Document', 'BibleChapter', 'BibleVerse']:
                    try:
                        cursor.execute(f"""
                            SELECT DocumentId, Content
                            FROM {table_name};
                        """)
                        rows = cursor.fetchall()
                        if rows:
                            content_entries.extend(rows)
                            logging.info(f"Found encrypted content in table '{table_name}', number of entries: {len(rows)}")
                    except sqlite3.Error as e:
                        logging.warning(f"Error accessing table '{table_name}': {e}")
                        continue

                if not content_entries:
                    logging.warning("No encrypted content found in any known table.")
                    conn.close()
                    return texts

                # Now, get publication data from catalog.db using MepsLanguageIndex, Symbol, IssueTagNumber
                catalog_cursor = catalog_db_conn.cursor()

                # Adjust IssueTagNumber for querying; if None or 0, set to 0
                issue_tag_number_for_query = issue_tag_number if issue_tag_number else 0

                # Query the catalog.db Publication table
                catalog_cursor.execute("""
                    SELECT 
                        PublicationRootKeyId,
                        MepsLanguageId,
                        PublicationTypeId,
                        IssueTagNumber,
                        Title,
                        IssueTitle,
                        ShortTitle,
                        CoverTitle,
                        UndatedTitle,
                        UndatedReferenceTitle,
                        Year,
                        Symbol,
                        KeySymbol,
                        Id
                    FROM Publication
                    WHERE MepsLanguageId=? AND Symbol=? AND (IssueTagNumber=? OR IssueTagNumber IS NULL)
                    LIMIT 1;
                """, (meps_language_index, symbol, issue_tag_number_for_query))

                pub_row = catalog_cursor.fetchone()
                if not pub_row:
                    logging.warning(f"No matching publication found in catalog.db for MepsLanguageId={meps_language_index}, Symbol='{symbol}', IssueTagNumber='{issue_tag_number_for_query}'")
                    conn.close()
                    return texts

                # Map the columns to their values
                pub_columns = [
                    'PublicationRootKeyId',
                    'MepsLanguageId',
                    'PublicationTypeId',
                    'IssueTagNumber',
                    'Title',
                    'IssueTitle',
                    'ShortTitle',
                    'CoverTitle',
                    'UndatedTitle',
                    'UndatedReferenceTitle',
                    'Year',
                    'Symbol',
                    'KeySymbol',
                    'Id'
                ]

                pub_data = dict(zip(pub_columns, pub_row))

                logging.info("Publication data extracted from catalog.db.")

                # Process entries and collect extracted text
                for idx, (document_id, encrypted_content) in enumerate(content_entries):
                    logging.info(f"Processing encrypted content entry {idx + 1}/{len(content_entries)}")
                    # logging.debug(f"Encrypted content length: {len(encrypted_content)} bytes")

                    try:
                        cipher = AES.new(aes_key, AES.MODE_CBC, aes_iv)
                        decrypted_data = cipher.decrypt(encrypted_content)
                    except Exception as e:
                        logging.error(f"Error decrypting data: {e}")
                        continue

                    try:
                        decrypted_data = unpad(decrypted_data, AES.block_size)
                    except ValueError as e:
                        logging.error(f"Error unpadding data: {e}")
                        continue

                    try:
                        decompressed_data = zlib.decompress(decrypted_data)
                    except Exception as e:
                        logging.error(f"Error decompressing data: {e}")
                        continue

                    # Decode the decompressed data as UTF-8 text
                    try:
                        html_content = decompressed_data.decode('utf-8', errors='replace')
                    except UnicodeDecodeError as e:
                        logging.error(f"Error decoding decompressed data: {e}")
                        html_content = ''

                    # Parse the HTML content to extract text
                    try:
                        soup = BeautifulSoup(html_content, 'html.parser')

                        # Extract paragraphs
                        paragraphs = soup.find_all('p')
                        paragraph_text = '\n\n'.join([para.get_text(strip=True) for para in paragraphs])

                        # If no paragraphs found, extract all text
                        if not paragraphs:
                            paragraph_text = soup.get_text(separator='\n', strip=True)

                        # Prepare the data entry
                        data_entry = {
                            'filename': os.path.basename(file_path),
                            'document_id': document_id,
                            'language': JW_LANG,  # Add language from JW_LANG variable
                            'text': paragraph_text
                        }

                        # Include all requested publication data
                        data_entry.update(pub_data)

                        texts.append(data_entry)

                    except Exception as e:
                        logging.error(f"Error parsing HTML content: {e}")
                        continue

            except Exception as e:
                logging.error(f"An unexpected error occurred: {e}")
                traceback.print_exc()
                return texts

            finally:
                cursor.close()
                conn.close()

        finally:
            shutil.rmtree(contents_tmpdirname)
    finally:
        shutil.rmtree(tmpdirname)

    logging.info(f"Finished processing of: {file_path}")
    return texts

def save_texts_to_file(texts, file_path):
    # Use double quotes in the JSON output (standard JSON format)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(texts, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    import argparse

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get JW_LANG from environment variable or default to 'S' for Spanish
    JW_LANG = os.environ.get('JW_LANG', 'S')
    # Set paths
    default_jwpubs_directory = "E:/jworg/jwpubs/"  # Update this path to your .jwpub files directory
    default_output_file_path = "E:/jworg/jwpubs/jwpubs.json"
    default_catalog_db_path = os.path.join(default_jwpubs_directory, "catalog.db")

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process .jwpub files and extract texts.')
    parser.add_argument('--directory', type=str, default=default_jwpubs_directory,
                        help=f'Directory containing .jwpub files (default: {default_jwpubs_directory})')
    parser.add_argument('--output', type=str, default=default_output_file_path,
                        help=f'Output JSON file path (default: {default_output_file_path})')
    parser.add_argument('--catalog_db', type=str, default=default_catalog_db_path,
                        help=f'Path to catalog.db file (default: {default_catalog_db_path})')

    args = parser.parse_args()

    directory = args.directory
    output_file_path = args.output
    catalog_db_path = args.catalog_db

    if not os.path.exists(directory):
        logging.error(f"The directory {directory} does not exist. Please check the path.")
    elif not os.path.exists(catalog_db_path):
        logging.error(f"The catalog.db file {catalog_db_path} does not exist. Please check the path.")
    else:
        all_texts = []
        # Connect to catalog.db
        catalog_db_conn = sqlite3.connect(catalog_db_path)
        # Iterate over all .jwpub files in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.jwpub'):
                jwpub_file = os.path.join(directory, filename)
                # Process the .jwpub file and extract texts
                texts = process_jwpub(jwpub_file, catalog_db_conn, JW_LANG)
                if texts:
                    all_texts.extend(texts)
                else:
                    logging.warning(f"No texts were extracted from {jwpub_file}.")

        # Close the catalog.db connection
        catalog_db_conn.close()

        if all_texts:
            save_texts_to_file(all_texts, output_file_path)
            logging.info(f"Extracted texts saved to {output_file_path}")
        else:
            logging.warning("No texts were extracted from any .jwpub files.")