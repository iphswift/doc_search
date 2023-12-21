#!/usr/bin/env python3
import json
import numpy as np
import os
import glob
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Constants
CACHE_FILE_PATH = ".doc_search_embeddings.json"
CONFIG_FILE_PATH = ".doc_search_config"
RESULTS_PER_PAGE = 5

# Load the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text):
    """Get the embedding of a given text."""
    embedding = model.encode(text)
    return embedding.flatten()

def read_text_from_file(file_name):
    """Read text from a file."""
    with open(file_name, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def get_file_paths_from_config():
    """Read file patterns from the configuration file and return matched file paths."""
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Error: Configuration file '{CONFIG_FILE_PATH}' not found. Please create this file with file patterns to search.")
        exit()

    with open(CONFIG_FILE_PATH, 'r') as file:
        patterns = file.readlines()
    file_paths = set()
    for pattern in patterns:
        pattern = pattern.strip()
        if pattern:
            matched_files = glob.glob(pattern, recursive=True)
            file_paths.update(matched_files)
    return list(file_paths)

def process_and_cache_files(file_paths):
    """Process files, compute embeddings, and cache them."""
    cache = {}
    for file_path in file_paths:
        text = read_text_from_file(file_path)
        document_parts = text.split('\n')
        embeddings = [get_embedding(part).tolist() for part in document_parts]  # Convert numpy arrays to lists
        cache[file_path] = embeddings

    with open(CACHE_FILE_PATH, 'w') as file:
        json.dump(cache, file)

def load_cache():
    """Load embeddings cache from disk."""
    if not os.path.exists(CACHE_FILE_PATH):
        return {}

    with open(CACHE_FILE_PATH, 'r') as file:
        cache = json.load(file)
        for file_path in cache:
            cache[file_path] = [np.array(embedding) for embedding in cache[file_path]]  # Convert lists back to numpy arrays
    return cache

def calculate_max_similarity(sentence, file_paths, cache):
    """Calculate maximum similarity of a sentence with documents."""
    sentence_embedding = get_embedding(sentence)
    similarities = []

    for file_path in file_paths:
        if file_path not in cache:
            continue

        document_embeddings = cache[file_path]
        max_similarity = max([1 - cosine(sentence_embedding, part_embedding) for part_embedding in document_embeddings], default=0)
        similarities.append((file_path, max_similarity))

    return sorted(similarities, key=lambda x: x[1], reverse=True)

def display_results(similarities, start_index):
    """Display a page of results."""
    for i in range(start_index, min(start_index + RESULTS_PER_PAGE, len(similarities))):
        file_path, similarity = similarities[i]
        print(f"{i+1}. {file_path} - Similarity: {similarity}")

    if start_index + RESULTS_PER_PAGE < len(similarities):
        return input("Type 'more' to see more results, or anything else to return to search: ").strip() == 'more'
    return False

# Main program
def main():
    # Check for configuration file
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"Configuration file '{CONFIG_FILE_PATH}' not found. Please create this file with patterns of files to be searched.")
        return

    # Load cache from disk
    cache = load_cache()

    # Loop for taking search strings from terminal
    while True:
        command = input("Enter a search string, '!load' to load or reload files, or 'exit' to quit: ").strip()

        if command.lower() == 'exit':
            break

        if command == '!load':
            file_paths = get_file_paths_from_config()
            
            # Confirmation message
            confirm = input(f"You are about to load {len(file_paths)} documents. Do you want to proceed? (y/n): ").strip().lower()
            if confirm == 'y':
                process_and_cache_files(file_paths)
                cache = load_cache()
                print("Files reloaded and embeddings recomputed.")
            else:
                print("Loading cancelled.")
            continue

        if not cache:
            print("No embeddings found. Please enter '!load' to load the most recent file structure.")
            continue

        file_paths = get_file_paths_from_config()
        similarities = calculate_max_similarity(command, file_paths, cache)
        start_index = 0
        while start_index < len(similarities) and display_results(similarities, start_index):
            start_index += RESULTS_PER_PAGE

    print("Exited the similarity search.")

if __name__ == "__main__":
    main()
