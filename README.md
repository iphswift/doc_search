# Document Search Project

This document search tool allows you to search for textual similarities across multiple documents. It uses advanced sentence embedding techniques to compute and compare text embeddings.

## Prerequisites

Before you can run the script, ensure that you have the following prerequisites installed:

- Python 3 (Python 3.6 or later is recommended)
- `sentence-transformers` Python library
- Other dependencies as required by `sentence-transformers` (like `torch`)

## Installation

1. **Install Python 3**:
   - If not already installed, download and install Python 3 from [python.org](https://www.python.org/).

2. **Install Required Python Libraries**:
   - Open your command line interface (CLI) and run the following command:
     ```bash
     pip install sentence-transformers scipy
     ```

## Configuration

Before running the script, you need to set up a `.doc_search_config` file. This file contains patterns of files you want to include in the search.

### `.doc_search_config` File Format

- Each line in the `.doc_search_config` file should contain a pattern that matches the files you wish to search.
- Patterns are Unix shell-style wildcards. For example:
  - `*.txt` matches all text files.
  - `./folder/*.py` matches all Python files in `folder`.
- Blank lines and lines starting with `#` are ignored as comments.

Example:

```
# Match all Python files in src directory
./src/*.py

# Match all TXT files in documents
./documents/*.txt
```

## Running the Script

To run the script, use the following command in your CLI:

```bash
python3 doc_search_script.py
```

- The script will prompt you to enter a search string or a command.
- You can type a sentence or phrase to search across the documents.
- Type `!load` to reload and reprocess the files as per the `.doc_search_config`.
- Type `exit` to quit the script.

## Output

- The script displays the top 5 documents with the highest similarity to the entered search string.
- Type `more` to view the next set of results or any other key to return to the search.

### Disclaimer
This project was coded using ChatGPT as a collaborative tool.
