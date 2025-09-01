import gzip
import json
import pprint
import os
import sys

def read_json_data(path):
    """
    Reads and parses a .json or .json.gz file.

    Args:
        path (str): The file path.

    Returns:
        dict or list: The parsed Python object (dictionary or list).
        None: If the file is not found or parsing fails.
    """
    try:
        # Check the file extension to determine how to open the file
        if path.endswith('.gz'):
            # Use gzip.open for compressed files
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                return data
        elif path.endswith('.json'):
            # Use standard open for uncompressed files
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data
        else:
            print(f"Error: Unsupported file format for '{path}'. Only .json and .json.gz are supported.")
            return None
    except FileNotFoundError:
        print(f"Error: File '{path}' not found. Please ensure the path is correct.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The content of '{path}' is not a valid JSON format.")
        return None
    except Exception as e:
        print(f"An unknown error occurred: {e}")
        return None

# --- Main program ---
if __name__ == "__main__":
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <file_path>")
        sys.exit(1)

    # Get the file path from the command-line arguments
    file_to_process = sys.argv[1]

    # Read the data
    dataset = read_json_data(file_to_process)

    # If data was successfully read, proceed with analysis and display
    if dataset:
        print(f"Successfully loaded data from '{file_to_process}'!\n")

        # Check the top-level data structure
        if isinstance(dataset, dict):
            print("--- Top-level data structure (Keys) ---")
            print(list(dataset.keys()))
            print("-" * 30 + "\n")

            # Assume 'episodes' is the main data list, based on the provided file structure
            if 'episodes' in dataset:
                episodes_list = dataset['episodes']
                print(f"The dataset contains {len(episodes_list)} 'episodes' (entries).\n")
                
                # To avoid overwhelming the output, we'll only print a detailed
                # structure of the first episode
                if len(episodes_list) > 0:
                    print("--- Example data structure of the first 'episode' ---")
                    pprint.pprint(episodes_list[0])
                else:
                    print("'episodes' list is empty.")
            else:
                print("The top-level dictionary does not contain an 'episodes' key.")
        elif isinstance(dataset, list):
            print("--- Data structure preview ---")
            print("The top level of the data is a list.")
            if len(dataset) > 0:
                print(f"This list contains {len(dataset)} elements.")
                print("--- Example structure of the first element ---")
                pprint.pprint(dataset[0])
            else:
                print("The list is empty.")
        else:
            print("The data format is neither a dictionary nor a list.")