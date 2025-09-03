#!/bin/bash

# This script extracts all supported archives in a specified directory,
# providing real-time progress updates.
# Usage: ./extract_with_progress.sh <path_to_directory>

# Check if a directory path is provided
if [ -z "$1" ]; then
  echo "Error: Please provide a directory path."
  echo "Usage: $0 <path_to_directory>"
  exit 1
fi

# Check if the provided path is a valid directory
if [ ! -d "$1" ]; then
  echo "Error: Directory '$1' not found."
  exit 1
fi

# Navigate to the specified directory. The double quotes handle paths with spaces.
echo "Navigating to '$1'..."
cd "$1" || exit

# Use nullglob to prevent the glob from expanding to a literal string if no files are found
shopt -s nullglob

# Gather all supported archive files into an array
archive_files=(*.tar *.tar.gz *.tgz *.zip)
total_files=${#archive_files[@]}

# Check if any archive files were found
if [ "$total_files" -eq 0 ]; then
    echo "No supported archive files found in this directory."
    exit 0
fi

echo "---"
echo "Detected $total_files archive files. Starting extraction."
echo "---"

# Initialize a counter
counter=0

# Loop through the array of archive files
for file in "${archive_files[@]}"; do
    counter=$((counter + 1))
    
    echo "Processing ($counter/$total_files): $file"
    
    case "$file" in
        *.tar)
            tar -xf "$file"
            ;;
        *.tar.gz | *.tgz)
            tar -zxf "$file"
            ;;
        *.zip)
            unzip "$file"
            ;;
        *)
            echo "Skipping unsupported file: $file"
            ;;
    esac
    echo "--------------------------"
done

echo "Extraction of all files is complete."