#!/bin/bash

# Check if a folder was provided as an argument.
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <folder>"
  exit 1
fi

folder="$1"

# Verify that the provided argument is a valid directory.
if [ ! -d "$folder" ]; then
  echo "Error: '$folder' is not a valid directory."
  exit 1
fi

batch_size=1000        # Number of files per batch.
files=()               # Array to store file paths.
counter=0              # Counter for the current batch.
batch_number=1         # Batch counter for commit messages.

# Use find with -print0 to correctly handle filenames with spaces or special characters.
while IFS= read -r -d '' file; do
  files+=("$file")
  ((counter++))
  
  # When we reach the batch size, add and commit the batch.
  if [ "$counter" -eq "$batch_size" ]; then
    echo "Adding batch $batch_number with $counter files..."
    git add "${files[@]}"
    
    # You can customize the commit message as desired.
    git commit -m "Batch commit $batch_number: $counter files added"
    
    git push https://github.com/Panesico/anotherOCRcudatest master
    # Reset for the next batch.
    files=()
    counter=0
    ((batch_number++))
  fi
done < <(find "$folder" -type f -print0)

# Commit any remaining files that didn't fill a complete batch.
if [ "$counter" -gt 0 ]; then
  echo "Adding final batch $batch_number with $counter files..."
  git add "${files[@]}"
  git commit -m "Batch commit $batch_number: $counter files added"
fi

echo "All files have been added and committed in batches."
