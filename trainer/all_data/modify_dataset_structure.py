import os
import shutil
import csv
from pathlib import Path

def read_text_file(file_path):
    """Read content from a text file."""
    with open(file_path, 'r') as f:
        return f.read().strip()

def create_csv_from_labels(base_dir, split_type):
    """Create CSV entries from label files for a specific split."""
    entries = []
    labels_dir = os.path.join(base_dir, split_type, 'labels')
    
    if not os.path.exists(labels_dir):
        return entries
    
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            image_name = label_file.replace('.txt', '.jpg')
            label_content = read_text_file(os.path.join(labels_dir, label_file))
            entries.append([image_name, label_content])
    
    return entries

def restructure_dataset(dataset_path):
    """Restructure the dataset by creating CSVs and copying images."""
    dataset_path = Path(dataset_path)
    
    # Create CSV files for each split
    splits = ['train', 'test', 'val']
    for split in splits:
        # Get entries for the current split
        entries = create_csv_from_labels(dataset_path, split)
        
        if entries:
            # Create CSV file
            csv_path = dataset_path / split / 'labels.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'words'])  # Header
                writer.writerows(entries)
            
            print(f"Created {split}_labels.csv")
            
            # Copy images to parent directory
            source_images_dir = dataset_path / split / 'images'
            target_images_dir = dataset_path / split
            
            if source_images_dir.exists():
                for image in source_images_dir.glob('*.jpg'):
                    shutil.copy2(image, target_images_dir / image.name)
                print(f"Copied images for {split} split")
        shutil.rmtree(dataset_path / split / 'images')
        shutil.rmtree(dataset_path / split / 'labels')

def main():
    # Assuming script is run from the parent directory of crnn_dataset
    dataset_path = 'crnn_dataset'
    
    try:
        restructure_dataset(dataset_path)
        print("Dataset restructuring completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()