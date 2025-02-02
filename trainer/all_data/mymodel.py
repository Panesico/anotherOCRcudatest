import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

class DataLoader:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.splits = ['train', 'test', 'val']
        self.data = {}
        for split in self.splits:
            split_path = self.base_path / split
            csv_file = list(split_path.glob('*.csv'))[0]
            self.data[split] = {
                'df': pd.read_csv(csv_file),
                'path': split_path
            }

    def load_image(self, split, filename):
        img_path = self.data[split]['path'] / filename
        return cv2.imread(str(img_path))

    def get_label(self, split, filename):
        df = self.data[split]['df']
        return df[df['filename'] == filename]['text'].iloc[0]

class ImagePreprocessor:
    def __init__(self):
        self.kernel = np.ones((3, 3), np.uint8)

    def preprocess(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Gentle denoising with non-local means
        denoised = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
        denoised = cv2.fastNlMeansDenoising(denoised, None, h=20, templateWindowSize=5, searchWindowSize=15)
        denoised2 = cv2.fastNlMeansDenoising(denoised, None, h=15, templateWindowSize=5, searchWindowSize=10)

        # Text enhancement after joining
        gaussian_blurred = cv2.GaussianBlur(denoised2, (3, 3), 1.0)
        unsharp_masked = cv2.addWeighted(denoised2, 1.5, gaussian_blurred, -0.5, 0)

        # Local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(unsharp_masked)

        # Edge-preserving smoothing
        text_enhanced = cv2.bilateralFilter(contrast_enhanced, d=5, sigmaColor=75, sigmaSpace=75)

        # Final adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            text_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4
        )

        gaussian_blurred = cv2.GaussianBlur(adaptive, (3, 3), 1.0)
        
        adaptive = cv2.adaptiveThreshold(
            gaussian_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4
        )

        return adaptive

def process_image(args):
    preprocessor, split, input_path, filename = args

    try:
        # Load image
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"Failed to load image {filename}")
            return filename, False

        # Process image
        processed_image = preprocessor.preprocess(image)

        # Save processed image
        cv2.imwrite(str(input_path), processed_image)
        return filename, True

    except Exception as e:
        print(f"Error processing {filename} in {split}: {str(e)}")
        return filename, False

def process_split(split, data_loader, preprocessor):
    print(f"\nProcessing {split} split...")

    df = data_loader.data[split]['df']
    total_images = len(df)

    args_list = [
        (preprocessor, split, data_loader.data[split]['path'] / row['filename'], row['filename'])
        for _, row in df.iterrows()
    ]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_image, args_list)

    success_count = sum(1 for _, success in results if success)
    print(f"\nFinished processing {split} split: {success_count}/{total_images} images successfully processed.")

def main():
    # Initialize loader and preprocessor
    data_loader = DataLoader("./crnn_dataset")
    preprocessor = ImagePreprocessor()

    # Process each split in parallel
    for split in data_loader.splits:
        process_split(split, data_loader, preprocessor)

    print("\nProcessing complete for all splits!")

if __name__ == "__main__":
    main()
