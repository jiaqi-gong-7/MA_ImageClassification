import os
import shutil
import random
from PIL import Image
import pillow_heif

def convert_heic_to_jpg(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.HEIC'):
                heic_path = os.path.join(root, file)
                jpg_path = os.path.join(root, file.replace('.HEIC', '.jpg'))
                
                # Open HEIC file and convert to JPG
                heif_file = pillow_heif.read_heif(heic_path)
                image = Image.frombytes(
                    heif_file.mode, 
                    heif_file.size, 
                    heif_file.data, 
                    "raw"
                )
                image.save(jpg_path, "JPEG")
                print(f"Converted {heic_path} to {jpg_path}")
                
                # Optionally, delete the original HEIC file
                os.remove(heic_path)

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Get all category folders
    categories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for category in categories:
        category_path = os.path.join(source_dir, category)
        images = [f for f in os.listdir(category_path) if f.endswith('.jpg') or f.endswith('.png')]  # Only process jpg/png
        random.shuffle(images)  # Shuffle images

        train_split = int(len(images) * train_ratio)
        val_split = int(len(images) * (train_ratio + val_ratio))

        # Split into different datasets
        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        # Create category folders in each dataset directory
        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(val_dir, category), exist_ok=True)
        os.makedirs(os.path.join(test_dir, category), exist_ok=True)

        # Copy images to the target directories
        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category))
        for img in val_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(val_dir, category))
        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(test_dir, category))

        # Debug output
        print(f"Category: {category}")
        print(f"Total images: {len(images)}")
        print(f"Training images: {len(train_images)}")
        print(f"Validation images: {len(val_images)}")
        print(f"Testing images: {len(test_images)}")

# Define the directories
source_directory = 'data/all_images'
train_directory = 'data/train'
val_directory = 'data/val'
test_directory = 'data/test'

# Convert HEIC images in the source directory to JPG
convert_heic_to_jpg(source_directory)

# Split dataset into train, val, and test directories
split_dataset(source_directory, train_directory, val_directory, test_directory)



