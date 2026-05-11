import os
import shutil
import glob
from sklearn.model_selection import train_test_split
import kagglehub

def prepare_dataset(dest_dir="dataset/prepared"):
    """
    Parses MRL Eye dataset and organizes into train/val/test splits.
    Format of MRL image: subject_image_gender_glasses_state_reflections_lighting_sensor.png
    State: 0 is closed, 1 is open.
    """
    # Check if they downloaded a pre-structured dataset instead
    if os.path.exists("dataset/train/Closed_Eyes"):
        print("Found existing 'dataset/train/Closed_Eyes' directory. Using pre-structured data.")
        return True

    print("Downloading dataset using kagglehub...")
    source_dir = kagglehub.dataset_download("akashshingha850/mrl-eye-dataset")
    print("Path to dataset files:", source_dir)

    print("Organizing MRL Eye Dataset...")
    
    # Create target directories
    for split in ['train', 'val', 'test']:
        for cls in ['Open_Eyes', 'Closed_Eyes']:
            os.makedirs(os.path.join(dest_dir, split, cls), exist_ok=True)

    # Collect all images
    all_images = glob.glob(os.path.join(source_dir, "**", "*.png"), recursive=True)
    if not all_images:
        print("No .png files found in the source directory.")
        return False

    open_eyes = []
    closed_eyes = []

    for img_path in all_images:
        basename = os.path.basename(img_path)
        parts = basename.split('_')
        if len(parts) >= 5:
            state = parts[4]
            if state == '0':
                closed_eyes.append(img_path)
            elif state == '1':
                open_eyes.append(img_path)

    print(f"Found {len(open_eyes)} open eyes and {len(closed_eyes)} closed eyes.")

    # Split data (70% train, 15% val, 15% test)
    # Ensure reproducibility with random_state
    def split_data(data):
        train, temp = train_test_split(data, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
        return train, val, test

    open_train, open_val, open_test = split_data(open_eyes)
    closed_train, closed_val, closed_test = split_data(closed_eyes)

    # Copy files
    def copy_files(file_list, split_name, class_name):
        target_dir = os.path.join(dest_dir, split_name, class_name)
        print(f"Copying {len(file_list)} files to {target_dir}...")
        for i, f in enumerate(file_list):
            if i % 5000 == 0 and i > 0:
                print(f"  Copied {i} files...")
            shutil.copy(f, os.path.join(target_dir, os.path.basename(f)))

    copy_files(open_train, 'train', 'Open_Eyes')
    copy_files(open_val, 'val', 'Open_Eyes')
    copy_files(open_test, 'test', 'Open_Eyes')

    copy_files(closed_train, 'train', 'Closed_Eyes')
    copy_files(closed_val, 'val', 'Closed_Eyes')
    copy_files(closed_test, 'test', 'Closed_Eyes')

    print("Dataset preparation complete.")
    return True

if __name__ == "__main__":
    prepare_dataset()
