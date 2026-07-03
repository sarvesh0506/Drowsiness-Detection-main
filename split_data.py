import os
import random
import shutil

# --- 1. SETUP PATHS ---
# The main directory where your 'open_eyes' and 'closed_eyes' folders are located.
SOURCE_DATA_DIR = 'data' 

# The names of your class folders.
CLASSES = ['open_eyes', 'closed_eyes']

# The names for your new split folders.
SPLIT_DIRS = ['trains', 'vals', 'tests']

# Define the ratio for splitting the data.
SPLIT_RATIOS = {'trains': 0.7, 'vals': 0.15, 'tests': 0.15}

# --- 2. CREATE THE NEW FOLDER STRUCTURE ---
print("Creating new directory structure...")
for split in SPLIT_DIRS:
    for class_name in CLASSES:
        # This creates folders like 'data/trains/open_eyes'
        os.makedirs(os.path.join(SOURCE_DATA_DIR, split, class_name), exist_ok=True)

# --- 3. SPLIT AND MOVE THE FILES ---
print("Processing and splitting files...")
for class_name in CLASSES:
    source_class_dir = os.path.join(SOURCE_DATA_DIR, class_name)
    
    # Check if the source folder exists and is not empty
    if not os.path.exists(source_class_dir) or not os.listdir(source_class_dir):
        print(f"Warning: Source folder '{source_class_dir}' is empty or does not exist. Skipping.")
        continue

    filenames = os.listdir(source_class_dir)
    random.shuffle(filenames) # Shuffle for randomness
    
    # Calculate split points
    train_split_end = int(len(filenames) * SPLIT_RATIOS['trains'])
    val_split_end = train_split_end + int(len(filenames) * SPLIT_RATIOS['vals'])
    
    # Get the filenames for each set
    train_files = filenames[:train_split_end]
    val_files = filenames[train_split_end:val_split_end]
    test_files = filenames[val_split_end:]
    
    # Move files to their new locations
    for f in train_files:
        shutil.move(os.path.join(source_class_dir, f), os.path.join(SOURCE_DATA_DIR, 'trains', class_name, f))
    for f in val_files:
        shutil.move(os.path.join(source_class_dir, f), os.path.join(SOURCE_DATA_DIR, 'vals', class_name, f))
    for f in test_files:
        shutil.move(os.path.join(source_class_dir, f), os.path.join(SOURCE_DATA_DIR, 'tests', class_name, f))

print("\nData splitting complete!")
print("Your original 'open_eyes' and 'closed_eyes' folders should now be empty.")