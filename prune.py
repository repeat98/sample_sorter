import os
import shutil

# Configuration
SOURCE_DIR = 'train/'    # Replace with the path to your original dataset
DEST_DIR = 'train_sm2/'      # Replace with the path to your pruned dataset
MAX_FILES = 10

def is_leaf_directory(dir_path):
    """
    Determines if a directory is a leaf directory (i.e., has no subdirectories).
    """
    return not any(os.path.isdir(os.path.join(dir_path, entry)) for entry in os.listdir(dir_path))

def copy_pruned_directory(source_dir, dest_dir):
    """
    Copies up to MAX_FILES from source_dir to dest_dir.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    # List all files in the source directory
    files = sorted([f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))])
    
    # Select files to copy
    files_to_copy = files[:MAX_FILES]
    
    for file in files_to_copy:
        src_file = os.path.join(source_dir, file)
        dest_file = os.path.join(dest_dir, file)
        shutil.copy2(src_file, dest_file)  # copy2 preserves metadata
        print(f"Copied: {src_file} --> {dest_file}")

def traverse_and_copy(source_current, dest_current):
    """
    Recursively traverses the source directory and copies files to the destination directory.
    """
    if is_leaf_directory(source_current):
        copy_pruned_directory(source_current, dest_current)
    else:
        for entry in os.listdir(source_current):
            source_entry_path = os.path.join(source_current, entry)
            dest_entry_path = os.path.join(dest_current, entry)
            if os.path.isdir(source_entry_path):
                traverse_and_copy(source_entry_path, dest_entry_path)

def main():
    """
    Main function to initiate the copying process.
    """
    if not os.path.exists(SOURCE_DIR):
        print(f"Source directory does not exist: {SOURCE_DIR}")
        return
    
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created destination directory: {DEST_DIR}")
    
    traverse_and_copy(SOURCE_DIR, DEST_DIR)
    print("Pruned dataset copy completed successfully.")

if __name__ == "__main__":
    main()