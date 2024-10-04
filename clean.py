import os
import sys
import numpy as np
import librosa
import tensorflow as tf
import pickle
from tqdm import tqdm
import shutil

# --------------------------- Parameters ---------------------------

# Audio processing parameters (should match those used in training)
SAMPLE_RATE = 22050
DURATION = 5  # Duration to which all audio files will be truncated or padded
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 20  # Should match the value in training script
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Supported audio file extensions
AUDIO_EXTENSIONS = (
    '.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif',
    '.aac', '.wma', '.m4a', '.alac', '.opus'
)

# Path to the training dataset
TRAINING_DATASET_PATH = 'train_backup/'  # Update this path as needed

# Path to the trained model and label encoder
MODEL_PATH = 'model/audio_classification_model.keras'
LABEL_ENCODER_PATH = 'model/label_encoder.pkl'

# Set this to True if you want to delete files immediately
# Set to False to move files to a 'misclassified' folder instead
DELETE_FILES = False

# ------------------------ Load Model and Encoder ------------------------

# Load the trained model and label encoder
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    print("Successfully loaded trained model and label encoder.")
except Exception as e:
    print(f"Error loading model or label encoder: {e}")
    sys.exit(1)

# Get the list of labels from the label encoder
model_labels = le.classes_

# ------------------------ Functions ------------------------

def extract_features_from_file(file_path):
    try:
        x, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        # Ensure x has the same length as during training (padded or truncated)
        max_len = SAMPLE_RATE * DURATION
        if len(x) > max_len:
            x = x[:max_len]
        else:
            pad_width = max_len - len(x)
            x = np.pad(x, (0, pad_width), 'constant')
        # Now extract MFCC features
        mfcc = librosa.feature.mfcc(y=x, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mfcc = librosa.power_to_db(mfcc, ref=np.max)
        mfcc = mfcc[..., np.newaxis]  # Add channel dimension
        return mfcc
    except Exception as e:
        print(f"Error extracting features from '{file_path}': {e}")
        return None

def check_and_delete_misclassified_files(dataset_path, model, le):
    # Gather all audio files and their labels
    file_paths = []
    labels = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                file_path = os.path.join(root, file)
                # Get the relative path from dataset root to the parent directory of the file
                relative_dir = os.path.relpath(root, dataset_path)
                # Use this relative directory path as the label
                label = relative_dir.replace(os.sep, '_')  # Replace os.sep with '_'
                file_paths.append((file_path, label))
                labels.append(label)
    
    print(f"Total files to check: {len(file_paths)}")
    
    # Process files with progress bar
    misclassified_files = []
    for file_path, true_label in tqdm(file_paths, desc="Checking files", unit="file"):
        features = extract_features_from_file(file_path)
        if features is not None:
            features = np.expand_dims(features, axis=0)  # Add batch dimension
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction, axis=1)
            predicted_label = le.inverse_transform(predicted_class)[0]
            if predicted_label != true_label:
                misclassified_files.append((file_path, true_label, predicted_label))
        else:
            print(f"Skipping file due to feature extraction failure: {file_path}")
            continue
    
    print(f"Total misclassified files: {len(misclassified_files)}")
    
    # Handle misclassified files
    if DELETE_FILES:
        for file_path, true_label, predicted_label in misclassified_files:
            try:
                os.remove(file_path)
                print(f"Deleted misclassified file: {file_path}")
            except Exception as e:
                print(f"Error deleting file '{file_path}': {e}")
    else:
        # Move misclassified files to a separate folder for review
        misclassified_dir = os.path.join(dataset_path, 'misclassified')
        os.makedirs(misclassified_dir, exist_ok=True)
        for file_path, true_label, predicted_label in misclassified_files:
            # Create a subdirectory for the true label
            dest_dir = os.path.join(misclassified_dir, true_label)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, os.path.basename(file_path))
            try:
                shutil.move(file_path, dest_path)
                print(f"Moved misclassified file to: {dest_path}")
            except Exception as e:
                print(f"Error moving file '{file_path}': {e}")
    
    print("Processing complete.")

def main():
    global TRAINING_DATASET_PATH
    if not os.path.isdir(TRAINING_DATASET_PATH):
        print(f"Error: Training dataset path '{TRAINING_DATASET_PATH}' does not exist or is not a directory.")
        sys.exit(1)
    
    # Confirm action with the user
    action = "delete" if DELETE_FILES else "move"
    print(f"This script will {action} misclassified files from the training dataset at '{TRAINING_DATASET_PATH}'.")
    confirm = input("Have you backed up your data? Type 'yes' to proceed: ")
    if confirm.lower() != 'yes':
        print("Operation cancelled.")
        sys.exit(0)
    
    check_and_delete_misclassified_files(TRAINING_DATASET_PATH, model, le)

if __name__ == "__main__":
    main()