import os
import shutil
import argparse
import sys
import math
from tqdm import tqdm
import essentia.standard as ess
import librosa
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Setup logging
logging.basicConfig(
    filename='organize_samples.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load the pre-trained model
model = load_model('audio_classification_model.keras')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Define tonal categories
TONAL_CATEGORIES = [
    'Bass', 'Chords', 'Melody', 'Voice', 'Brass', 'Chord',
    'Guitar & Plucked', 'Lead', 'Mallets', 'Strings', 'Woodwind', 'Synth Stabs'
]

# Extended class mappings
LOOP_MAPPING = {
    # Loops/Drums
    'Breakbeat': 'Loops/Drums/Breakbeat',
    'Hihat': 'Loops/Drums/Hihat',
    'Percussion': 'Loops/Drums/Percussion',

    # Loops/Sounds
    'Bass': 'Loops/Sounds/Bass',
    'Chords': 'Loops/Sounds/Chords',
    'FX': 'Loops/Sounds/FX',
    'Synth': 'Loops/Sounds/Synth',
    'Voice': 'Loops/Sounds/Voice',
}

ONESHOT_MAPPING = {
    # Oneshots/Drums
    'Clap': 'Oneshots/Drums/Clap',
    'Cymbal': 'Oneshots/Drums/Cymbal',
    'Hand Percussion': 'Oneshots/Drums/Hand Percussion',
    'Hihat': 'Oneshots/Drums/Hihat',
    'Kick': 'Oneshots/Drums/Kick',
    'Percussion': 'Oneshots/Drums/Percussion',
    'Snare': 'Oneshots/Drums/Snare',
    'Tom': 'Oneshots/Drums/Tom',

    # Oneshots/Sounds
    'Ambience & FX': 'Oneshots/Sounds/Ambience & FX',
    'Bass': 'Oneshots/Sounds/Bass',
    'Brass': 'Oneshots/Sounds/Brass',
    'Chord': 'Oneshots/Sounds/Chord',
    'Guitar & Plucked': 'Oneshots/Sounds/Guitar & Plucked',
    'Lead': 'Oneshots/Sounds/Lead',
    'Mallets': 'Oneshots/Sounds/Mallets',
    'Strings': 'Oneshots/Sounds/Strings',
    'Synth Stabs': 'Oneshots/Sounds/Synth Stabs',
    'Voice': 'Oneshots/Sounds/Voice',
    'Woodwind': 'Oneshots/Sounds/Woodwind',
}

# Combined list of all possible categories
ALL_CATEGORIES = list(set(list(LOOP_MAPPING.keys()) + list(ONESHOT_MAPPING.keys())))

# Supported audio file extensions
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.aiff', '.flac', '.ogg', '.m4a')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Organize audio samples into categorized folders using a pre-trained model, and append Key and BPM information to filenames.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing audio samples.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where organized samples will be stored.')
    args = parser.parse_args()
    return args.input_folder, args.output_folder

def create_directory_structure(base_path):
    """
    Creates the required directory structure.
    """
    structure = [
        # Loops/Drums
        'Loops/Drums/Breakbeat',
        'Loops/Drums/Hihat',
        'Loops/Drums/Percussion',

        # Loops/Sounds
        'Loops/Sounds/Bass',
        'Loops/Sounds/Chords',
        'Loops/Sounds/FX',
        'Loops/Sounds/Synth',
        'Loops/Sounds/Voice',

        # Oneshots/Drums
        'Oneshots/Drums/Clap',
        'Oneshots/Drums/Cymbal',
        'Oneshots/Drums/Hand Percussion',
        'Oneshots/Drums/Hihat',
        'Oneshots/Drums/Kick',
        'Oneshots/Drums/Percussion',
        'Oneshots/Drums/Snare',
        'Oneshots/Drums/Tom',

        # Oneshots/Sounds
        'Oneshots/Sounds/Ambience & FX',
        'Oneshots/Sounds/Bass',
        'Oneshots/Sounds/Brass',
        'Oneshots/Sounds/Chord',
        'Oneshots/Sounds/Guitar & Plucked',
        'Oneshots/Sounds/Lead',
        'Oneshots/Sounds/Mallets',
        'Oneshots/Sounds/Strings',
        'Oneshots/Sounds/Synth Stabs',
        'Oneshots/Sounds/Voice',
        'Oneshots/Sounds/Woodwind',
    ]

    for folder in structure:
        dir_path = os.path.join(base_path, folder)
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Ensured directory exists: {dir_path}")

    # Create 'Unknown' folder
    unknown_dir = os.path.join(base_path, 'Unknown')
    os.makedirs(unknown_dir, exist_ok=True)
    logging.info(f"Ensured directory exists: {unknown_dir}")

def extract_features(file_path, sr=22050, n_mfcc=40, max_len=173):
    """
    Extracts MFCCs from an audio file.

    Parameters:
    - file_path (str): Path to the audio file.
    - sr (int): Sampling rate for audio loading.
    - n_mfcc (int): Number of MFCCs to extract.
    - max_len (int): Maximum length of the MFCC feature vector.

    Returns:
    - mfccs (numpy.ndarray): Extracted MFCC features.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        # Pad or truncate the MFCCs to a fixed length
        if mfccs.shape[1] < max_len:
            pad_width = max_len - mfccs.shape[1]
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
        return mfccs
    except Exception as e:
        logging.error(f"Error extracting features from '{file_path}': {e}")
        return None

def detect_key(file_path, category):
    """
    Detects the musical key of an audio file using Essentia's KeyExtractor if the sample is tonal.

    Parameters:
    - file_path (str): Path to the audio file.
    - category (str): Predicted category of the sample.

    Returns:
    - key_scale (str): Detected key and scale (e.g., 'C Major'), or 'Unknown' if detection fails or not tonal.
    """
    if category not in TONAL_CATEGORIES:
        return 'Unknown'

    try:
        # Instantiate the key extractor
        key_extractor = ess.KeyExtractor()

        # Load audio at 16kHz mono
        loader = ess.MonoLoader(filename=file_path, sampleRate=16000)
        audio = loader()

        # Extract the key, scale, and strength
        key, scale, strength = key_extractor(audio)

        # Check detection strength
        if strength < 0.1:
            logging.warning(f"Low confidence ({strength}) in key detection for '{file_path}'.")
            return 'Unknown'

        # Format the key and scale
        key_scale = f"{key} {scale.capitalize()}"

        return key_scale
    except Exception as e:
        logging.error(f"Error extracting key for '{file_path}': {e}")
        return 'Unknown'

def is_loop(file_path, bpm_threshold=30, beats_per_bar=4, tolerance=0.05):
    """
    Determines if a file is a loop based on BPM and duration.

    Parameters:
    - file_path (str): Path to the audio file.
    - bpm_threshold (float): Minimum BPM to consider as rhythmic.
    - beats_per_bar (int): Number of beats in a musical bar.
    - tolerance (float): Acceptable deviation when checking bar multiples.

    Returns:
    - bool: True if loop, False otherwise.
    - bpm (float): Estimated BPM of the loop.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)

        if tempo < bpm_threshold:
            logging.info(f"File '{file_path}' tempo {tempo} BPM below threshold {bpm_threshold} BPM.")
            return False, float(tempo)  # Too slow to be considered rhythmic

        # Calculate duration per beat and per bar
        beat_duration = 60.0 / tempo
        bar_duration = beats_per_bar * beat_duration

        # Number of bars (could be fractional)
        num_bars = duration / bar_duration

        # Check if num_bars is close to an integer power of 2
        if num_bars < 0.5:
            logging.info(f"File '{file_path}' has {num_bars} bars, which is too short to be a loop.")
            return False, float(tempo)  # Too short to be a loop

        nearest_power = 2 ** round(math.log(num_bars, 2))
        if abs(num_bars - nearest_power) / nearest_power <= tolerance:
            logging.info(f"File '{file_path}' identified as loop with {tempo} BPM and approximately {nearest_power} bars.")
            return True, float(tempo)
        else:
            logging.info(f"File '{file_path}' not a loop. BPM: {tempo}, Bars: {num_bars}, Nearest Power: {nearest_power}.")
            return False, float(tempo)
    except Exception as e:
        logging.error(f"Error processing '{file_path}': {e}")
        return False, 0.0

def categorize_file(file_path, is_loop_flag):
    """
    Categorizes the file using the pre-trained model.

    Returns the destination subfolder path and predicted category.
    """
    # Extract features from the audio file
    features = extract_features(file_path)
    if features is None:
        return 'Unknown', 'Unknown'  # Could not extract features

    # Reshape features to match model input
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)  # Add channel dimension if needed

    # Predict using the model
    try:
        predictions = model.predict(features)
        predicted_index = np.argmax(predictions, axis=1)
        predicted_label = label_encoder.inverse_transform(predicted_index)[0]
    except Exception as e:
        logging.error(f"Error predicting category for '{file_path}': {e}")
        return 'Unknown', 'Unknown'

    # Map predicted label to destination path
    if is_loop_flag:
        if predicted_label in LOOP_MAPPING:
            return LOOP_MAPPING[predicted_label], predicted_label
    else:
        if predicted_label in ONESHOT_MAPPING:
            return ONESHOT_MAPPING[predicted_label], predicted_label

    return 'Unknown', predicted_label  # If the predicted label is not in our mappings

def organize_samples(input_folder, output_folder):
    """
    Organizes the audio samples from the input folder into the structured output folder,
    and appends Key and BPM information to filenames.
    """
    # Create directory structure
    create_directory_structure(output_folder)

    # Gather all audio files
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                audio_files.append(os.path.join(root, file))

    total_files = len(audio_files)
    logging.info(f"Starting organization of {total_files} files from '{input_folder}' to '{output_folder}'.")

    # Create 'Unknown' directory
    unknown_dir = os.path.join(output_folder, 'Unknown')
    os.makedirs(unknown_dir, exist_ok=True)

    # Process files with progress bar
    for file_path in tqdm(audio_files, desc="Organizing Samples", unit="file"):
        loop_flag, bpm = is_loop(file_path)
        category_path, predicted_category = categorize_file(file_path, loop_flag)

        # Determine if the sample is tonal
        is_tonal = predicted_category in TONAL_CATEGORIES
        key = detect_key(file_path, predicted_category)

        if category_path == 'Unknown':
            destination_dir = unknown_dir
        else:
            destination_dir = os.path.join(output_folder, category_path)

        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)

        # Prepare new filename
        new_filename = name
        append_info = []

        if key != 'Unknown':
            append_info.append(key)

        if loop_flag and bpm > 0:
            append_info.append(f"{int(round(bpm))}BPM")

        if append_info:
            new_filename += '_' + '_'.join(append_info)

        new_filename += ext

        destination_path = os.path.join(destination_dir, new_filename)

        try:
            shutil.copy2(file_path, destination_path)
            logging.info(f"Copied '{filename}' to '{destination_dir}' as '{new_filename}'.")
        except Exception as e:
            logging.error(f"Failed to copy '{file_path}': {e}")

def main():
    input_folder, output_folder = parse_arguments()

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist or is not a directory.")
        logging.error(f"Input folder '{input_folder}' does not exist or is not a directory.")
        sys.exit(1)

    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder, exist_ok=True)
            logging.info(f"Created output folder '{output_folder}'.")
        except Exception as e:
            print(f"Error creating output folder '{output_folder}': {e}")
            logging.error(f"Error creating output folder '{output_folder}': {e}")
            sys.exit(1)

    organize_samples(input_folder, output_folder)
    print("Organization complete.")
    logging.info("Organization complete.")

if __name__ == "__main__":
    main()