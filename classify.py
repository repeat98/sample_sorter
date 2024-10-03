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
import pickle

# Setup logging
logging.basicConfig(
    filename='organize_samples.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Constants for model and feature extraction
SAMPLE_RATE = 22050
DURATION = 5
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 20
CONFIDENCE_THRESHOLD = 0.8  # Adjust as needed

# Load the trained model and label encoder
model = tf.keras.models.load_model("audio_classification_model.keras")
with open("label_encoder.pkl", "rb") as le_file:
    le_filtered = pickle.load(le_file)
model_classes = le_filtered.classes_

# Define substrings for each class. These should be lowercase for case-insensitive matching.
filename_substrings = {
    # Drums
    'Clap': ['clap', 'snap', 'handclap', 'hand clap'],
    'Cymbal': ['cymbal', 'china', 'ride cymbal', 'splash cymbal'],
    'Hihat': ['hihat', 'hi-hat', 'hat', 'closed hat', 'open hat'],
    'Kick': ['kick', 'bass drum', 'bd', 'kick drum'],
    'Percussion': ['perc', 'percussion', 'shaker', 'tambourine', 'conga', 'bongo', 'ashiko', 'caxixi', 'dumbek', 'shekere', 'talking drum', 'djuns djuns'],
    'Snare': ['snare', 'snr', 'rimshot', 'brushes'],
    'Tom': ['tom', 'toms', 'floor tom', 'rack tom'],
    'Breakbeat': ['breakbeat', 'break', 'drum break'],

    # Sounds (Tonal)
    'Bass': ['bass', 'bassline', 'sub bass'],
    'Chords': ['chord', 'chords', 'keyboard', 'piano', 'synth chords'],
    'Melody': ['melody', 'lead', 'lead synth', 'lead guitar'],
    'Voice': ['voice', 'vocals', 'vocal', 'singing'],
    'Brass': ['brass', 'trumpet', 'saxophone', 'trombone'],
    'Chord': ['chord', 'chords', 'harmonic', 'harmony'],
    'Guitar & Plucked': ['guitar', 'pluck', 'plucked', 'acoustic guitar', 'electric guitar'],
    'Lead': ['lead', 'lead part', 'lead instrument'],
    'Mallets': ['mallet', 'mallets', 'xylophone', 'marimba'],
    'Strings': ['string', 'strings', 'violin', 'cello', 'orchestra'],
    'Woodwind': ['woodwind', 'flute', 'sax', 'clarinet', 'oboe'],
    'Synth Stabs': ['synth stabs'],
    # Sounds (Non-Tonal)
    'FX': ['fx', 'effects', 'sfx', 'reverb', 'delay', 'echo'],
    'Ambience & FX': ['ambience', 'ambient', 'fx', 'atmosphere', 'background noise'],
    # Additional classes can be added here
}

# Extended class mappings with combined keys
LOOP_MAPPING = {
    # Loops/Drums
    'Loops_Breakbeat': 'Loops/Drums/Breakbeat',
    'Loops_Hihat': 'Loops/Drums/Hihat',
    'Loops_Percussion': 'Loops/Drums/Percussion',

    # Loops/Sounds (Tonal)
    'Loops_Bass': 'Loops/Sounds/Bass',
    'Loops_Chords': 'Loops/Sounds/Chords',
    'Loops_Melody': 'Loops/Sounds/Melody',
    'Loops_Voice': 'Loops/Sounds/Voice',
    'Loops_Brass': 'Loops/Sounds/Brass',
    'Loops_Chord': 'Loops/Sounds/Chord',
    'Loops_Guitar & Plucked': 'Loops/Sounds/Guitar & Plucked',
    'Loops_Lead': 'Loops/Sounds/Lead',
    'Loops_Mallets': 'Loops/Sounds/Mallets',
    'Loops_Strings': 'Loops/Sounds/Strings',
    'Loops_Woodwind': 'Loops/Sounds/Woodwind',
    'Loops_Synth Stabs': 'Loops/Sounds/Synth Stabs',

    # Loops/Sounds (Non-Tonal)
    'Loops_FX': 'Loops/Sounds/FX',
    # Add more loop-specific categories if needed
}

ONESHOT_MAPPING = {
    # Oneshots/Drums
    'Oneshots_Clap': 'Oneshots/Drums/Clap',
    'Oneshots_Cymbal': 'Oneshots/Drums/Cymbal',
    'Oneshots_Hihat': 'Oneshots/Drums/Hihat',
    'Oneshots_Kick': 'Oneshots/Drums/Kick',
    'Oneshots_Percussion': 'Oneshots/Drums/Percussion',
    'Oneshots_Snare': 'Oneshots/Drums/Snare',
    'Oneshots_Tom': 'Oneshots/Drums/Tom',

    # Oneshots/Sounds
    'Oneshots_Ambience & FX': 'Oneshots/Sounds/Ambience & FX',
    'Oneshots_Bass': 'Oneshots/Sounds/Bass',
    'Oneshots_Brass': 'Oneshots/Sounds/Brass',
    'Oneshots_Chord': 'Oneshots/Sounds/Chord',
    'Oneshots_Guitar & Plucked': 'Oneshots/Sounds/Guitar & Plucked',
    'Oneshots_Lead': 'Oneshots/Sounds/Lead',
    'Oneshots_Mallets': 'Oneshots/Sounds/Mallets',
    'Oneshots_Strings': 'Oneshots/Sounds/Strings',
    'Oneshots_Synth Stabs': 'Oneshots/Sounds/Synth Stabs',
    'Oneshots_Voice': 'Oneshots/Sounds/Voice',
    'Oneshots_Woodwind': 'Oneshots/Sounds/Woodwind',
    # Add more oneshot-specific categories if needed
}

# Tonal categories for which key detection should be performed
TONAL_CATEGORIES = [
    'Bass',
    'Chords',
    'Melody',
    'Voice',
    'Brass',
    'Chord',
    'Guitar & Plucked',
    'Lead',
    'Mallets',
    'Strings',
    'Woodwind',
    'Synth Stabs',
]

# Combined list of all possible substrings for categorization
ALL_SUBSTRINGS = list(set(list(filename_substrings.keys())))

# Supported audio file extensions
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.aiff', '.flac', '.ogg', '.m4a')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Organize audio samples into categorized folders based on loops and oneshots, and append Key and BPM information to filenames.')
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
        'Loops/Sounds/Melody',
        'Loops/Sounds/Brass',
        'Loops/Sounds/Chord',
        'Loops/Sounds/Guitar & Plucked',
        'Loops/Sounds/Lead',
        'Loops/Sounds/Mallets',
        'Loops/Sounds/Strings',
        'Loops/Sounds/Woodwind',
        'Loops/Sounds/Synth Stabs',

        # Oneshots/Drums
        'Oneshots/Drums/Clap',
        'Oneshots/Drums/Cymbal',
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

def detect_key(file_path):
    """
    Detects the musical key and scale of an audio file using Essentia's KeyExtractor.

    Parameters:
    - file_path (str): Path to the audio file.

    Returns:
    - key_scale (str): Detected key and scale (e.g., 'C Major'), or 'Unknown' if detection fails.
    """
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

        # Check if num_bars is close to an integer power of 2 (1, 2, 4, 8, 16, ...)
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
    Categorizes the file based on substrings in the filename and its directory names.

    Returns the destination subfolder path.
    """
    filename = os.path.basename(file_path).lower()
    # Normalize filename: remove '&', spaces, hyphens, and underscores
    filename_clean = filename.replace('&', '').replace(' ', '').replace('-', '').replace('_', '')

    # Extract directory names
    dir_names = []
    parent_dir = os.path.dirname(file_path)
    while parent_dir and parent_dir != os.path.dirname(parent_dir):
        dir_names.append(os.path.basename(parent_dir).lower())
        parent_dir = os.path.dirname(parent_dir)

    # Combine all directory names into a single string for easier matching
    dir_names_combined = ' '.join(dir_names)

    # First, check directory names for substrings
    for substring in ALL_SUBSTRINGS:
        substring_clean = substring.lower().replace('&', '').replace(' ', '').replace('-', '').replace('_', '')
        if substring_clean in dir_names_combined:
            if is_loop_flag:
                key = f"Loops_{substring}"
                if key in LOOP_MAPPING:
                    return LOOP_MAPPING[key]
            else:
                key = f"Oneshots_{substring}"
                if key in ONESHOT_MAPPING:
                    return ONESHOT_MAPPING[key]

    # If no match in directories, check the filename
    for substring in ALL_SUBSTRINGS:
        substring_clean = substring.lower().replace('&', '').replace(' ', '').replace('-', '').replace('_', '')
        if substring_clean in filename_clean:
            if is_loop_flag:
                key = f"Loops_{substring}"
                if key in LOOP_MAPPING:
                    return LOOP_MAPPING[key]
            else:
                key = f"Oneshots_{substring}"
                if key in ONESHOT_MAPPING:
                    return ONESHOT_MAPPING[key]

    return None  # If no category matches

def extract_features_from_file(file_path):
    try:
        # Load audio file
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        # Ensure signal length is SAMPLES_PER_TRACK
        if len(signal) > SAMPLES_PER_TRACK:
            signal = signal[:SAMPLES_PER_TRACK]
        else:
            pad_width = SAMPLES_PER_TRACK - len(signal)
            signal = np.pad(signal, (0, pad_width), 'constant')

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                    n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mfcc = librosa.power_to_db(mfcc, ref=np.max)

        # Add channel dimension
        mfcc = mfcc[..., np.newaxis]

        # Expand dimensions to match model input
        mfcc = np.expand_dims(mfcc, axis=0)

        return mfcc
    except Exception as e:
        logging.error(f"Error extracting features from '{file_path}': {e}")
        return None

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

    # Process files with progress bar
    for file_path in tqdm(audio_files, desc="Organizing Samples", unit="file"):
        loop_flag, bpm = is_loop(file_path)
        category_path_substring = categorize_file(file_path, loop_flag)

        # Model prediction
        features = extract_features_from_file(file_path)
        if features is not None:
            model_prediction = model.predict(features)
            predicted_class_index = np.argmax(model_prediction)
            predicted_class = model_classes[predicted_class_index]
            confidence = model_prediction[0][predicted_class_index]
            logging.info(f"Model predicted '{predicted_class}' with confidence {confidence} for file '{file_path}'")

            # Map predicted_class to category path
            if loop_flag:
                if predicted_class in LOOP_MAPPING:
                    category_path_model = LOOP_MAPPING[predicted_class]
                else:
                    category_path_model = None
            else:
                if predicted_class in ONESHOT_MAPPING:
                    category_path_model = ONESHOT_MAPPING[predicted_class]
                else:
                    category_path_model = None
        else:
            # If feature extraction failed
            category_path_model = None
            confidence = 0.0

        # Decision logic
        if category_path_substring == category_path_model:
            category_path = category_path_substring  # Both methods agree
        elif category_path_substring is not None and category_path_model is not None:
            # They disagree
            if confidence >= CONFIDENCE_THRESHOLD:
                category_path = category_path_model  # Trust the model
                logging.info(f"Disagreement on category for '{file_path}'. Using model prediction '{predicted_class}' with confidence {confidence}.")
            else:
                category_path = category_path_substring  # Trust substring matching
                logging.info(f"Disagreement on category for '{file_path}'. Using substring matching result.")
        elif category_path_substring is None and category_path_model is not None:
            if confidence >= CONFIDENCE_THRESHOLD:
                category_path = category_path_model  # Use the model's prediction
                logging.info(f"Substring matching failed for '{file_path}'. Using model prediction '{predicted_class}' with confidence {confidence}.")
            else:
                category_path = None  # Cannot categorize
                logging.warning(f"Cannot categorize '{file_path}'. Model confidence too low ({confidence}).")
        elif category_path_substring is not None and category_path_model is None:
            # Model could not predict category
            category_path = category_path_substring  # Use substring matching
            logging.info(f"Model could not predict category for '{file_path}'. Using substring matching result.")
        else:
            # Both methods failed
            category_path = None
            logging.warning(f"Cannot categorize '{file_path}' by either method.")

        if category_path:
            destination_dir = os.path.join(output_folder, category_path)
            filename = os.path.basename(file_path)
            name, ext = os.path.splitext(filename)

            # Extract category name from category_path
            category_name = category_path.split('/')[-1]

            # Prepare new filename
            new_filename = name
            append_info = []

            # Only perform key detection for tonal samples
            if category_name in TONAL_CATEGORIES:
                key = detect_key(file_path)
                if key != 'Unknown':
                    append_info.append(key)
            else:
                key = 'Unknown'  # Not applicable

            # Only append BPM if it's a loop and BPM is valid
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
        else:
            # File does not have a matching substring or model prediction; skip copying or handle accordingly
            logging.warning(f"No matching category found for '{file_path}'. Skipping.")
            continue

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