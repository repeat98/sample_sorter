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
import soundfile as sf

# Setup logging
logging.basicConfig(
    filename='organize_samples.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# --------------------------- Parameters ---------------------------

# Processing sample rates
PROC_SAMPLE_RATE = 22050  # For loop detection and feature extraction
KEY_SAMPLE_RATE = 16000   # For key detection

DURATION = 5  # Duration to which all audio files will be truncated or padded (in seconds)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 20  # Should match the value in training script
SAMPLES_PER_TRACK = PROC_SAMPLE_RATE * DURATION

# Tonal categories for which key detection should be performed
TONAL_CATEGORIES = {
    'Bass',
    'Chords',
    'Synth',
    'Voice',
    'Brass',
    'Guitar & Plucked',
    'Lead',
    'Mallets',
    'Strings',
    'Woodwind',
}

# Supported audio file extensions
AUDIO_EXTENSIONS = (
    '.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif',
    '.aac', '.wma', '.m4a', '.alac', '.opus'
)

# ------------------------ Load Model and Encoder ------------------------

# Load the trained model and label encoder
try:
    model = tf.keras.models.load_model('model/audio_classification_model.keras')
    with open('model/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    logging.info("Successfully loaded trained model and label encoder.")
except Exception as e:
    logging.error(f"Error loading model or label encoder: {e}")
    sys.exit(1)

# Get the list of labels from the label encoder
model_labels = le.classes_

# ----------------------- Substring Definitions -----------------------

# Define substrings for each class. These should be lowercase for case-insensitive matching.
filename_substrings = {
    # Loops/Drums
    'Full Drum Loop': ['full drum loop', 'drum loop', 'drums', 'drum', 'breakbeat', 'break', 'drum break'],
    'Hihat': ['hihat loop', 'hi-hat loop', 'hat loop', 'closed hat loop', 'open hat loop'],
    'Percussion': ['perc loop', 'percussion loop', 'shaker loop', 'tambourine loop', 'conga loop', 'bongo loop'],

    # Loops/Sounds
    'Bass': ['bass loop', 'bassline loop', 'sub bass loop'],
    'Chords': ['chord loop', 'chords loop', 'keyboard loop', 'piano loop', 'synth chords loop', 'stab', 'pad'],
    'Synth': ['synth loop', 'synthesizer loop', 'synth lead loop', 'synth pad loop', 'syn'],
    'Voice': ['vocal loop', 'vocals loop', 'voice loop', 'singing loop', 'vox'],

    # Oneshots/Drums
    'Clap': ['clap', 'snap', 'handclap', 'hand clap'],
    'Cymbal': ['cymbal', 'china', 'ride cymbal', 'splash cymbal', 'ride'],
    'Hand Percussion': ['hand percussion', 'hand drums', 'conga', 'bongo', 'djembe', 'tabla', 'shaker', 'tambourine', 'cowbell'],
    'Hihat': ['hihat', 'hi-hat', 'hat', 'closed hat', 'open hat'],
    'Kick': ['kick', 'bass drum', 'bd', 'kick drum', 'bassdrum'],
    'Percussion': ['perc', 'percussion', 'shaker', 'tambourine', 'cowbell', 'shaker', 'tambourine', ],
    'Snare': ['snare', 'snr', 'rimshot', 'brushes'],
    'Tom': ['tom', 'toms', 'floor tom', 'rack tom'],

    # Oneshots/Sounds
    'Ambience & FX': ['ambience', 'ambient', 'fx', 'effects', 'sfx', 'reverb', 'delay', 'echo', 'atmosphere', 'background noise'],
    'Bass': ['bass', 'bassline', 'sub bass'],
    'Brass': ['brass', 'trumpet', 'saxophone', 'trombone'],
    'Chords': ['chord', 'chords', 'keyboard', 'piano', 'synth chords', 'pad', 'stab'],
    'Guitar & Plucked': ['guitar', 'pluck', 'plucked', 'acoustic guitar', 'electric guitar', 'harp', 'banjo', 'ukulele', 'mandolin'],
    'Lead': ['lead', 'lead synth', 'lead guitar', 'melody'],
    'Mallets': ['mallet', 'mallets', 'xylophone', 'marimba', 'vibraphone', 'glockenspiel'],
    'Strings': ['string', 'strings', 'violin', 'cello', 'orchestra'],
    'Voice': ['voice', 'vocals', 'vocal', 'singing', 'vox'],
    'Woodwind': ['woodwind', 'flute', 'sax', 'clarinet', 'oboe', 'bassoon','horn'],
    # Additional categories can be added here
}

# Extended class mappings
LOOP_MAPPING = {
    # Loops/Drums
    'Full Drum Loop': 'Loops/Drums/Full Drum Loop',
    'Hihat': 'Loops/Drums/Hihat',
    'Percussion': 'Loops/Drums/Percussion',

    # Loops/Sounds
    'Bass': 'Loops/Sounds/Bass',
    'Chords': 'Loops/Sounds/Chords',
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
    'Chords': 'Oneshots/Sounds/Chords',
    'Guitar & Plucked': 'Oneshots/Sounds/Guitar & Plucked',
    'Lead': 'Oneshots/Sounds/Lead',
    'Mallets': 'Oneshots/Sounds/Mallets',
    'Strings': 'Oneshots/Sounds/Strings',
    'Voice': 'Oneshots/Sounds/Voice',
    'Woodwind': 'Oneshots/Sounds/Woodwind',
}

# Combined list of all possible substrings for categorization
ALL_SUBSTRINGS = list(set(list(LOOP_MAPPING.keys()) + list(ONESHOT_MAPPING.keys())))

# Preprocess substrings for faster access
preprocessed_substrings = {
    category: [substr.lower().replace('&', '').replace(' ', '').replace('-', '').replace('_', '') 
               for substr in substrings]
    for category, substrings in filename_substrings.items()
}

# ------------------------ Functions ------------------------

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
        'Loops/Drums/Full Drum Loop',
        'Loops/Drums/Hihat',
        'Loops/Drums/Percussion',
        # Loops/Sounds
        'Loops/Sounds/Bass',
        'Loops/Sounds/Chords',
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
        'Oneshots/Sounds/Chords',
        'Oneshots/Sounds/Guitar & Plucked',
        'Oneshots/Sounds/Lead',
        'Oneshots/Sounds/Mallets',
        'Oneshots/Sounds/Strings',
        'Oneshots/Sounds/Voice',
        'Oneshots/Sounds/Woodwind',
    ]

    for folder in structure:
        dir_path = os.path.join(base_path, folder)
        os.makedirs(dir_path, exist_ok=True)
        # Reduced logging to DEBUG level for directory creation
        logging.debug(f"Ensured directory exists: {dir_path}")

    return set(structure)  # Return as a set for faster lookup

def detect_key(audio_data, sample_rate, key_extractor):
    """
    Detects the musical key and scale of an audio signal using Essentia's KeyExtractor.

    Parameters:
    - audio_data (np.ndarray): Audio time series.
    - sample_rate (int): Sample rate of the audio.
    - key_extractor (ess.KeyExtractor): Pre-instantiated KeyExtractor object.

    Returns:
    - key_scale (str): Detected key and scale (e.g., 'C Major'), or 'Unknown' if detection fails.
    """
    try:
        # Extract the key, scale, and strength
        key, scale, strength = key_extractor(audio_data)

        # Check detection strength
        if strength < 0.1:
            logging.warning(f"Low confidence ({strength}) in key detection.")
            return 'Unknown'

        # Format the key and scale
        key_scale = f"{key} {scale.capitalize()}"

        return key_scale
    except Exception as e:
        logging.error(f"Error extracting key: {e}")
        return 'Unknown'

def is_loop(y, sr, bpm_threshold=30, beats_per_bar=4, tolerance=0.05, min_transients=2, beat_track_kwargs=None, onset_detect_kwargs=None):
    """
    Determines if a file is a loop based on BPM, duration, transients, and rhythmic consistency.

    Parameters:
    - y (np.ndarray): Audio time series.
    - sr (int): Sample rate of the audio.
    - bpm_threshold (float): Minimum BPM to consider as rhythmic.
    - beats_per_bar (int): Number of beats in a musical bar.
    - tolerance (float): Acceptable deviation when checking bar multiples.
    - min_transients (int): Minimum number of transients to consider as rhythmic.
    - beat_track_kwargs (dict): Additional keyword arguments for beat_track.
    - onset_detect_kwargs (dict): Additional keyword arguments for onset_detect.

    Returns:
    - bool: True if loop, False otherwise.
    - float: Estimated BPM of the loop.
    """
    try:
        # Normalize audio to have maximum amplitude of 1.0
        y = librosa.util.normalize(y)

        # Estimate tempo (BPM) and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, **(beat_track_kwargs or {}))

        # Ensure tempo is a scalar float
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo.item())
        else:
            tempo = float(tempo)

        # Ensure tempo is above the threshold
        if tempo < bpm_threshold:
            logging.debug(f"Tempo {tempo:.2f} BPM below threshold {bpm_threshold} BPM.")
            return False, tempo  # Too slow to be considered rhythmic

        # Detect onsets (transients) to ensure multiple transients exist
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, **(onset_detect_kwargs or {}))
        num_transients = len(onset_frames)

        if num_transients < min_transients:
            logging.debug(f"Only {num_transients} transients detected, fewer than the minimum required {min_transients}.")
            return False, tempo  # Not enough transients to be considered rhythmic

        # Calculate duration in seconds
        duration = librosa.get_duration(y=y, sr=sr)

        # Calculate duration per beat and per bar
        beat_duration = 60.0 / tempo
        bar_duration = beats_per_bar * beat_duration

        # Number of bars (could be fractional)
        num_bars = duration / bar_duration

        if num_bars < 1.0:
            logging.debug(f"Duration {duration:.2f}s is too short to be a loop.")
            return False, tempo  # Too short to be a loop

        # Check if num_bars is close to an integer number of bars within the tolerance
        nearest_int = round(num_bars)
        if nearest_int == 0:
            return False, tempo
        if abs(num_bars - nearest_int) / nearest_int <= tolerance:
            logging.debug(f"Identified as loop with {tempo:.2f} BPM and approximately {nearest_int} bars.")
            return True, tempo
        else:
            logging.debug(f"Not a loop. BPM: {tempo:.2f}, Bars: {num_bars:.2f}, Nearest Int: {nearest_int}.")
            return False, tempo

    except Exception as e:
        logging.error(f"Error in loop detection: {e}")
        return False, 0.0

def extract_features(y):
    """
    Extracts MFCC features from audio time series.

    Parameters:
    - y (np.ndarray): Audio time series.

    Returns:
    - mfcc (np.ndarray): Extracted MFCC features.
    """
    try:
        # Ensure y has the same length as during training (padded or truncated)
        max_len = SAMPLES_PER_TRACK
        if len(y) > max_len:
            y = y[:max_len]
        else:
            pad_width = max_len - len(y)
            y = np.pad(y, (0, pad_width), 'constant')

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=PROC_SAMPLE_RATE, n_mfcc=N_MFCC, n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mfcc = librosa.power_to_db(mfcc, ref=np.max)
        mfcc = mfcc[..., np.newaxis]  # Add channel dimension
        return mfcc
    except Exception as e:
        logging.error(f"Error extracting MFCC features: {e}")
        return None

def label_to_category_path(label):
    category_path = label.replace('_', os.sep)
    return category_path

def categorize_file(filename, dir_names_combined, is_loop_flag):
    """
    Categorizes the file based on substrings in the filename and its directory names.

    Returns the destination subfolder path or None if no match is found.
    """
    filename_clean = filename.lower().replace('&', '').replace(' ', '').replace('-', '').replace('_', '')

    # Check directory names first
    for category, substrings in preprocessed_substrings.items():
        for substr in substrings:
            if substr in dir_names_combined:
                if is_loop_flag and category in LOOP_MAPPING:
                    return LOOP_MAPPING[category]
                elif not is_loop_flag and category in ONESHOT_MAPPING:
                    return ONESHOT_MAPPING[category]

    # If no match in directories, check the filename
    for category, substrings in preprocessed_substrings.items():
        for substr in substrings:
            if substr in filename_clean:
                if is_loop_flag and category in LOOP_MAPPING:
                    return LOOP_MAPPING[category]
                elif not is_loop_flag and category in ONESHOT_MAPPING:
                    return ONESHOT_MAPPING[category]

    return None  # If no category matches

def organize_samples(input_folder, output_folder, model, le, key_extractor):
    """
    Organizes the audio samples from the input folder into the structured output folder,
    normalizes the audio, and appends Key and BPM information to filenames.
    """
    # Create directory structure and get allowed categories
    structure = create_directory_structure(output_folder)
    allowed_categories = structure  # Already a set

    # Gather all audio files
    audio_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                audio_files.append(os.path.join(root, file))

    total_files = len(audio_files)
    logging.info(f"Starting organization of {total_files} files from '{input_folder}' to '{output_folder}'.")

    # Predefine librosa functions' kwargs to avoid repetitive dictionary creation
    beat_track_kwargs = {'units': 'frames'}
    onset_detect_kwargs = {'units': 'frames'}

    # Process files with progress bar
    for file_path in tqdm(audio_files, desc="Organizing Samples", unit="file"):
        try:
            # Load original audio at its original sample rate
            y_orig, sr_orig = librosa.load(file_path, sr=None, mono=True)

            # Resample for processing (loop detection and feature extraction) if needed
            if sr_orig != PROC_SAMPLE_RATE:
                y_proc = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=PROC_SAMPLE_RATE)
            else:
                y_proc = y_orig

            # Resample for key detection if needed
            if sr_orig != KEY_SAMPLE_RATE:
                y_key = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=KEY_SAMPLE_RATE)
            else:
                y_key = y_orig

            # Determine if the audio is a loop
            loop_flag, bpm = is_loop(y_proc, PROC_SAMPLE_RATE, beat_track_kwargs=beat_track_kwargs, onset_detect_kwargs=onset_detect_kwargs)

            # Extract directory names for categorization
            dir_names = []
            parent_dir = os.path.dirname(file_path)
            while parent_dir and parent_dir != os.path.dirname(parent_dir):
                dir_names.append(os.path.basename(parent_dir).lower())
                parent_dir = os.path.dirname(parent_dir)
            dir_names_combined = ''.join(dir_names)  # Concatenate for faster substring checks

            # Categorize the file based on substrings
            filename = os.path.basename(file_path)
            category_path = categorize_file(filename, dir_names_combined, loop_flag)

            if category_path is None:
                # Use model to predict category
                features = extract_features(y_proc)
                if features is not None:
                    features = np.expand_dims(features, axis=0)  # Add batch dimension
                    prediction = model.predict(features, verbose=0)
                    predicted_class = np.argmax(prediction, axis=1)
                    predicted_label = le.inverse_transform(predicted_class)[0]
                    category_path = label_to_category_path(predicted_label)
                    logging.debug(f"File '{file_path}' classified by model as '{category_path}'.")
                else:
                    logging.error(f"Feature extraction failed for '{file_path}'. Skipping.")
                    continue
            else:
                logging.debug(f"File '{file_path}' categorized by substring as '{category_path}'.")

            # Ensure category_path is valid
            if category_path not in allowed_categories:
                logging.warning(f"Category '{category_path}' not in allowed structure. Skipping file '{file_path}'.")
                continue

            destination_dir = os.path.join(output_folder, category_path)

            filename_base, ext = os.path.splitext(filename)

            # Prepare new filename
            new_filename = filename_base
            append_info = []

            # Only perform key detection for tonal samples
            category_name = os.path.basename(category_path)
            if category_name in TONAL_CATEGORIES:
                key = detect_key(y_key, KEY_SAMPLE_RATE, key_extractor)
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

            # Normalize the original audio to have maximum amplitude of 1.0
            y_normalized = librosa.util.normalize(y_orig)

            # Save the normalized audio to the destination path with original sample rate
            sf.write(destination_path, y_normalized, sr_orig)

            logging.debug(f"Copied and normalized '{filename}' to '{destination_dir}' as '{new_filename}' with original sample rate {sr_orig} Hz.")

        except Exception as e:
            logging.error(f"Failed to process '{file_path}': {e}")
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

    # Pre-instantiate Essentia's KeyExtractor
    key_extractor = ess.KeyExtractor()

    organize_samples(input_folder, output_folder, model, le, key_extractor)
    print("Organization complete.")
    logging.info("Organization complete.")

if __name__ == "__main__":
    main()