import os
import shutil
import argparse
import sys
import math
from tqdm import tqdm
import librosa
import numpy as np
import logging
import pickle
from tensorflow.keras.models import load_model

# Setup logging
logging.basicConfig(
    filename='organize_samples.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Supported audio file extensions
AUDIO_EXTENSIONS = (
    '.wav', '.mp3', '.aiff', '.flac', '.ogg', '.m4a',
    '.aac', '.wma', '.alac', '.opus'
)

# Define substrings for each class, keys match the desired directory paths
filename_substrings = {
    # Loops/Drums
    'Loops/Drums/Full Drum Loop': ['breakbeat', 'break', 'drum break', 'full drum loop'],
    'Loops/Drums/Hihat': ['hihat loop', 'hi-hat loop', 'hat loop', 'closed hat loop', 'open hat loop'],
    'Loops/Drums/Percussion': ['perc loop', 'percussion loop', 'shaker loop', 'tambourine loop'],

    # Loops/Sounds
    'Loops/Sounds/Bass': ['bass loop', 'bassline loop', 'sub bass loop'],
    'Loops/Sounds/Chords': ['chord loop', 'chords loop', 'keyboard loop', 'piano loop', 'synth chords loop'],
    'Loops/Sounds/Synth': ['synth loop', 'synthesizer loop', 'pad loop'],
    'Loops/Sounds/Voice': ['voice loop', 'vocals loop', 'vocal loop', 'singing loop'],

    # Oneshots/Drums
    'Oneshots/Drums/Clap': ['clap', 'handclap', 'hand clap', 'snap'],
    'Oneshots/Drums/Cymbal': ['cymbal', 'china', 'ride cymbal', 'splash cymbal'],
    'Oneshots/Drums/Hand Percussion': ['bongo', 'conga', 'djembe', 'cajon', 'tabla', 'hand drum', 'bata', 'claves', 'cowbell'],
    'Oneshots/Drums/Hihat': ['hihat', 'hi-hat', 'hat', 'closed hat', 'open hat'],
    'Oneshots/Drums/Kick': ['kick', 'bass drum', 'bd', 'kick drum'],
    'Oneshots/Drums/Percussion': ['perc', 'percussion', 'shaker', 'tambourine'],
    'Oneshots/Drums/Snare': ['snare', 'snr', 'rimshot', 'brushes'],
    'Oneshots/Drums/Tom': ['tom', 'toms', 'floor tom', 'rack tom'],

    # Oneshots/Sounds
    'Oneshots/Sounds/Ambience & FX': ['ambience', 'ambient', 'fx', 'atmosphere', 'background noise'],
    'Oneshots/Sounds/Bass': ['bass', 'bassline', 'sub bass'],
    'Oneshots/Sounds/Brass': ['brass', 'trumpet', 'saxophone', 'trombone'],
    'Oneshots/Sounds/Chords': ['chord', 'chords', 'harmonic', 'harmony', 'keyboard', 'piano', 'synth chords'],
    'Oneshots/Sounds/Guitar & Plucked': ['guitar', 'pluck', 'plucked', 'acoustic guitar', 'electric guitar'],
    'Oneshots/Sounds/Lead': ['lead', 'lead part', 'lead instrument'],
    'Oneshots/Sounds/Mallets': ['mallet', 'mallets', 'xylophone', 'marimba'],
    'Oneshots/Sounds/Strings': ['string', 'strings', 'violin', 'cello', 'orchestra'],
    'Oneshots/Sounds/Voice': ['voice', 'vocals', 'vocal', 'singing'],
    'Oneshots/Sounds/Woodwind': ['woodwind', 'flute', 'sax', 'clarinet', 'oboe'],
}

# Tonal categories for which key detection should be performed
TONAL_CATEGORIES = [
    'Bass',
    'Chords',
    'Voice',
    'Brass',
    'Guitar & Plucked',
    'Lead',
    'Mallets',
    'Strings',
    'Woodwind',
    'Synth',
    'Piano',
    'Pads'
]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Organize audio samples into categorized folders based on model predictions and substring matching, and append Key and BPM information to filenames.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing audio samples.')
    parser.add_argument('output_folder', type=str, help='Path to the output folder where organized samples will be stored.')
    parser.add_argument('--move', action='store_true', help='Move files instead of copying them.')
    args = parser.parse_args()
    return args.input_folder, args.output_folder, args.move

def create_directory_structure(base_path):
    """
    Creates the required directory structure based on the specified folder hierarchy.
    """
    directories = [
        # Loops
        'Loops/Drums/Full Drum Loop',
        'Loops/Drums/Hihat',
        'Loops/Drums/Percussion',
        'Loops/Sounds/Bass',
        'Loops/Sounds/Chords',
        'Loops/Sounds/Synth',
        'Loops/Sounds/Voice',

        # Oneshots
        'Oneshots/Drums/Clap',
        'Oneshots/Drums/Cymbal',
        'Oneshots/Drums/Hand Percussion',
        'Oneshots/Drums/Hihat',
        'Oneshots/Drums/Kick',
        'Oneshots/Drums/Percussion',
        'Oneshots/Drums/Snare',
        'Oneshots/Drums/Tom',
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
    for directory in directories:
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Ensured directory exists: {dir_path}")
    # Ensure 'Unclassified' directory exists under both 'Oneshots' and 'Loops'
    for sample_type in ['Oneshots', 'Loops']:
        unclassified_dir = os.path.join(base_path, sample_type, 'Unclassified')
        os.makedirs(unclassified_dir, exist_ok=True)
        logging.info(f"Ensured directory exists: {unclassified_dir}")

def detect_key(file_path):
    """
    Detects the musical key of an audio file using librosa's key estimation.
    """
    try:
        y, sr = librosa.load(file_path, sr=22050)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        key_idx = np.argmax(chroma_mean)
        key_list = ['C', 'C#', 'D', 'D#', 'E', 'F',
                    'F#', 'G', 'G#', 'A', 'A#', 'B']
        key = key_list[key_idx]
        return key
    except Exception as e:
        logging.error(f"Error extracting key for '{file_path}': {e}")
        return 'Unknown'

def is_loop(file_path, bpm_threshold=30, beats_per_bar=4, tolerance=0.05):
    """
    Determines if a file is a loop based on BPM, duration, and number of transients.

    Returns:
    - bool: True if loop, False otherwise.
    - bpm (float): Estimated BPM of the loop.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)

        # Count the number of transients (onsets)
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        num_onsets = len(onsets)

        if num_onsets <= 1:
            return False, float(tempo)

        if tempo < bpm_threshold:
            return False, float(tempo)  # Too slow to be considered rhythmic

        # Calculate duration per beat and per bar
        beat_duration = 60.0 / tempo
        bar_duration = beats_per_bar * beat_duration

        # Number of bars (could be fractional)
        num_bars = duration / bar_duration

        if num_bars < 0.5:
            return False, float(tempo)  # Too short to be a loop

        # Ensure num_bars is a scalar
        if isinstance(num_bars, np.ndarray):
            num_bars_scalar = num_bars.item()
        else:
            num_bars_scalar = num_bars

        nearest_power = 2 ** round(math.log(num_bars_scalar, 2))

        if abs(num_bars_scalar - nearest_power) / nearest_power <= tolerance:
            return True, float(tempo)
        else:
            return False, float(tempo)
    except Exception as e:
        logging.error(f"Error processing '{file_path}': {e}")
        return False, 0.0

def extract_features_from_file(file_path):
    """
    Extracts MFCC features from an audio file, consistent with the training process.
    """
    SAMPLE_RATE = 22050
    DURATION = 5  # Duration used during training
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048
    N_MFCC = 20  # Should match the training script

    try:
        # Load audio file
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(signal) > SAMPLES_PER_TRACK:
            signal = signal[:SAMPLES_PER_TRACK]
        else:
            pad_width = SAMPLES_PER_TRACK - len(signal)
            signal = np.pad(signal, (0, pad_width), 'constant')

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(y=signal, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                    n_mels=N_MELS, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mfcc = librosa.power_to_db(mfcc, ref=np.max)
        mfcc = mfcc[..., np.newaxis]  # Add channel dimension
        return mfcc
    except Exception as e:
        logging.error(f"Error extracting features from '{file_path}': {e}")
        return None

def classify_file(file_path, model, label_encoder):
    """
    Classifies the audio file using the pretrained model.

    Returns:
    - predicted_category (str): The predicted category label.
    - confidence (float): The confidence of the prediction.
    """
    features = extract_features_from_file(file_path)
    if features is None:
        return None, 0.0  # Could not extract features

    # The model expects input shape of (batch_size, height, width, channels)
    features = np.expand_dims(features, axis=0)
    # Predict
    predictions = model.predict(features)
    confidence = np.max(predictions)
    predicted_class_index = np.argmax(predictions)
    predicted_label = label_encoder.inverse_transform([predicted_class_index])[0]
    predicted_category = label_to_directory(predicted_label)
    return predicted_category, confidence

def label_to_directory(label):
    """
    Maps a model's predicted label to the corresponding directory path.
    """
    # Map labels to directory paths
    label_directory_mapping = {
        # Loops/Drums
        'Full Drum Loop': 'Loops/Drums/Full Drum Loop',
        'Hihat Loop': 'Loops/Drums/Hihat',
        'Percussion Loop': 'Loops/Drums/Percussion',

        # Loops/Sounds
        'Bass Loop': 'Loops/Sounds/Bass',
        'Chords Loop': 'Loops/Sounds/Chords',
        'Synth Loop': 'Loops/Sounds/Synth',
        'Voice Loop': 'Loops/Sounds/Voice',

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
    return label_directory_mapping.get(label, None)

def categorize_file(file_path):
    """
    Categorizes the file based on substrings in the filename and its directory names.

    Returns the category directory path if matched, otherwise None.
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
    for category, substrings in filename_substrings.items():
        for substring in substrings:
            substring_clean = substring.lower().replace('&', '').replace(' ', '').replace('-', '').replace('_', '')
            if substring_clean in dir_names_combined:
                return category

    # If no match in directories, check the filename
    for category, substrings in filename_substrings.items():
        for substring in substrings:
            substring_clean = substring.lower().replace('&', '').replace(' ', '').replace('-', '').replace('_', '')
            if substring_clean in filename_clean:
                return category

    return None  # If no category matches

def organize_samples(input_folder, output_folder, model, label_encoder, move_files, tonal_categories):
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
        substring_category = categorize_file(file_path)

        # Initialize variables
        category_name = None
        destination_dir = None
        predicted_category = None  # Initialize predicted_category
        confidence = None  # Initialize confidence

        # First, try substring matching
        if substring_category is not None:
            category_name = substring_category
            destination_dir = os.path.join(output_folder, category_name)
        else:
            # Use model prediction
            predicted_category, confidence = classify_file(file_path, model, label_encoder)
            if confidence < 0.2 or predicted_category is None:
                # Move/copy to 'Unclassified'
                sample_type = 'Loops' if loop_flag else 'Oneshots'
                destination_dir = os.path.join(output_folder, sample_type, 'Unclassified')
                category_name = f"{sample_type}/Unclassified"
                logging.info(f"Low confidence ({confidence:.2f}) for '{file_path}'. Moving to 'Unclassified'.")
            else:
                category_name = predicted_category
                destination_dir = os.path.join(output_folder, category_name)
                # Ensure destination directory exists
                os.makedirs(destination_dir, exist_ok=True)

        # If both methods provided categories, and they differ, log a warning
        if substring_category and predicted_category and substring_category != predicted_category:
            logging.warning(f"Mismatch between substring category '{substring_category}' and model prediction '{predicted_category}' for '{file_path}'.")

        # Ensure destination directory exists
        if destination_dir is None:
            sample_type = 'Loops' if loop_flag else 'Oneshots'
            destination_dir = os.path.join(output_folder, sample_type, 'Unclassified')
            category_name = f"{sample_type}/Unclassified"
            os.makedirs(destination_dir, exist_ok=True)

        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)

        # Prepare new filename
        new_filename = name
        append_info = []

        # Only perform key detection for tonal samples
        # Extract the subcategory name from the category path
        subcategory_name = category_name.split('/')[-1]
        if subcategory_name in tonal_categories:
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
            if move_files:
                shutil.move(file_path, destination_path)
                logging.info(f"Moved '{filename}' to '{destination_dir}' as '{new_filename}'.")
            else:
                shutil.copy2(file_path, destination_path)
                logging.info(f"Copied '{filename}' to '{destination_dir}' as '{new_filename}'.")
        except Exception as e:
            logging.error(f"Failed to move/copy '{file_path}': {e}")

    logging.info("Organization complete.")

def main():
    input_folder, output_folder, move_files = parse_arguments()

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

    # Load the pretrained model and label encoder
    try:
        model = load_model('model/audio_classification_model.keras')
        with open('model/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        logging.info('Loaded pretrained model and label encoder.')
    except Exception as e:
        print(f"Error loading model or label encoder: {e}")
        logging.error(f"Error loading model or label encoder: {e}")
        sys.exit(1)

    # Define tonal categories for which key detection should be performed
    # These should match the subcategory names in the directory structure
    tonal_categories = [
        'Bass',
        'Chords',
        'Voice',
        'Brass',
        'Guitar & Plucked',
        'Lead',
        'Mallets',
        'Strings',
        'Woodwind',
        'Synth',
        'Piano',
        'Pads'
    ]

    organize_samples(input_folder, output_folder, model, label_encoder, move_files, tonal_categories)
    print("Organization complete.")
    logging.info("Organization complete.")

if __name__ == "__main__":
    main()