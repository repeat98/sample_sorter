import os
import argparse
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder
from shutil import copy2
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --------------------------- Parameters ---------------------------

SAMPLE_RATE = 22050  # Must match the training sample rate
DURATION = 5  # Must match the training duration in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Mel-spectrogram parameters (must match training)
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 20  # Must match training

# Pre-sorting parameters
ONESHOT_MAX_DURATION = 2.0  # Seconds; adjust based on your data
BEATS_PER_BAR = 4  # Assuming 4/4 time signature
BARS_ALLOWED = [1, 2, 4, 8, 16]  # Allowed number of bars for loops

# Supported audio file extensions
AUDIO_EXTENSIONS = (
    '.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif', 
    '.aac', '.wma', '.m4a', '.alac', '.opus', '.mid', '.midi'
)

# ----------------------- Logging Configuration -----------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ----------------------- Feature Extraction -----------------------

def extract_features(signal, sample_rate, n_mels, hop_length, n_fft, n_mfcc=20):
    """
    Extracts MFCC features from an audio signal.
    """
    try:
        mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, 
                                    n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        mfcc = librosa.power_to_db(mfcc, ref=np.max)
        mfcc = mfcc[..., np.newaxis]  # Add channel dimension
        return mfcc
    except Exception as e:
        logger.debug(f"Error extracting features: {e}")
        return None

# ----------------------- Load Label Encoder -----------------------

def load_label_encoder(le_path):
    """
    Loads the LabelEncoder from a pickle file.
    """
    try:
        with open(le_path, 'rb') as file:
            le = pickle.load(file)
        logger.info(f"Label encoder loaded from '{le_path}'.")
        return le
    except Exception as e:
        logger.error(f"Error loading label encoder: {e}")
        exit(1)



# ----------------------- Filename Substrings Mapping -----------------------

# Define substrings for each class. These should be lowercase for case-insensitive matching.
filename_substrings = {
    'Clap': ['clap', 'snap'],
    'Cymbal': ['cymbal', 'china'],
    'Hihat': ['hihat', 'hi-hat', 'hat'],
    'Kick': ['kick', 'bass drum'],
    'Percussion': ['perc', 'percussion', 'shaker'],
    'Snare': ['snare', 'snr'],
    'Tom': ['tom', 'toms'],
    'Breakbeat': ['breakbeat', 'break'],
    'Bass': ['bass'],
    'Chords': ['chord', 'chords'],
    'FX': ['fx', 'effects'],
    'Melody': ['melody', 'lead'],
    'Voice': ['voice', 'vocals'],
    'Ambience & FX': ['ambience', 'ambient', 'fx'],
    'Brass': ['brass'],
    'Chord': ['chord'],
    'Guitar & Plucked': ['guitar', 'pluck', 'plucked'],
    'Lead': ['lead'],
    'Mallets': ['mallet', 'mallets'],
    'Strings': ['string', 'strings'],
    'Woodwind': ['woodwind', 'flute', 'sax'],
}

# ----------------------- Define Class Mapping -----------------------

# Updated class_mapping based on model's class labels
class_mapping = {
    # Oneshots/Drums
    'Clap': 'Oneshots/Drums/Clap',
    'Cymbal': 'Oneshots/Drums/Cymbal',
    'Hihat': 'Oneshots/Drums/Hihat',
    'Kick': 'Oneshots/Drums/Kick',
    'Percussion': 'Oneshots/Drums/Percussion',
    'Snare': 'Oneshots/Drums/Snare',
    'Tom': 'Oneshots/Drums/Tom',
    
    # Loops/Drums
    'Breakbeat': 'Loops/Drums/Breakbeat',
    
    # Loops/Sounds
    'Bass': 'Loops/Sounds/Bass',
    'Chords': 'Loops/Sounds/Chords',
    'FX': 'Loops/Sounds/FX',
    'Melody': 'Loops/Sounds/Melody',
    'Voice': 'Loops/Sounds/Voice',
    
    # Oneshots/Sounds
    'Ambience & FX': 'Oneshots/Sounds/Ambience & FX',
    'Brass': 'Oneshots/Sounds/Brass',
    'Chord': 'Oneshots/Sounds/Chord',
    'Guitar & Plucked': 'Oneshots/Sounds/Guitar & Plucked',
    'Lead': 'Oneshots/Sounds/Lead',
    'Mallets': 'Oneshots/Sounds/Mallets',
    'Strings': 'Oneshots/Sounds/Strings',
    'Woodwind': 'Oneshots/Sounds/Woodwind',
}

# ----------------------- Pre-sorting Function -----------------------

def pre_sort(file_path):
    """
    Determines whether an audio file is a 'Oneshot' or a 'Loop' based on its duration and BPM.
    
    Parameters:
        file_path (Path): Path to the audio file.
    
    Returns:
        category (str): 'Oneshot', 'Loop', or 'Unknown'.
    """
    try:
        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        duration = librosa.get_duration(y=signal, sr=sr)
        
        if duration <= ONESHOT_MAX_DURATION:
            return 'Oneshot'
        else:
            # Estimate tempo (BPM)
            tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=sr)

            # Validate tempo
            if not (40 <= tempo <= 250):
                logger.debug(f"Unrealistic tempo {tempo} BPM for '{file_path.name}'.")
                return 'Unknown'

            # Calculate number of bars
            bars = (duration * tempo) / (60 * BEATS_PER_BAR)
            
            # Introduce tolerance (e.g., +/- 0.2 bars)
            for allowed_bars in BARS_ALLOWED:
                if abs(bars - allowed_bars) <= 0.2:
                    return 'Loop'
            
            return 'Oneshot'
    except Exception as e:
        logger.debug(f"Error in pre-sorting '{file_path}': {e}")
        return 'Unknown'
# ----------------------- Classification Function -----------------------
def process_file(file_path, model, le, destination_folder, filename_substrings):
    """
    Processes a single audio file: pre-sorts, classifies using filename hints and model predictions, and copies to the destination.
    
    Parameters:
        file_path (Path): Path to the audio file.
        model (tf.keras.Model): Trained Keras model for classification.
        le (LabelEncoder): Fitted LabelEncoder.
        destination_folder (Path): Base destination directory.
        filename_substrings (dict): Mapping of class labels to substrings.
    
    Returns:
        result (str): Status message.
    """
    try:
        # Pre-sort: Determine if 'Oneshot' or 'Loop'
        category = pre_sort(file_path)
        
        if category in ['Oneshot', 'Loop']:
            # Initialize probable_class as None
            probable_class = None
            
            # Extract filename without extension and convert to lowercase
            filename = file_path.stem.lower()
            
            # Search for substrings in the filename
            for class_label, substrings in filename_substrings.items():
                for substring in substrings:
                    if substring in filename:
                        probable_class = class_label
                        logger.debug(f"Filename hint detected: '{substring}' suggests class '{class_label}' for file '{file_path.name}'.")
                        break
                if probable_class:
                    break  # Stop after the first match
            
            # Load audio file
            signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # Truncate or pad signal
            if len(signal) > SAMPLES_PER_TRACK:
                signal = signal[:SAMPLES_PER_TRACK]
            else:
                pad_width = SAMPLES_PER_TRACK - len(signal)
                signal = np.pad(signal, (0, pad_width), 'constant')

            # Extract features
            features = extract_features(signal, SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, N_MFCC)
            if features is None:
                logger.debug(f"Feature extraction failed for '{file_path.name}'.")
                return "error"
            
            # Expand dimensions to match model input
            features = np.expand_dims(features, axis=0)  # Shape: (1, n_mfcc, time, 1)

            # Predict
            predictions = model.predict(features, verbose=0)
            predicted_index = np.argmax(predictions, axis=1)[0]
            predicted_label = le.inverse_transform([predicted_index])[0]
            predicted_confidence = predictions[0][predicted_index]
            logger.debug(f"Model prediction for '{file_path.name}': {predicted_label} ({predicted_confidence:.2f})")
            
            final_label = predicted_label  # Default to model's prediction

            # If a probable class is detected from filename, decide how to integrate it
            if probable_class:
                if probable_class == predicted_label:
                    # If both agree, you might boost confidence or proceed as usual
                    logger.debug(f"Filename hint and model prediction agree for '{file_path.name}'.")
                else:
                    # If they disagree, you can choose to override, adjust, or log for manual review
                    # Here, we'll prioritize the filename hint
                    logger.debug(f"Filename hint '{probable_class}' overrides model prediction '{predicted_label}' for '{file_path.name}'.")
                    final_label = probable_class

            # Determine destination path based on 'Oneshot' or 'Loop'
            if final_label not in class_mapping:
                unknown_dir = destination_folder / 'Unknown'
                unknown_dir.mkdir(parents=True, exist_ok=True)
                dest_path = unknown_dir / file_path.name
                logger.debug(f"Class '{final_label}' not found in class mapping. Moving '{file_path.name}' to 'Unknown'.")
            else:
                # Prefix the path with 'Oneshots' or 'Loops' based on pre-sorting
                base_category = 'Oneshots' if category == 'Oneshot' else 'Loops'
                specific_category = class_mapping[final_label].replace('Oneshots/', '').replace('Loops/', '')
                dest_path = destination_folder / base_category / specific_category / file_path.name
                logger.debug(f"Moving '{file_path.name}' to '{dest_path}'.")
        
            # Ensure destination directory exists
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy file
            copy2(file_path, dest_path)

            return "processed"
        
        else:
            # Handle unknown category
            logger.debug(f"File '{file_path.name}' classified as 'Unknown'. Skipping.")
            return "skipped"
    
    except Exception as e:
        logger.debug(f"Error processing '{file_path}': {e}")
        return "error"
def classify_and_sort(input_folder, destination_folder, model, le, filename_substrings, max_workers=4):
    """
    Pre-sorts and classifies audio files into their respective directories based on 'Oneshot' or 'Loop' categorization.
    Utilizes concurrency and displays a progress bar.
    
    Parameters:
        input_folder (Path): Path to the input folder.
        destination_folder (Path): Path to the destination folder.
        model (tf.keras.Model): Trained Keras model.
        le (LabelEncoder): Loaded LabelEncoder.
        filename_substrings (dict): Mapping of class labels to substrings.
        max_workers (int): Number of concurrent threads.
    """
    input_folder = Path(input_folder)
    destination_folder = Path(destination_folder)

    if not input_folder.exists():
        logger.error(f"Input folder '{input_folder}' does not exist.")
        exit(1)

    # Create destination directories if they don't exist
    for path in class_mapping.values():
        dest_path = destination_folder / path
        dest_path.mkdir(parents=True, exist_ok=True)
    (destination_folder / 'Unknown').mkdir(parents=True, exist_ok=True)
    logger.info("Destination directories ensured.")

    # Traverse the input directory
    audio_files = [f for f in input_folder.rglob('*') if f.suffix.lower() in AUDIO_EXTENSIONS]
    logger.info(f"Found {len(audio_files)} audio files in '{input_folder}'. Starting classification...")

    if not audio_files:
        logger.warning("No audio files to process. Exiting.")
        return

    # Use ThreadPoolExecutor for concurrency
    processed = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file, file_path, model, le, destination_folder, filename_substrings): file_path
            for file_path in audio_files
        }

        # Use tqdm to display progress
        for future in tqdm(as_completed(future_to_file), total=len(audio_files), desc="Processing files"):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result == "processed":
                    processed += 1
                elif result == "skipped":
                    skipped += 1
                elif result == "error":
                    errors += 1
            except Exception as e:
                errors += 1
                logger.error(f"Unhandled exception for '{file_path.name}': {e}")

    logger.info("Classification and sorting completed.")
    logger.info(f"Total files processed: {processed}")
    logger.info(f"Total files skipped: {skipped}")
    logger.info(f"Total errors: {errors}")

# ----------------------- Main Function -----------------------

def main():
    parser = argparse.ArgumentParser(description="Classify and sort audio files into directories based on a trained model with integrated pre-sorting and filename heuristics.")
    parser.add_argument('input_folder', type=str, help="Path to the input folder containing audio files to classify.")
    parser.add_argument('destination_folder', type=str, help="Path to the destination folder where files will be sorted.")
    parser.add_argument('--model_path', type=str, default='audio_classification_model.keras', help="Path to the trained Keras model.")
    parser.add_argument('--label_encoder', type=str, default='label_encoder.pkl', help="Path to the saved LabelEncoder pickle file.")
    parser.add_argument('--max_workers', type=int, default=4, help="Maximum number of worker threads for concurrent processing.")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging for debugging.")
    args = parser.parse_args()

    # Adjust logging level based on verbosity
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")

    # Load the trained model
    if not os.path.exists(args.model_path):
        logger.error(f"Model file '{args.model_path}' does not exist.")
        exit(1)
    model = load_model(args.model_path)
    logger.info(f"Model loaded from '{args.model_path}'.")

    # Load the label encoder
    le = load_label_encoder(args.label_encoder)

    # Define filename substrings mapping (ensure this matches Step 1)
    filename_substrings = {
        'Clap': ['clap', 'snap'],
        'Cymbal': ['cymbal', 'china'],
        'Hihat': ['hihat', 'hi-hat', 'hat'],
        'Kick': ['kick', 'bass drum'],
        'Percussion': ['perc', 'percussion', 'shaker'],
        'Snare': ['snare', 'snr'],
        'Tom': ['tom', 'toms'],
        'Breakbeat': ['breakbeat', 'break'],
        'Bass': ['bass'],
        'Chords': ['chord', 'chords'],
        'FX': ['fx', 'effects'],
        'Melody': ['melody', 'lead'],
        'Voice': ['voice', 'vocals'],
        'Ambience & FX': ['ambience', 'ambient', 'fx'],
        'Brass': ['brass'],
        'Chord': ['chord'],
        'Guitar & Plucked': ['guitar', 'pluck', 'plucked'],
        'Lead': ['lead'],
        'Mallets': ['mallet', 'mallets'],
        'Strings': ['string', 'strings'],
        'Woodwind': ['woodwind', 'flute', 'sax'],
    }

    # Classify and sort files
    classify_and_sort(args.input_folder, args.destination_folder, model, le, filename_substrings, max_workers=args.max_workers)

if __name__ == "__main__":
    main()