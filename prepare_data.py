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

# Setup logging
logging.basicConfig(
    filename='organize_samples.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Define substrings for each class. These should be lowercase for case-insensitive matching.
filename_substrings = {
    # Loops/Drums
    'Full Drum Loop': ['full drum loop', 'drum loop', 'drums', 'drum', 'breakbeat', 'break', 'drum break'],
    'Hihat': ['hihat loop', 'hi-hat loop', 'hat loop', 'closed hat loop', 'open hat loop'],
    'Percussion': ['perc loop', 'percussion loop', 'shaker loop', 'tambourine loop', 'conga loop', 'bongo loop'],

    # Loops/Sounds
    'Bass': ['bass loop', 'bassline loop', 'sub bass loop'],
    'Chords': ['chord loop', 'chords loop', 'keyboard loop', 'piano loop', 'synth chords loop'],
    'Synth': ['synth loop', 'synthesizer loop', 'synth lead loop', 'synth pad loop'],
    'Voice': ['vocal loop', 'vocals loop', 'voice loop', 'singing loop'],

    # Oneshots/Drums
    'Clap': ['clap', 'snap', 'handclap', 'hand clap'],
    'Cymbal': ['cymbal', 'china', 'ride cymbal', 'splash cymbal'],
    'Hand Percussion': ['hand percussion', 'hand drums', 'conga', 'bongo', 'djembe', 'tabla', 'shaker', 'tambourine', 'cowbell'],
    'Hihat': ['hihat', 'hi-hat', 'hat', 'closed hat', 'open hat'],
    'Kick': ['kick', 'bass drum', 'bd', 'kick drum'],
    'Percussion': ['perc', 'percussion', 'shaker', 'tambourine', 'cowbell'],
    'Snare': ['snare', 'snr', 'rimshot', 'brushes'],
    'Tom': ['tom', 'toms', 'floor tom', 'rack tom'],

    # Oneshots/Sounds
    'Ambience & FX': ['ambience', 'ambient', 'fx', 'effects', 'sfx', 'reverb', 'delay', 'echo', 'atmosphere', 'background noise'],
    'Bass': ['bass', 'bassline', 'sub bass'],
    'Brass': ['brass', 'trumpet', 'saxophone', 'trombone'],
    'Chords': ['chord', 'chords', 'keyboard', 'piano', 'synth chords'],
    'Guitar & Plucked': ['guitar', 'pluck', 'plucked', 'acoustic guitar', 'electric guitar', 'harp', 'banjo', 'ukulele', 'mandolin'],
    'Lead': ['lead', 'lead synth', 'lead guitar', 'melody'],
    'Mallets': ['mallet', 'mallets', 'xylophone', 'marimba', 'vibraphone', 'glockenspiel'],
    'Strings': ['string', 'strings', 'violin', 'cello', 'orchestra'],
    'Voice': ['voice', 'vocals', 'vocal', 'singing'],
    'Woodwind': ['woodwind', 'flute', 'sax', 'clarinet', 'oboe', 'bassoon'],
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

# Tonal categories for which key detection should be performed
TONAL_CATEGORIES = [
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
]

# Combined list of all possible substrings for categorization
ALL_SUBSTRINGS = list(set(list(LOOP_MAPPING.keys()) + list(ONESHOT_MAPPING.keys())))

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
    for category, substrings in filename_substrings.items():
        for substring in substrings:
            substring_clean = substring.lower().replace('&', '').replace(' ', '').replace('-', '').replace('_', '')
            if substring_clean in dir_names_combined:
                if is_loop_flag and category in LOOP_MAPPING:
                    return LOOP_MAPPING[category]
                elif not is_loop_flag and category in ONESHOT_MAPPING:
                    return ONESHOT_MAPPING[category]
    # If no match in directories, check the filename
    for category, substrings in filename_substrings.items():
        for substring in substrings:
            substring_clean = substring.lower().replace('&', '').replace(' ', '').replace('-', '').replace('_', '')
            if substring_clean in filename_clean:
                if is_loop_flag and category in LOOP_MAPPING:
                    return LOOP_MAPPING[category]
                elif not is_loop_flag and category in ONESHOT_MAPPING:
                    return ONESHOT_MAPPING[category]

    return None  # If no category matches

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
        category_path = categorize_file(file_path, loop_flag)

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
            # File does not have a matching substring; skip copying or handle accordingly
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