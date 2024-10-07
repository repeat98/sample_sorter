import os
import sys
import argparse
import librosa
import numpy as np
from scipy.signal import correlate
import soundfile as sf
from tqdm import tqdm
import logging
import csv

def is_audio_file(filename):
    """
    Check if a file is an audio file based on its extension.
    """
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif', '.aifc', '.m4a']
    return any(filename.lower().endswith(ext) for ext in audio_extensions)

def load_audio(file_path):
    """
    Load an audio file using librosa.
    Returns the waveform and sample rate.
    """
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def normalize_audio(y):
    """
    Normalize the audio signal to have a maximum amplitude of 1.0.
    """
    if y is not None and np.max(np.abs(y)) > 0:
        return y / np.max(np.abs(y))
    return y

def extract_transients(y, sr):
    """
    Detect transients (onsets) in the audio signal.
    Returns the number of transients detected.
    """
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Detect onsets; you can adjust the parameters as needed
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=False, units='time')
        num_transients = len(onsets)
        return num_transients
    except Exception as e:
        print(f"Error extracting transients: {e}")
        return 0

def estimate_bpm(y, sr):
    """
    Estimate the BPM (tempo) of the audio signal.
    Returns the estimated BPM and the beat frames.
    """
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        # Ensure tempo is a scalar float
        if isinstance(tempo, np.ndarray):
            if tempo.size == 1:
                tempo = float(tempo)
            else:
                print(f"Unexpected tempo array size: {tempo.size}. Using first element.")
                tempo = float(tempo[0])
        return tempo, beat_frames
    except Exception as e:
        print(f"Error estimating BPM: {e}")
        return None, None

def is_loop(file_path, bpm_threshold=30, beats_per_bar=4, tolerance=0.05, transient_threshold=1, min_duration=2.0):
    """
    Determines if a file is a loop based on BPM, duration, and transients.

    Parameters:
    - file_path (str): Path to the audio file.
    - bpm_threshold (float): Minimum BPM to consider as rhythmic.
    - beats_per_bar (int): Number of beats in a musical bar.
    - tolerance (float): Acceptable deviation when checking bar multiples.
    - transient_threshold (int): Minimum number of transients to consider as a loop.
    - min_duration (float): Minimum duration in seconds to consider as a loop.

    Returns:
    - bool: True if loop, False otherwise.
    - bpm (float): Estimated BPM of the loop.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None, mono=True)
        
        # Normalize the audio
        y = librosa.util.normalize(y)

        # Estimate the tempo (BPM) and track beats
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)

        # If the sample is too short, it's unlikely to be a loop
        if duration < min_duration:
            logging.info(f"File '{file_path}' is too short ({duration:.2f} seconds) to be a loop.")
            return False, float(tempo)
        
        # If the tempo is below the threshold, it's unlikely to be a rhythmic loop
        if tempo < bpm_threshold:
            logging.info(f"File '{file_path}' tempo {tempo} BPM below threshold {bpm_threshold} BPM.")
            return False, float(tempo)
        
        # Detect transients (onset events)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        transients = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)

        # If there's only one transient, it's unlikely to be a loop
        if len(transients) <= transient_threshold:
            logging.info(f"File '{file_path}' has {len(transients)} transient(s), which is too few to be a loop.")
            return False, float(tempo)
        
        # Calculate duration per beat and per bar
        beat_duration = 60.0 / tempo
        bar_duration = beats_per_bar * beat_duration

        # Number of bars (could be fractional)
        num_bars = duration / bar_duration

        # Check if num_bars is close to an integer power of 2 (1, 2, 4, 8, 16, ...)
        if num_bars < 0.5:
            logging.info(f"File '{file_path}' has {num_bars:.2f} bars, which is too short to be a loop.")
            return False, float(tempo)  # Too short to be a loop

        nearest_power = 2 ** round(math.log(num_bars, 2))
        if abs(num_bars - nearest_power) / nearest_power <= tolerance:
            logging.info(f"File '{file_path}' identified as loop with {tempo} BPM and approximately {nearest_power} bars.")
            return True, float(tempo)
        else:
            logging.info(f"File '{file_path}' not a loop. BPM: {tempo}, Bars: {num_bars:.2f}, Nearest Power: {nearest_power}.")
            return False, float(tempo)
    except Exception as e:
        logging.error(f"Error processing '{file_path}': {e}")
        return False, 0.0

def classify_sample(y, sr, file_path, tolerance=0.05):
    """
    Classify the audio sample as 'Loop' or 'One-Shot' based on defined criteria.
    Includes path-based override for classification.
    """
    if y is None or sr is None:
        return "Error"

    # Path-Based Classification Override
    path_lower = file_path.lower()
    if "loop" in path_lower or "bpm" in path_lower:
        return "Loop"

    duration = librosa.get_duration(y=y, sr=sr)

    num_transients = extract_transients(y, sr)
    bpm, _ = estimate_bpm(y, sr)

    # Classification logic
    if duration <= 2.0 and num_transients <= 1:
        return "One-Shot"
    elif is_loop(y, sr, duration, num_transients, bpm, tolerance):
        return "Loop"
    else:
        return "One-Shot"

def process_file(file_path, tolerance=0.05):
    """
    Process a single audio file and return its classification details.
    """
    y, sr = load_audio(file_path)
    if y is None or sr is None:
        return {
            'file_path': file_path,
            'classification': 'Error',
            'duration_sec': None,
            'num_transients': None,
            'bpm': None
        }

    # Normalize the audio before processing
    y = normalize_audio(y)

    duration = librosa.get_duration(y=y, sr=sr)
    num_transients = extract_transients(y, sr)
    bpm, _ = estimate_bpm(y, sr)

    # Handle bpm to ensure it's a float
    if bpm is not None:
        try:
            bpm = float(bpm)
        except (ValueError, TypeError) as e:
            print(f"Error converting BPM to float for {file_path}: {e}")
            bpm = None

    classification = classify_sample(y, sr, file_path, tolerance)

    return {
        'file_path': file_path,
        'classification': classification,
        'duration_sec': round(duration, 2) if duration is not None else None,
        'num_transients': num_transients,
        'bpm': round(bpm, 2) if bpm is not None else None
    }

def traverse_directory(input_folder):
    """
    Traverse the input folder recursively and yield audio file paths.
    """
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if is_audio_file(file):
                yield os.path.join(root, file)

def save_results_to_csv(results, output_csv):
    """
    Save the classification results to a CSV file.
    """
    fieldnames = ['file_path', 'classification', 'duration_sec', 'num_transients', 'bpm']
    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"\nResults saved to {output_csv}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Classify audio samples as Loops or One-Shots.")
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing audio samples.')
    parser.add_argument('--output_csv', type=str, default='classification_results.csv',
                        help='Path to save the classification results CSV.')
    parser.add_argument('--tolerance', type=float, default=0.05,
                        help='Tolerance for loop length matching (default: 0.05).')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_csv = args.output_csv
    tolerance = args.tolerance

    if not os.path.isdir(input_folder):
        print(f"Error: {input_folder} is not a valid directory.")
        sys.exit(1)

    # Collect all audio files
    audio_files = list(traverse_directory(input_folder))
    if not audio_files:
        print("No audio files found in the provided directory.")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio files. Processing...\n")

    results = []
    for file_path in tqdm(audio_files, desc="Processing Files"):
        result = process_file(file_path, tolerance)
        results.append(result)
        # Print the result
        print(f"File: {file_path}")
        print(f"  Classification: {result['classification']}")
        print(f"  Duration (sec): {result['duration_sec']}")
        print(f"  Number of Transients: {result['num_transients']}")
        print(f"  BPM: {result['bpm']}\n")

    # Save results to CSV
    save_results_to_csv(results, output_csv)

if __name__ == "__main__":
    main()