import os
import sys
import argparse
import librosa
import numpy as np
from scipy.signal import correlate
import soundfile as sf
from tqdm import tqdm
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

def is_loop(y, sr, duration, num_transients, bpm, tolerance=0.05):
    """
    Determine if the audio is a loop based on:
    - Rhythmicity (BPM extractable)
    - Length is a multiple of a musical bar
    - More than one transient
    """
    if bpm is None or bpm == 0:
        return False

    # Assume 4/4 time signature
    beats_per_bar = 4
    bar_duration = (60 / bpm) * beats_per_bar  # Duration of one bar in seconds

    # Calculate how many bars fit into the audio duration
    num_bars = duration / bar_duration

    # Ensure num_bars is a scalar float
    if isinstance(num_bars, np.ndarray):
        try:
            num_bars = num_bars.item()
        except:
            print(f"Unexpected type for num_bars: {type(num_bars)}")
            return False

    # Check if num_bars is approximately an integer
    if np.isclose(num_bars, np.round(num_bars), atol=tolerance * num_bars):
        if num_transients > 1:
            return True
    return False

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