import os
import librosa
import numpy as np

def is_oneshot(y, sr):
    length = librosa.get_duration(y=y, sr=sr)
    if length > 2:
        return False
    
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
    if len(peaks) > 1:
        return False
    
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return tempo <= 0

def is_loop(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    if tempo <= 0:
        return False
    
    length = librosa.get_duration(y=y, sr=sr)
    length_of_bar = 60 / tempo * 4
    return length >= length_of_bar * 0.8 and length <= length_of_bar * 1.2

def classify_samples(folder_path):
    results = {}
    
    if not os.path.exists(folder_path):
        print(f"Directory does not exist: {folder_path}")
        return
    
    print(f"Classifying samples in: {folder_path}")
    
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.wav', '.mp3', '.ogg', '.flac')):  # Add other formats as needed
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")  # Track which file is being processed
                
                try:
                    y, sr = librosa.load(file_path, sr=None)
                    print(f"Loaded {file_path} with shape {y.shape} and sample rate {sr}")

                    length = librosa.get_duration(y=y, sr=sr)
                    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                    peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True)
                    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
                    
                    classification = 'Unknown'  # Default classification
                    if is_oneshot(y, sr):
                        classification = 'Oneshot'
                    elif is_loop(y, sr):
                        classification = 'Loop'
                    
                    # Store result
                    results[file_path] = classification
                    
                    # Print result and additional debug info
                    print(f"{file_path}: {classification} (Length: {length:.2f} sec, Tempo: {tempo:.2f}, Transients: {len(peaks)})")
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Usage
folder_path = '/Users/jannikassfalg/Documents/Sample Packs/Audiotent - Hypnosis'  # Update this path
classify_samples(folder_path)