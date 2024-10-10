import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import joblib
from tqdm import tqdm
from pathlib import Path
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

# Set the directories
input_dir = 'train_sm'  # Replace with your audio directory path
cache_dir = 'cache'
output_dir = 'clustered_audio'

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Collect all audio file paths recursively
audio_extensions = ['*.wav', '*.mp3', '*.flac']  # Add other extensions if needed
audio_files = []
for ext in audio_extensions:
    audio_files.extend(Path(input_dir).rglob(ext))

# Check if any audio files were found
if not audio_files:
    raise FileNotFoundError(f"No audio files found in directory '{input_dir}' with the specified extensions.")

print(f"Found {len(audio_files)} audio files.")

# Feature cache files
feature_cache_file = os.path.join(cache_dir, 'features.joblib')
file_paths_cache_file = os.path.join(cache_dir, 'file_paths.joblib')

# Step 1 & 2: Load and extract features
if os.path.exists(feature_cache_file) and os.path.exists(file_paths_cache_file):
    print("Loading features from cache...")
    features = joblib.load(feature_cache_file)
    file_paths = joblib.load(file_paths_cache_file)
else:
    print("Extracting features...")
    features = []
    file_paths = []
    for file_path in tqdm(audio_files, desc="Extracting features"):
        try:
            # Load audio file with a fixed sample rate
            y, sr = librosa.load(file_path, sr=22050)  # Common sample rate for consistency

            # Skip very short audio files
            if len(y) < 2048:
                print(f"Warning: {file_path} is too short (length={len(y)}). Skipping.")
                continue

            # Normalize the audio signal
            y = librosa.util.normalize(y)

            # Zero-padding for short signals
            if len(y) < sr:
                pad_length = sr - len(y)
                y = np.pad(y, (0, pad_length), 'constant')

            # Feature extraction
            feature_vector = []

            # --- New Features ---
            # Audio Length in seconds
            duration = len(y) / sr

            # Transient Count (Onset Detection)
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            transient_count = len(onset_frames)

            # We'll include these raw values and let StandardScaler normalize them later
            feature_vector.append(duration)
            feature_vector.append(transient_count)
            # ---------------------

            # MFCCs
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_var = np.var(mfcc, axis=1)
            feature_vector.extend(mfcc_mean)
            feature_vector.extend(mfcc_var)

            # Chroma Features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_var = np.var(chroma, axis=1)
            feature_vector.extend(chroma_mean)
            feature_vector.extend(chroma_var)

            # Mel Spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_mean = np.mean(mel, axis=1)
            mel_var = np.var(mel, axis=1)
            feature_vector.extend(mel_mean)
            feature_vector.extend(mel_var)

            # Spectral Centroid
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            feature_vector.append(np.mean(spec_centroid))
            feature_vector.append(np.var(spec_centroid))

            # Spectral Bandwidth
            spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            feature_vector.append(np.mean(spec_bandwidth))
            feature_vector.append(np.var(spec_bandwidth))

            # Spectral Rolloff
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            feature_vector.append(np.mean(spec_rolloff))
            feature_vector.append(np.var(spec_rolloff))

            # Zero-Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            feature_vector.append(np.mean(zcr))
            feature_vector.append(np.var(zcr))

            # Root Mean Square Energy
            rms = librosa.feature.rms(y=y)
            feature_vector.append(np.mean(rms))
            feature_vector.append(np.var(rms))

            # Tonnetz Features
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_var = np.var(tonnetz, axis=1)
            feature_vector.extend(tonnetz_mean)
            feature_vector.extend(tonnetz_var)

            # Spectral Contrast
            spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            spec_contrast_mean = np.mean(spec_contrast, axis=1)
            spec_contrast_var = np.var(spec_contrast, axis=1)
            feature_vector.extend(spec_contrast_mean)
            feature_vector.extend(spec_contrast_var)

            # Add the feature vector to the list
            features.append(feature_vector)
            file_paths.append(str(file_path))

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not features:
        raise ValueError("No features were extracted from the audio files. Please check the files and try again.")

    features = np.array(features)
    joblib.dump(features, feature_cache_file)
    joblib.dump(file_paths, file_paths_cache_file)

# Step 2: Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 2: Dimensionality reduction with t-SNE
print("Performing dimensionality reduction...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
features_2d = tsne.fit_transform(features_scaled)

# Step 3: Clustering
print("Clustering the data...")
# Use DBSCAN for clustering
dbscan = DBSCAN(eps=5, min_samples=10)
cluster_labels = dbscan.fit_predict(features_2d)

# Handle noise points assigned with label -1
unique_labels = set(cluster_labels)
if -1 in unique_labels:
    unique_labels.remove(-1)
num_clusters = len(unique_labels)
print(f"Number of clusters found: {num_clusters}")

# Step 4: Organize audio files into cluster directories
print("Organizing files into cluster directories...")
for label in unique_labels:
    cluster_dir = os.path.join(output_dir, f'cluster_{label}')
    os.makedirs(cluster_dir, exist_ok=True)

# Separate noise points into a separate directory
noise_dir = os.path.join(output_dir, 'noise')
os.makedirs(noise_dir, exist_ok=True)

for file_path, label in tqdm(zip(file_paths, cluster_labels), desc="Organizing files", total=len(file_paths)):
    if label == -1:
        dest_dir = noise_dir
    else:
        dest_dir = os.path.join(output_dir, f'cluster_{label}')
    try:
        shutil.copy(file_path, dest_dir)
    except Exception as e:
        print(f"Error copying {file_path} to {dest_dir}: {e}")

# Step 6: Visualization
print("Visualizing the results...")
# Plot the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='tab10', s=5)
plt.colorbar(scatter)
plt.title('Clusters Visualization with t-SNE')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.savefig('clusters_visualization.png')
plt.show()

# Plot cluster sizes
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
plt.figure(figsize=(10, 6))
cluster_counts.plot(kind='bar')
plt.title('Cluster Sizes')
plt.xlabel('Cluster Label')
plt.ylabel('Number of Samples')
plt.savefig('cluster_sizes.png')
plt.show()

print("Process completed successfully.")