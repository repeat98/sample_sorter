import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
import pickle
from tqdm import tqdm  # For progress bars
import hashlib

# --------------------------- Parameters ---------------------------

DATASET_PATH = "train_sm/"  # Update this path as needed
SAMPLE_RATE = 22050  # Sampling rate for audio
DURATION = 5  # Duration to which all audio files will be truncated or padded
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Audio feature extraction parameters
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 40  # Increased from 20 to capture more detailed features
N_CHROMA = 12  # Number of chroma features
N_SPECTRAL_CONTRAST = 5  # Reduced from 7 to prevent frequency band issues
N_ZERO_CROSSING = 1  # Zero Crossing Rate
N_SPECTRAL_CENTROID = 1  # Spectral Centroid
N_SPECTRAL_BANDWIDTH = 1  # Spectral Bandwidth
N_SPECTRAL_ROLLOFF = 1  # Spectral Rolloff
N_TONNETTE = 6  # Tonnetz features (librosa's tonnetz returns 6 features)

# Model parameters
BATCH_SIZE = 32  # Increased batch size for faster convergence
EPOCHS = 30  # Set back to 30 epochs
VALIDATION_SPLIT = 0.2
MIN_SAMPLES_PER_CLASS = 10  # Ensures a reasonable number of samples per class

# Supported audio file extensions
AUDIO_EXTENSIONS = (
    '.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif',
    '.aac', '.wma', '.m4a', '.alac', '.opus'
)

# Ensure the 'model' directory exists
os.makedirs('model', exist_ok=True)

# ----------------------- Caching Setup -----------------------

# Paths for cached data
FINGERPRINT_PATH = "model/fingerprint.pkl"
FEATURES_PATH = "model/X_features.npy"
LABELS_PATH = "model/y_filtered.npy"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
CLASSES_TO_KEEP_PATH = "model/classes_to_keep.pkl"

def compute_fingerprint(dataset_path, audio_extensions, sample_rate, duration,
                       min_samples_per_class, n_mels, hop_length, n_fft, n_mfcc,
                       n_chroma, n_spectral_contrast, n_zero_crossing,
                       n_spectral_centroid, n_spectral_bandwidth,
                       n_spectral_rolloff, n_tonnette):
    """
    Compute a fingerprint of the dataset based on file paths, modification times, and processing parameters.
    """
    fingerprint = {
        'files': [],
        'parameters': {
            'sample_rate': sample_rate,
            'duration': duration,
            'min_samples_per_class': min_samples_per_class,
            'n_mels': n_mels,
            'hop_length': hop_length,
            'n_fft': n_fft,
            'n_mfcc': n_mfcc,
            'n_chroma': n_chroma,
            'n_spectral_contrast': n_spectral_contrast,
            'n_zero_crossing': n_zero_crossing,
            'n_spectral_centroid': n_spectral_centroid,
            'n_spectral_bandwidth': n_spectral_bandwidth,
            'n_spectral_rolloff': n_spectral_rolloff,
            'n_tonnette': n_tonnette
        }
    }

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(audio_extensions):
                file_path = os.path.join(root, file)
                try:
                    mod_time = os.path.getmtime(file_path)
                    fingerprint['files'].append((file_path, mod_time))
                except Exception as e:
                    print(f"Error accessing {file_path}: {e}")

    # Sort the files to ensure consistent ordering
    fingerprint['files'].sort()

    # Serialize the fingerprint dictionary
    fingerprint_bytes = pickle.dumps(fingerprint)

    # Compute SHA256 hash of the serialized fingerprint
    fingerprint_hash = hashlib.sha256(fingerprint_bytes).hexdigest()

    return fingerprint_hash

def load_cached_data():
    """
    Load cached features, labels, label encoder, and classes_to_keep.
    """
    X_features = np.load(FEATURES_PATH)
    y_filtered = np.load(LABELS_PATH)
    with open(LABEL_ENCODER_PATH, "rb") as le_file:
        le = pickle.load(le_file)
    # Load the cached classes_to_keep
    with open(CLASSES_TO_KEEP_PATH, "rb") as ck_file:
        classes_to_keep = pickle.load(ck_file)
    return X_features, y_filtered, le, classes_to_keep

def save_cached_data(X_features, y_filtered, le, fingerprint_hash, classes_to_keep):
    """
    Save features, labels, label encoder, fingerprint hash, and classes_to_keep to cache.
    """
    np.save(FEATURES_PATH, X_features)
    np.save(LABELS_PATH, y_filtered)
    with open(LABEL_ENCODER_PATH, "wb") as le_file:
        pickle.dump(le, le_file)
    with open(FINGERPRINT_PATH, "wb") as fp_file:
        pickle.dump(fingerprint_hash, fp_file)
    # Save classes_to_keep
    with open(CLASSES_TO_KEEP_PATH, "wb") as ck_file:
        pickle.dump(classes_to_keep, ck_file)

def load_audio_files(dataset_path, sample_rate, duration, audio_extensions):
    X = []
    labels = []
    max_len = sample_rate * duration
    success_count = 0

    # Gather all file paths
    file_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(audio_extensions):
                file_path = os.path.join(root, file)
                file_paths.append((file_path, root))

    # Loop over file paths with progress bar
    for file_path, root in tqdm(file_paths, desc="Loading audio files"):
        # Get the relative path from dataset root to the parent directory of the file
        relative_dir = os.path.relpath(root, dataset_path)
        # Use this relative directory path as the label
        label = relative_dir.replace(os.sep, '_')  # Replace os.sep with '_'

        # Load audio file
        try:
            # Specify dtype to handle FutureWarning
            signal, sr = librosa.load(file_path, sr=sample_rate, dtype=np.float32)
            if len(signal) > max_len:
                signal = signal[:max_len]
            else:
                pad_width = max_len - len(signal)
                signal = np.pad(signal, (0, pad_width), 'constant')

            X.append(signal)
            labels.append(label)
            success_count += 1
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {success_count} samples successfully.")
    return np.array(X), np.array(labels)

def extract_features(X, sample_rate, n_mels, hop_length, n_fft, n_mfcc=40, n_chroma=12, n_spectral_contrast=5,
                    n_zero_crossing=1, n_spectral_centroid=1, n_spectral_bandwidth=1,
                    n_spectral_rolloff=1, n_tonnette=6):
    mfccs = []
    for x in tqdm(X, desc="Extracting features"):
        try:
            # Compute MFCCs
            mfcc = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=n_mfcc,
                                        n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
            mfcc = librosa.power_to_db(mfcc, ref=np.max)

            # Compute Delta MFCCs
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)

            # Compute Chroma Features
            chroma = librosa.feature.chroma_stft(y=x, sr=sample_rate, hop_length=hop_length, n_fft=n_fft, n_chroma=n_chroma)

            # Compute Spectral Contrast
            spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sample_rate, hop_length=hop_length, n_fft=n_fft, n_bands=n_spectral_contrast)

            # Compute Zero Crossing Rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=x, hop_length=hop_length)
            zero_crossing_rate = np.repeat(zero_crossing_rate, n_zero_crossing, axis=0)

            # Compute Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=x, sr=sample_rate, hop_length=hop_length)
            spectral_centroid = np.repeat(spectral_centroid, n_spectral_centroid, axis=0)

            # Compute Spectral Bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x, sr=sample_rate, hop_length=hop_length)
            spectral_bandwidth = np.repeat(spectral_bandwidth, n_spectral_bandwidth, axis=0)

            # Compute Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=x, sr=sample_rate, hop_length=hop_length)
            spectral_rolloff = np.repeat(spectral_rolloff, n_spectral_rolloff, axis=0)

            # Compute Tonnetz Features
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(x), sr=sample_rate)
            # Ensure tonnetz has the correct number of features
            if tonnetz.shape[0] < n_tonnette:
                tonnetz = np.pad(tonnetz, ((0, n_tonnette - tonnetz.shape[0]), (0,0)), 'constant')
            else:
                tonnetz = tonnetz[:n_tonnette, :]

            # Stack all features
            combined = np.vstack((
                mfcc, 
                delta_mfcc, 
                delta2_mfcc, 
                chroma, 
                spectral_contrast,
                zero_crossing_rate,
                spectral_centroid,
                spectral_bandwidth,
                spectral_rolloff,
                tonnetz
            ))

            mfccs.append(combined)
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Placeholder for failed extraction
            combined_shape = (
                n_mfcc * 3 + 
                n_chroma + 
                n_spectral_contrast + 
                n_zero_crossing + 
                n_spectral_centroid + 
                n_spectral_bandwidth + 
                n_spectral_rolloff + 
                n_tonnette, 
                int(np.ceil(len(x)/hop_length))
            )
            mfccs.append(np.zeros(combined_shape, dtype=np.float32))
    return np.array(mfccs)

# -------------------- Main Execution Flow --------------------

def main():
    print("Computing dataset fingerprint...")
    current_fingerprint = compute_fingerprint(
        DATASET_PATH,
        AUDIO_EXTENSIONS,
        SAMPLE_RATE,
        DURATION,
        MIN_SAMPLES_PER_CLASS,
        N_MELS,
        HOP_LENGTH,
        N_FFT,
        N_MFCC,
        N_CHROMA,
        N_SPECTRAL_CONTRAST,
        N_ZERO_CROSSING,
        N_SPECTRAL_CENTROID,
        N_SPECTRAL_BANDWIDTH,
        N_SPECTRAL_ROLLOFF,
        N_TONNETTE
    )

    cache_exists = all([
        os.path.exists(FINGERPRINT_PATH),
        os.path.exists(FEATURES_PATH),
        os.path.exists(LABELS_PATH),
        os.path.exists(LABEL_ENCODER_PATH),
        os.path.exists(CLASSES_TO_KEEP_PATH)
    ])

    if cache_exists:
        print("Loading cached fingerprint...")
        with open(FINGERPRINT_PATH, "rb") as fp_file:
            saved_fingerprint = pickle.load(fp_file)

        if current_fingerprint == saved_fingerprint:
            print("Fingerprint matches. Loading cached features and labels...")
            X_features, y_filtered, le, classes_to_keep = load_cached_data()
        else:
            print("Fingerprint does not match. Processing dataset...")
            X, labels = load_audio_files(DATASET_PATH, SAMPLE_RATE, DURATION, AUDIO_EXTENSIONS)

            # -------------------- Label Encoding --------------------
            print("\nEncoding labels...")
            le = LabelEncoder()
            y_encoded = le.fit_transform(labels)
            num_classes = len(le.classes_)
            print(f"Number of classes before filtering: {num_classes}")

            # Print class distribution
            print("\nClass distribution before filtering:")
            class_counts = Counter(labels)
            for label, count in class_counts.items():
                print(f"{label}: {count} samples")

            # -------------------- Filter Classes --------------------
            # Define minimum samples per class
            MIN_SAMPLES = MIN_SAMPLES_PER_CLASS

            # Identify classes to keep and removed classes
            classes_to_keep = [label for label, count in class_counts.items() if count >= MIN_SAMPLES]
            removed_classes = [label for label, count in class_counts.items() if count < MIN_SAMPLES]
            print(f"\nClasses to keep (>= {MIN_SAMPLES} samples): {len(classes_to_keep)}")
            print(f"Classes removed (<{MIN_SAMPLES} samples): {removed_classes}")
            print(f"Filtered out {len(removed_classes)} classes.")

            if removed_classes:
                # Filter out samples from classes with fewer than MIN_SAMPLES
                filtered_indices = [i for i, label in enumerate(labels) if label in classes_to_keep]
                X_filtered = X[filtered_indices]
                y_filtered = y_encoded[filtered_indices]
                filtered_labels = labels[filtered_indices]
            else:
                X_filtered = X
                y_filtered = y_encoded
                filtered_labels = labels

            print(f"Filtered dataset size: {X_filtered.shape[0]} samples")
            print(f"Number of classes after filtering: {len(classes_to_keep)}")

            # Print filtered class distribution
            print("\nClass distribution after filtering:")
            filtered_class_counts = Counter(filtered_labels)
            for label, count in filtered_class_counts.items():
                print(f"{label}: {count} samples")

            # -------------------- Feature Extraction --------------------
            print("\nExtracting features...")
            X_features = extract_features(
                X_filtered, 
                SAMPLE_RATE, 
                N_MELS, 
                HOP_LENGTH, 
                N_FFT, 
                N_MFCC, 
                N_CHROMA, 
                N_SPECTRAL_CONTRAST,
                N_ZERO_CROSSING,
                N_SPECTRAL_CENTROID,
                N_SPECTRAL_BANDWIDTH,
                N_SPECTRAL_ROLLOFF,
                N_TONNETTE
            )
            # Ensure all feature arrays have the same shape
            max_time = X_features.shape[2]
            target_time = 216  # Adjusted target time based on model input requirements

            if max_time < target_time:
                pad_width = target_time - max_time
                X_features = np.pad(X_features, ((0,0), (0,0), (0, pad_width), (0,0)), 'constant')
            elif max_time > target_time:
                X_features = X_features[:, :, :target_time, :]
            print(f"Feature shape before normalization: {X_features.shape}")

            # -------------------- Feature Normalization --------------------
            print("\nNormalizing features...")
            # Compute mean and std from the training data after splitting
            # Split first to prevent data leakage
            X_train, X_val, y_train, y_val = train_test_split(
                X_features, y_filtered, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_filtered
            )
            print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

            # Compute mean and std for normalization based on training data
            X_train_mean = X_train.mean(axis=(0, 2), keepdims=True)
            X_train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8  # To avoid division by zero

            # Normalize training and validation data
            X_train = (X_train - X_train_mean) / X_train_std
            X_val = (X_val - X_train_mean) / X_train_std
            print("Features normalized.")

            # Reshape for the model
            # Assuming the last dimension is the channel
            # If not, adjust accordingly
            # X_train = X_train[..., np.newaxis]
            # X_val = X_val[..., np.newaxis]

            # Save cached data
            print("\nSaving extracted features and labels to cache...")
            save_cached_data(X_features, y_filtered, le, current_fingerprint, classes_to_keep)
            print("Cached data saved successfully.")

    else:
        print("No cache found. Processing dataset...")
        X, labels = load_audio_files(DATASET_PATH, SAMPLE_RATE, DURATION, AUDIO_EXTENSIONS)

        # -------------------- Label Encoding --------------------
        print("\nEncoding labels...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(labels)
        num_classes = len(le.classes_)
        print(f"Number of classes before filtering: {num_classes}")

        # Print class distribution
        print("\nClass distribution before filtering:")
        class_counts = Counter(labels)
        for label, count in class_counts.items():
            print(f"{label}: {count} samples")

        # -------------------- Filter Classes --------------------
        # Define minimum samples per class
        MIN_SAMPLES = MIN_SAMPLES_PER_CLASS

        # Identify classes to keep and removed classes
        classes_to_keep = [label for label, count in class_counts.items() if count >= MIN_SAMPLES]
        removed_classes = [label for label, count in class_counts.items() if count < MIN_SAMPLES]
        print(f"\nClasses to keep (>= {MIN_SAMPLES} samples): {len(classes_to_keep)}")
        print(f"Classes removed (<{MIN_SAMPLES} samples): {removed_classes}")
        print(f"Filtered out {len(removed_classes)} classes.")

        if removed_classes:
            # Filter out samples from classes with fewer than MIN_SAMPLES
            filtered_indices = [i for i, label in enumerate(labels) if label in classes_to_keep]
            X_filtered = X[filtered_indices]
            y_filtered = y_encoded[filtered_indices]
            filtered_labels = labels[filtered_indices]
        else:
            X_filtered = X
            y_filtered = y_encoded
            filtered_labels = labels

        print(f"Filtered dataset size: {X_filtered.shape[0]} samples")
        print(f"Number of classes after filtering: {len(classes_to_keep)}")

        # Print filtered class distribution
        print("\nClass distribution after filtering:")
        filtered_class_counts = Counter(filtered_labels)
        for label, count in filtered_class_counts.items():
            print(f"{label}: {count} samples")

        # -------------------- Feature Extraction --------------------
        print("\nExtracting features...")
        X_features = extract_features(
            X_filtered, 
            SAMPLE_RATE, 
            N_MELS, 
            HOP_LENGTH, 
            N_FFT, 
            N_MFCC, 
            N_CHROMA, 
            N_SPECTRAL_CONTRAST,
            N_ZERO_CROSSING,
            N_SPECTRAL_CENTROID,
            N_SPECTRAL_BANDWIDTH,
            N_SPECTRAL_ROLLOFF,
            N_TONNETTE
        )
        # Ensure all feature arrays have the same shape
        max_time = X_features.shape[2]
        target_time = 216  # Adjusted target time based on model input requirements

        if max_time < target_time:
            pad_width = target_time - max_time
            X_features = np.pad(X_features, ((0,0), (0,0), (0, pad_width), (0,0)), 'constant')
        elif max_time > target_time:
            X_features = X_features[:, :, :target_time, :]
        print(f"Feature shape before normalization: {X_features.shape}")

        # -------------------- Feature Normalization --------------------
        print("\nNormalizing features...")
        # Split data first to prevent data leakage
        X_train, X_val, y_train, y_val = train_test_split(
            X_features, y_filtered, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_filtered
        )
        print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

        # Compute mean and std for normalization based on training data
        X_train_mean = X_train.mean(axis=(0, 2), keepdims=True)
        X_train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8  # To avoid division by zero

        # Normalize training and validation data
        X_train = (X_train - X_train_mean) / X_train_std
        X_val = (X_val - X_train_mean) / X_train_std
        print("Features normalized.")

        # Reshape for the model
        # Assuming the last dimension is the channel
        # If not, adjust accordingly
        # X_train = X_train[..., np.newaxis]
        # X_val = X_val[..., np.newaxis]

        # -------------------- Save Cached Data --------------------
        print("\nSaving extracted features and labels to cache...")
        save_cached_data(X_features, y_filtered, le, current_fingerprint, classes_to_keep)
        print("Cached data saved successfully.")

    # -------------------- Continue Main Flow --------------------

    # If cache was loaded, and data was already split and normalized
    if cache_exists and current_fingerprint == saved_fingerprint:
        print("\nSplitting data into train and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_features, y_filtered, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_filtered
        )
        print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

        # -------------------- Feature Normalization --------------------
        print("\nNormalizing features...")
        # Compute mean and std from the training data
        X_train_mean = X_train.mean(axis=(0, 2), keepdims=True)
        X_train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8  # To avoid division by zero

        # Normalize training and validation data
        X_train = (X_train - X_train_mean) / X_train_std
        X_val = (X_val - X_train_mean) / X_train_std
        print("Features normalized.")

    # -------------------- Model Building --------------------
    def build_model(input_shape, num_classes):
        model = models.Sequential()

        # First Convolutional Block
        model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=input_shape,
                                kernel_regularizer=regularizers.l2(0.0005)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.3))

        # Second Convolutional Block
        model.add(layers.Conv2D(128, (3,3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.0005)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.3))

        # Third Convolutional Block with increased filters
        model.add(layers.Conv2D(256, (3,3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.0005)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.4))

        # Fourth Convolutional Block with deeper architecture
        model.add(layers.Conv2D(512, (3,3), activation='relu', padding='same',
                                kernel_regularizer=regularizers.l2(0.0005)))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.4))

        # Global Average Pooling
        model.add(layers.GlobalAveragePooling2D())

        # Dense Layers with increased units and Dropout
        model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(num_classes, activation='softmax'))

        return model

    input_shape = X_train.shape[1:]
    num_classes_filtered = len(classes_to_keep)  # Use classes_to_keep here
    model = build_model(input_shape, num_classes_filtered)
    model.summary()

    # -------------------- Save the Label Encoder --------------------
    # (Already saved during caching)
    print("Label encoder is already saved to 'model/label_encoder.pkl'")

    # -------------------- Compilation --------------------
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # -------------------- Callbacks --------------------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("model/best_model.keras", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    ]

    # -------------------- Calculate Class Weights --------------------
    print("\nCalculating class weights...")
    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_filtered),
        y=y_filtered
    )
    # Normalize class weights to have a maximum of 10 to prevent excessively large weights
    class_weights_values = np.clip(class_weights_values, 1, 10)
    class_weights_dict = dict(enumerate(class_weights_values))
    print(f"Class weights: {class_weights_dict}")

    # -------------------- Training --------------------
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights_dict  # Integrated class weights
    )

    # -------------------- Evaluation --------------------
    # Plot training & validation accuracy and loss
    def plot_history(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    plot_history(history)

    # Classification Report
    print("\nEvaluating model...")
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = y_val

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred_classes,
        target_names=le.inverse_transform(range(num_classes_filtered)),
        zero_division=0
    ))

    # Confusion Matrix
    def plot_confusion_matrix(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(20, 20))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        # Normalize the confusion matrix.
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm_normalized.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, f"{cm_normalized[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    plot_confusion_matrix(y_true, y_pred_classes, le.inverse_transform(range(num_classes_filtered)))

    # -------------------- Save the Model --------------------
    model.save("model/audio_classification_model.keras")
    print("\nModel saved to 'model/audio_classification_model.keras'")

if __name__ == "__main__":
    main()