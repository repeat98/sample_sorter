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
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import pickle
from tqdm import tqdm  # For progress bars
import hashlib
import pickle

# --------------------------- Parameters ---------------------------

DATASET_PATH = "/Users/jannikassfalg/coding/sample_sorter/train"  # Update this path as needed
SAMPLE_RATE = 22050  # Sampling rate for audio
DURATION = 5  # Duration to which all audio files will be truncated or padded
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Mel-spectrogram parameters
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 20  # Increased from 13 to 20

# Model parameters
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.2
MIN_SAMPLES_PER_CLASS = 5  # Minimum samples required per class

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

def compute_fingerprint(dataset_path, audio_extensions, sample_rate, duration,
                       min_samples_per_class, n_mels, hop_length, n_fft, n_mfcc):
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
            'n_mfcc': n_mfcc
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
    Load cached features, labels, and label encoder.
    """
    X_features = np.load(FEATURES_PATH)
    y_filtered = np.load(LABELS_PATH)
    with open(LABEL_ENCODER_PATH, "rb") as le_file:
        le = pickle.load(le_file)
    return X_features, y_filtered, le

def save_cached_data(X_features, y_filtered, le, fingerprint_hash):
    """
    Save features, labels, label encoder, and fingerprint hash to cache.
    """
    np.save(FEATURES_PATH, X_features)
    np.save(LABELS_PATH, y_filtered)
    with open(LABEL_ENCODER_PATH, "wb") as le_file:
        pickle.dump(le, le_file)
    with open(FINGERPRINT_PATH, "wb") as fp_file:
        pickle.dump(fingerprint_hash, fp_file)

# ----------------------- Data Loading and Caching -----------------------

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
            signal, sr = librosa.load(file_path, sr=sample_rate)
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

def extract_features(X, sample_rate, n_mels, hop_length, n_fft, n_mfcc=13):
    mfccs = []
    for x in tqdm(X, desc="Extracting features"):
        try:
            mfcc = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=n_mfcc,
                                        n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
            mfcc = librosa.power_to_db(mfcc, ref=np.max)
            mfccs.append(mfcc)
        except Exception as e:
            print(f"Error extracting features: {e}")
            mfccs.append(np.zeros((n_mfcc, int(np.ceil(len(x)/hop_length))), dtype=np.float32))  # Placeholder
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
        N_MFCC
    )

    cache_exists = os.path.exists(FINGERPRINT_PATH) and \
                   os.path.exists(FEATURES_PATH) and \
                   os.path.exists(LABELS_PATH) and \
                   os.path.exists(LABEL_ENCODER_PATH)

    if cache_exists:
        print("Loading cached fingerprint...")
        with open(FINGERPRINT_PATH, "rb") as fp_file:
            saved_fingerprint = pickle.load(fp_file)

        if current_fingerprint == saved_fingerprint:
            print("Fingerprint matches. Loading cached features and labels...")
            X_features, y_filtered, le = load_cached_data()
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

            # Filter out samples from classes with fewer than MIN_SAMPLES
            filtered_indices = [i for i, label in enumerate(labels) if label in classes_to_keep]
            X_filtered = X[filtered_indices]
            y_filtered = y_encoded[filtered_indices]
            filtered_labels = labels[filtered_indices]

            print(f"Filtered dataset size: {X_filtered.shape[0]} samples")
            print(f"Number of classes after filtering: {len(classes_to_keep)}")

            # Print filtered class distribution
            print("\nClass distribution after filtering:")
            filtered_class_counts = Counter(filtered_labels)
            for label, count in filtered_class_counts.items():
                print(f"{label}: {count} samples")

            # -------------------- Feature Extraction --------------------
            print("\nExtracting features...")
            X_features = extract_features(X_filtered, SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, N_MFCC)
            X_features = X_features[..., np.newaxis]  # Add channel dimension
            print(f"Feature shape: {X_features.shape}")

            # Save cached data
            print("\nSaving extracted features and labels to cache...")
            save_cached_data(X_features, y_filtered, le, current_fingerprint)
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

        # Filter out samples from classes with fewer than MIN_SAMPLES
        filtered_indices = [i for i, label in enumerate(labels) if label in classes_to_keep]
        X_filtered = X[filtered_indices]
        y_filtered = y_encoded[filtered_indices]
        filtered_labels = labels[filtered_indices]

        print(f"Filtered dataset size: {X_filtered.shape[0]} samples")
        print(f"Number of classes after filtering: {len(classes_to_keep)}")

        # Print filtered class distribution
        print("\nClass distribution after filtering:")
        filtered_class_counts = Counter(filtered_labels)
        for label, count in filtered_class_counts.items():
            print(f"{label}: {count} samples")

        # -------------------- Feature Extraction --------------------
        print("\nExtracting features...")
        X_features = extract_features(X_filtered, SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, N_MFCC)
        X_features = X_features[..., np.newaxis]  # Add channel dimension
        print(f"Feature shape: {X_features.shape}")

        # Save cached data
        print("\nSaving extracted features and labels to cache...")
        save_cached_data(X_features, y_filtered, le, current_fingerprint)
        print("Cached data saved successfully.")

    # -------------------- Train-Test Split --------------------
    print("\nSplitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_features, y_filtered, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_filtered
    )
    y_train_categorical = to_categorical(y_train)
    y_val_categorical = to_categorical(y_val)
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

    # -------------------- Model Building --------------------
    def build_model(input_shape, num_classes):
        model = models.Sequential()

        # First Convolutional Block
        model.add(layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.25))

        # Second Convolutional Block
        model.add(layers.Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.25))

        # Third Convolutional Block
        model.add(layers.Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.25))

        # Flatten and Dense Layers
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        # Output Layer
        model.add(layers.Dense(num_classes, activation='softmax'))

        return model

    input_shape = X_train.shape[1:]
    num_classes_filtered = len(classes_to_keep)
    model = build_model(input_shape, num_classes_filtered)
    model.summary()

    # -------------------- Save the Label Encoder --------------------
    # (Already saved during caching)
    print("Label encoder is already saved to 'model/label_encoder.pkl'")

    # -------------------- Compilation --------------------
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # -------------------- Callbacks --------------------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("model/best_model.keras", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # -------------------- Calculate Class Weights --------------------
    print("\nCalculating class weights...")
    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_filtered),
        y=y_filtered
    )
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