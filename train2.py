import os
import numpy as np
import librosa
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import pickle
from tqdm import tqdm
import hashlib

# --------------------------- Parameters ---------------------------

DATASET_PATH = "train"
SAMPLE_RATE = 22050
DURATION = 2
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Audio feature extraction parameters
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 30
N_CHROMA = 12
N_SPECTRAL_CONTRAST = 5
N_ZERO_CROSSING = 1
N_SPECTRAL_CENTROID = 1
N_SPECTRAL_BANDWIDTH = 1
N_SPECTRAL_ROLLOFF = 1
N_TONNETZ = 6

# Model parameters
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
MIN_SAMPLES_PER_CLASS = 10

# Supported audio file extensions
AUDIO_EXTENSIONS = (
    '.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aif',
    '.aac', '.wma', '.m4a', '.alac', '.opus'
)

os.makedirs('model', exist_ok=True)

# ----------------------- Caching Setup -----------------------

FINGERPRINT_PATH = "model/fingerprint.pkl"
FEATURES_PATH = "model/X_features.npy"
LABELS_PATH = "model/y_filtered.npy"
LABEL_ENCODER_PATH = "model/label_encoder.pkl"
CLASSES_TO_KEEP_PATH = "model/classes_to_keep.pkl"

def compute_fingerprint(dataset_path, audio_extensions, sample_rate, duration,
                        min_samples_per_class, n_mels, hop_length, n_fft,
                        n_mfcc, n_chroma, n_spectral_contrast,
                        n_zero_crossing, n_spectral_centroid,
                        n_spectral_bandwidth, n_spectral_rolloff, n_tonnetz):
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
            'n_tonnetz': n_tonnetz
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
    fingerprint['files'].sort()
    fingerprint_bytes = pickle.dumps(fingerprint)
    fingerprint_hash = hashlib.sha256(fingerprint_bytes).hexdigest()
    return fingerprint_hash

def load_cached_data():
    X_features = np.load(FEATURES_PATH)
    y_filtered = np.load(LABELS_PATH)
    with open(LABEL_ENCODER_PATH, "rb") as le_file:
        le = pickle.load(le_file)
    with open(CLASSES_TO_KEEP_PATH, "rb") as ck_file:
        classes_to_keep = pickle.load(ck_file)
    return X_features, y_filtered, le, classes_to_keep

def save_cached_data(X_features, y_filtered, le, fingerprint_hash,
                     classes_to_keep):
    np.save(FEATURES_PATH, X_features)
    np.save(LABELS_PATH, y_filtered)
    with open(LABEL_ENCODER_PATH, "wb") as le_file:
        pickle.dump(le, le_file)
    with open(FINGERPRINT_PATH, "wb") as fp_file:
        pickle.dump(fingerprint_hash, fp_file)
    with open(CLASSES_TO_KEEP_PATH, "wb") as ck_file:
        pickle.dump(classes_to_keep, ck_file)

def load_audio_files(dataset_path, sample_rate, duration, audio_extensions):
    X = []
    labels = []
    max_len = sample_rate * duration
    success_count = 0
    file_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(audio_extensions):
                file_path = os.path.join(root, file)
                file_paths.append((file_path, root))
    for file_path, root in tqdm(file_paths, desc="Loading audio files"):
        relative_dir = os.path.relpath(root, dataset_path)
        label = relative_dir.replace(os.sep, '_')
        try:
            signal, sr = librosa.load(file_path, sr=sample_rate,
                                      dtype=np.float32)
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

def extract_features(X, sample_rate, n_mels, hop_length, n_fft, n_mfcc=30,
                     n_chroma=12, n_spectral_contrast=5, n_zero_crossing=1,
                     n_spectral_centroid=1, n_spectral_bandwidth=1,
                     n_spectral_rolloff=1, n_tonnetz=6):
    mfccs = []
    for x in tqdm(X, desc="Extracting features"):
        try:
            x_norm = (x - np.mean(x)) / (np.std(x) + 1e-8)
            D = librosa.stft(x, hop_length=hop_length, n_fft=n_fft)
            S = np.abs(D) ** 2
            mel = librosa.feature.melspectrogram(S=S, sr=sample_rate,
                                                 n_mels=n_mels)
            mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel),
                                        n_mfcc=n_mfcc)
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            chroma = librosa.feature.chroma_stft(S=S, sr=sample_rate,
                                                 hop_length=hop_length,
                                                 n_fft=n_fft,
                                                 n_chroma=n_chroma)
            spectral_contrast = librosa.feature.spectral_contrast(
                S=S, sr=sample_rate, hop_length=hop_length, n_fft=n_fft,
                n_bands=n_spectral_contrast)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(
                x, hop_length=hop_length)
            zero_crossing_rate = np.repeat(zero_crossing_rate,
                                           n_zero_crossing, axis=0)
            spectral_centroid = librosa.feature.spectral_centroid(
                S=S, sr=sample_rate, hop_length=hop_length)
            spectral_centroid = np.repeat(spectral_centroid,
                                          n_spectral_centroid, axis=0)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                S=S, sr=sample_rate, hop_length=hop_length)
            spectral_bandwidth = np.repeat(spectral_bandwidth,
                                           n_spectral_bandwidth, axis=0)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                S=S, sr=sample_rate, hop_length=hop_length)
            spectral_rolloff = np.repeat(spectral_rolloff,
                                         n_spectral_rolloff, axis=0)
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(x),
                                              sr=sample_rate)
            if tonnetz.shape[0] != n_tonnetz:
                tonnetz = np.resize(tonnetz, (n_tonnetz, tonnetz.shape[1]))
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
            combined_shape = (
                n_mfcc * 3 +
                n_chroma +
                n_spectral_contrast +
                n_zero_crossing +
                n_spectral_centroid +
                n_spectral_bandwidth +
                n_spectral_rolloff +
                n_tonnetz,
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
        N_TONNETZ
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
            cache_exists = False
    if not cache_exists:
        print("Processing dataset...")
        X, labels = load_audio_files(DATASET_PATH, SAMPLE_RATE, DURATION,
                                     AUDIO_EXTENSIONS)
        print("\nEncoding labels...")
        le = LabelEncoder()
        y_encoded = le.fit_transform(labels)
        num_classes = len(le.classes_)
        print(f"Number of classes before filtering: {num_classes}")
        print("\nClass distribution before filtering:")
        class_counts = Counter(labels)
        for label, count in class_counts.items():
            print(f"{label}: {count} samples")
        MIN_SAMPLES = MIN_SAMPLES_PER_CLASS
        classes_to_keep = [label for label, count in class_counts.items()
                           if count >= MIN_SAMPLES]
        removed_classes = [label for label, count in class_counts.items()
                           if count < MIN_SAMPLES]
        print(f"\nClasses to keep (>= {MIN_SAMPLES} samples):"
              f" {len(classes_to_keep)}")
        print(f"Classes removed (<{MIN_SAMPLES} samples): {removed_classes}")
        print(f"Filtered out {len(removed_classes)} classes.")
        if removed_classes:
            filtered_indices = [i for i, label in enumerate(labels)
                                if label in classes_to_keep]
            X_filtered = X[filtered_indices]
            y_filtered = y_encoded[filtered_indices]
            filtered_labels = labels[filtered_indices]
        else:
            X_filtered = X
            y_filtered = y_encoded
            filtered_labels = labels
        print(f"Filtered dataset size: {X_filtered.shape[0]} samples")
        print(f"Number of classes after filtering: {len(classes_to_keep)}")
        print("\nClass distribution after filtering:")
        filtered_class_counts = Counter(filtered_labels)
        for label, count in filtered_class_counts.items():
            print(f"{label}: {count} samples")
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
            N_TONNETZ
        )
        num_samples = len(X_features)
        num_frames_list = [feat.shape[1] for feat in X_features]
        target_time = max(num_frames_list)
        X_features_padded = []
        for feat in X_features:
            pad_width_feat = target_time - feat.shape[1]
            if pad_width_feat > 0:
                feat_padded = np.pad(feat, ((0, 0), (0, pad_width_feat)),
                                     'constant')
            else:
                feat_padded = feat
            X_features_padded.append(feat_padded)
        X_features = np.array(X_features_padded)
        print(f"Feature shape before normalization: {X_features.shape}")
        print("\nSaving extracted features and labels to cache...")
        save_cached_data(X_features, y_filtered, le, current_fingerprint,
                         classes_to_keep)
        print("Cached data saved successfully.")

    print("\nSplitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_features, y_filtered, test_size=VALIDATION_SPLIT, random_state=42,
        stratify=y_filtered
    )
    print(f"Training samples: {X_train.shape[0]},"
          f" Validation samples: {X_val.shape[0]}")

    print("\nNormalizing features...")
    X_train_mean = X_train.mean(axis=(0, 2), keepdims=True)
    X_train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train = (X_train - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std
    print("Features normalized.")

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]

    def build_model(input_shape, num_classes):
        inputs = layers.Input(shape=input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    input_shape = X_train.shape[1:]
    num_classes_filtered = len(classes_to_keep)
    model = build_model(input_shape, num_classes_filtered)
    model.summary()

    print("\nSaving label encoder to 'model/label_encoder.pkl'")
    with open(LABEL_ENCODER_PATH, "wb") as le_file:
        pickle.dump(le, le_file)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                         restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("model/best_model.keras",
                                           save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                             patience=5, verbose=1)
    ]

    print("\nCalculating class weights...")
    class_weights_values = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_filtered),
        y=y_filtered
    )
    class_weights_values = np.clip(class_weights_values, 1, 10)
    class_weights_dict = dict(enumerate(class_weights_values))
    print(f"Class weights: {class_weights_dict}")

    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

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

    model.save("model/audio_classification_model.keras")
    print("\nModel saved to 'model/audio_classification_model.keras'")

if __name__ == "__main__":
    main()