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

# --------------------------- Parameters ---------------------------

DATASET_PATH = "/Users/jannikassfalg/coding/sample_sorter/dataset/train"  # Update this path as needed
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
    '.aac', '.wma', '.m4a', '.alac', '.opus', '.mid', '.midi'
)

# ----------------------- Data Loading -----------------------

def augment_audio(signal, sample_rate):
    try:
        # Time Stretch
        stretch_rate = np.random.uniform(0.8, 1.2)
        stretched_signal = librosa.effects.time_stretch(signal, stretch_rate)
        
        # Pitch Shift
        n_steps = np.random.randint(-5, 5)
        pitched_signal = librosa.effects.pitch_shift(stretched_signal, sample_rate, n_steps)
        
        # Add Noise
        noise = np.random.randn(len(pitched_signal))
        augmented_signal = pitched_signal + 0.005 * noise
        
        # Volume Control
        augmented_signal = augmented_signal * np.random.uniform(0.8, 1.2)
        
        # Time Cropping or Padding
        if len(augmented_signal) > SAMPLES_PER_TRACK:
            start = np.random.randint(0, len(augmented_signal) - SAMPLES_PER_TRACK)
            augmented_signal = augmented_signal[start:start + SAMPLES_PER_TRACK]
        else:
            pad_width = SAMPLES_PER_TRACK - len(augmented_signal)
            augmented_signal = np.pad(augmented_signal, (0, pad_width), 'constant')
        
        return augmented_signal
    except Exception as e:
        print(f"Error during augmentation: {e}")
        return None  # Return None to indicate failure

def load_audio_files(dataset_path, sample_rate, duration, audio_extensions, augment=False):
    X = []
    labels = []
    max_len = sample_rate * duration
    success_count = 0

    # Traverse the directory structure
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(audio_extensions):
                file_path = os.path.join(root, file)
                # Extract label based on the leaf folder
                label = os.path.basename(root)
                
                # Load audio file
                try:
                    signal, sr = librosa.load(file_path, sr=sample_rate)
                    if len(signal) > max_len:
                        signal = signal[:max_len]
                    else:
                        pad_width = max_len - len(signal)
                        signal = np.pad(signal, (0, pad_width), 'constant')
                    
                    if augment:
                        augmented_signal = augment_audio(signal, sr)
                        if augmented_signal is not None:
                            signal = augmented_signal
                        else:
                            print(f"Skipping augmentation for {file_path} due to previous error.")
                    
                    X.append(signal)
                    labels.append(label)
                    success_count += 1
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    print(f"{'Augmented' if augment else 'Loaded'} {success_count} samples successfully.")
    return np.array(X), np.array(labels)

print("Loading audio files...")
# Load original data
X, labels = load_audio_files(DATASET_PATH, SAMPLE_RATE, DURATION, AUDIO_EXTENSIONS, augment=False)
# Load augmented data
augmented_X, augmented_labels = load_audio_files(DATASET_PATH, SAMPLE_RATE, DURATION, AUDIO_EXTENSIONS, augment=True)

# Verify dimensions and concatenate if possible
print("\nVerifying and concatenating augmented data...")
if augmented_X.size > 0 and augmented_X.ndim == 2 and X.ndim == 2:
    X = np.concatenate((X, augmented_X), axis=0)
    labels = np.concatenate((labels, augmented_labels), axis=0)
    print(f"Combined dataset size: {X.shape[0]} samples.")
else:
    print("No augmented data to concatenate or inconsistent dimensions.")

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
print(f"Classes removed (<{ MIN_SAMPLES} samples): {removed_classes}")

# Filter out samples from classes with fewer than MIN_SAMPLES
filtered_indices = [i for i, label in enumerate(labels) if label in classes_to_keep]
X_filtered = X[filtered_indices]
y_filtered = y_encoded[filtered_indices]
filtered_labels = labels[filtered_indices]

print(f"Filtered dataset size: {X_filtered.shape[0]} samples")

# Re-encode labels after filtering
le_filtered = LabelEncoder()
y_encoded_filtered = le_filtered.fit_transform(filtered_labels)
y_categorical = to_categorical(y_encoded_filtered)
num_classes_filtered = y_categorical.shape[1]
print(f"Number of classes after filtering: {num_classes_filtered}")

# Print filtered class distribution
print("\nClass distribution after filtering:")
filtered_class_counts = Counter(filtered_labels)
for label, count in filtered_class_counts.items():
    print(f"{label}: {count} samples")

# -------------------- Feature Extraction --------------------

def extract_features(X, sample_rate, n_mels, hop_length, n_fft, n_mfcc=13):
    mfccs = []
    for x in X:
        try:
            mfcc = librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=n_mfcc, 
                                        n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
            mfcc = librosa.power_to_db(mfcc, ref=np.max)
            mfccs.append(mfcc)
        except Exception as e:
            print(f"Error extracting features: {e}")
            mfccs.append(np.zeros((n_mfcc, int(np.ceil(len(x)/hop_length))), dtype=np.float32))  # Placeholder
    return np.array(mfccs)

print("\nExtracting features...")
X_features = extract_features(X_filtered, SAMPLE_RATE, N_MELS, HOP_LENGTH, N_FFT, N_MFCC)
X_features = X_features[..., np.newaxis]  # Add channel dimension
print(f"Feature shape: {X_features.shape}")

# -------------------- Train-Test Split --------------------

print("\nSplitting data into train and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(
    X_features, y_categorical, test_size=VALIDATION_SPLIT, random_state=42, stratify=y_categorical
)
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
model = build_model(input_shape, num_classes_filtered)
model.summary()

# Save the label encoder
with open("label_encoder.pkl", "wb") as le_file:
    pickle.dump(le_filtered, le_file)
print("Label encoder saved to 'label_encoder.pkl'")

# Print the class labels
print("Model Class Labels:")
print(le_filtered.classes_)

# -------------------- Compilation --------------------

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------- Callbacks --------------------

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# -------------------- Calculate Class Weights --------------------

print("\nCalculating class weights...")
class_weights_values = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded_filtered),
    y=y_encoded_filtered
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
y_true = np.argmax(y_val, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=le_filtered.classes_, zero_division=0))

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
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(y_true, y_pred_classes, le_filtered.classes_)

# -------------------- Save the Model --------------------

model.save("audio_classification_model.keras")  # Updated extension
print("\nModel saved to 'audio_classification_model.keras'")