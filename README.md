# Audio Sample Organizer and Classifier

This project provides a set of scripts to organize and classify audio samples into categorized folders based on whether they are loops or oneshots. It utilizes a combination of substring matching and a trained neural network model to enhance the confidence of the sorting process. The model is capable of distinguishing between loops and oneshots as separate classes, ensuring accurate categorization of your audio samples.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Organizing Audio Samples](#organizing-audio-samples)
- [Usage](#usage)
- [Logging](#logging)
- [Notes](#notes)
- [License](#license)

## Features

- **Audio Classification Model**: Trains a neural network model to classify audio samples into categories, treating loops and oneshots as separate classes.
- **Sample Organization**: Organizes audio samples into a structured folder hierarchy based on their classification.
- **Key and BPM Detection**: Detects the musical key and BPM (Beats Per Minute) of tonal loops and appends this information to the filenames.
- **Substring Matching**: Uses filename and directory substrings to aid in categorization.
- **Error Handling and Logging**: Provides detailed logging and error handling throughout the process.

## Prerequisites

- Python 3.7 or higher
- **Python Packages**:
  - `numpy`
  - `librosa`
  - `tensorflow`
  - `scikit-learn`
  - `matplotlib`
  - `essentia`
  - `tqdm`
  - `pickle`
  - `argparse`
  - `logging`

**Note**: Essentia requires additional steps for installation. Refer to the [Essentia installation guide](https://essentia.upf.edu/installing.html) for detailed instructions.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/audio-sample-organizer.git
   cd audio-sample-organizer
   ```

2. **Create a Virtual Environment (Recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

   If you don't have a `requirements.txt` file, install the packages individually:
   ```bash
   pip install numpy librosa tensorflow scikit-learn matplotlib essentia tqdm
   ```


## Dataset Preparation

Organize your dataset in the following structure:

```
/dataset
  ├── train
      ├── Loops
          ├── Drums
              ├── Kick
                  ├── sample1.wav
                  ├── sample2.wav
                  └── ...
              ├── Snare
              └── ...
          ├── Sounds
              ├── Bass
              ├── Chords
              └── ...
      ├── Oneshots
          ├── Drums
              ├── Kick
              ├── Snare
              └── ...
          ├── Sounds
              ├── Bass
              ├── Chords
              └── ...
```

- **Loops**: Contains loop samples.
- **Oneshots**: Contains oneshot samples.
- **Categories**: Further divided into `Drums` and `Sounds`, and then into specific categories like `Kick`, `Snare`, `Bass`, `Chords`, etc.

**Important**: Ensure that the directory names match the categories defined in the scripts.

## Training the Model

The model distinguishes between loops and oneshots as separate classes.

1. **Update the `DATASET_PATH`**:

   In the training script (`train_model.py`), update the `DATASET_PATH` variable to point to your training data:

   ```python
   DATASET_PATH = "/path/to/your/dataset/train"
   ```

2. **Run the Training Script**:

   ```bash
   python train_model.py
   ```

   This script will:

   - Load and augment the audio data.
   - Extract features (MFCCs) from the audio samples.
   - Train a convolutional neural network model.
   - Evaluate the model and display training statistics.
   - Save the trained model as `audio_classification_model.keras` and the label encoder as `label_encoder.pkl`.

3. **Model Output**:

   - `audio_classification_model.keras`: The trained Keras model.
   - `label_encoder.pkl`: The label encoder used to encode the class labels.

## Organizing Audio Samples

Use the organizing script to categorize and organize your audio samples.

1. **Ensure Model and Encoder are Present**:

   Place `audio_classification_model.keras` and `label_encoder.pkl` in the same directory as the organizing script (`organize_samples.py`).

2. **Run the Organizing Script**:

   ```bash
   python organize_samples.py /path/to/input_folder /path/to/output_folder
   ```

   - `/path/to/input_folder`: The directory containing the audio samples you wish to organize.
   - `/path/to/output_folder`: The directory where the organized samples will be stored.

   The script will:

   - Analyze each audio file to determine if it is a loop or oneshot.
   - Use substring matching and the trained model to categorize the sample.
   - Detect the key and BPM (for tonal loops) and append this information to the filename.
   - Organize the samples into the appropriate folders.

## Usage

### Training Script (`train_model.py`)

- **Description**: Trains a neural network model for classifying audio samples.

- **Usage**:

  1. **Set the Dataset Path**:
     ```python
     DATASET_PATH = "/path/to/your/dataset/train"
     ```

  2. **Run the Script**:
     ```bash
     python train_model.py
     ```

  3. **Monitor Training**:
     - Training and validation accuracy and loss will be displayed.
     - After training, evaluation metrics and a confusion matrix will be shown.

  4. **Model and Encoder Output**:
     - The trained model and label encoder are saved for use in the organizing script.

### Organizing Script (`classify.py`)

- **Description**: Organizes audio samples into categorized folders using the trained model and substring matching.

- **Usage**:

  ```bash
  python classify.py /path/to/input_folder /path/to/output_folder
  ```

- **Arguments**:
  - `input_folder`: Path to the folder containing audio samples to organize.
  - `output_folder`: Path to the folder where organized samples will be stored.

- **Process**:
  - The script analyzes each audio file to determine if it is a loop or oneshot.
  - Categorizes the file based on substring matching and model prediction.
  - If both methods agree, that category is used.
  - If they disagree, the method with higher confidence is used.
  - Appends key and BPM information to the filenames for tonal loops.
  - Copies the files to the appropriate folders in the output directory.

## Logging

- The organizing script generates a log file named `organize_samples.log`.
- The log contains detailed information about the categorization process, including:
  - Files processed.
  - Model predictions and confidence levels.
  - Decisions made when categorization methods disagree.
  - Errors and warnings.

## Notes

- **Model Accuracy**: The accuracy of the model depends on the quality and quantity of your training data. Ensure that you have a balanced dataset with enough samples for each class.

- **Essentia Installation**: Essentia can be challenging to install. If you're having trouble, consider using the pre-built binaries or Docker images provided by the Essentia team.

- **Adjusting Confidence Threshold**: In the organizing script, you can adjust the `CONFIDENCE_THRESHOLD` variable to control how much you trust the model's predictions over substring matching.

- **Custom Categories**: If you wish to add more categories, update the `filename_substrings`, `LOOP_MAPPING`, and `ONESHOT_MAPPING` dictionaries in both scripts accordingly.

- **Dependencies**: Ensure that all dependencies are installed and compatible with your system. Some packages may have specific system requirements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Contact Information**:

For any questions or issues, please contact [your_email@example.com](mailto:jannik.assfalg@gmail.com).

**GitHub Repository**:

[https://github.com/yourusername/audio-sample-organizer](https://github.com/yourusername/audio-sample-organizer)