import os
import argparse
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, correlate
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

def normalize_audio(y):
    """
    Normalize the audio to 0 dBFS.
    """
    max_val = np.max(np.abs(y))
    if max_val == 0:
        return y
    return y / max_val

def detect_transients(y, sr, buffer_ms=50, hop_length=512, threshold=0.7):
    """
    Detect transients in the audio signal based on the normalized onset strength envelope.
    Only detect a transient if it is spaced out by at least buffer_ms milliseconds.
    Threshold is relative to the maximum onset strength.

    Returns:
        transients (list): List of transient times in seconds.
    """
    # Compute onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Normalize onset envelope
    onset_env = onset_env / np.max(onset_env) if np.max(onset_env) != 0 else onset_env
    # Find peaks in the onset envelope
    distance_in_frames = int(buffer_ms * sr / (1000 * hop_length))
    peak_indices, _ = find_peaks(onset_env, height=threshold, distance=distance_in_frames)
    # Convert peak indices to time
    transient_times = librosa.frames_to_time(peak_indices, sr=sr, hop_length=hop_length)
    return transient_times

def is_loop(y, sr, buffer_ms=50, hop_length=512):
    """
    Analyze the audio features to determine if it's a loop or one-shot.
    Returns a tuple (is_loop_flag, transient_count, duration, transients_detected, autocorr_peaks_times, features).
    """
    # Feature Extraction

    # 1. Duration Analysis
    duration = librosa.get_duration(y=y, sr=sr)

    # 2. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # 3. Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness_mean = np.mean(spectral_flatness)

    # 4. Repetition Detection via Auto-Correlation
    autocorr = correlate(y, y, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Find peaks in autocorrelation above 30% of the maximum autocorr value
    peak_height = np.max(autocorr) * 0.3
    autocorr_peaks, _ = find_peaks(autocorr, height=peak_height)

    has_repetition = len(autocorr_peaks) > 1

    # 5. Envelope Shape / Transient Peaks
    transients_detected = detect_transients(y, sr, buffer_ms=buffer_ms, hop_length=hop_length)
    transient_count = len(transients_detected)

    # Analyze transient periodicity
    is_periodic = analyze_transient_periodicity(transients_detected)

    # Loops typically have more transients
    is_peaky = transient_count > 2

    # 6. Root Mean Square (RMS) Amplitude
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    # **Transient Count Rule**
    # Samples with less than 3 transients are always One-Shots
    if transient_count < 6:
        return False, transient_count, duration, transients_detected, autocorr_peaks, {
            'zcr_mean': zcr_mean,
            'spectral_flatness_mean': spectral_flatness_mean,
            'rms_mean': rms_mean
        }

    # Classification Logic

    # **Loop Classification**
    if is_long_loop(duration) and (has_repetition or is_periodic) and \
       spectral_flatness_mean < 0.2 and zcr_mean < 0.08 and rms_mean > 0.03 and is_peaky:
        return True, transient_count, duration, transients_detected, autocorr_peaks, {
            'zcr_mean': zcr_mean,
            'spectral_flatness_mean': spectral_flatness_mean,
            'rms_mean': rms_mean
        }

    # **Fallback: Scoring System with Adjusted Weights**
    score_loop = 0
    score_one_shot = 0

    # Loop features
    if is_long_loop(duration):
        score_loop += 1
    if has_repetition or is_periodic:
        score_loop += 1
    if spectral_flatness_mean < 0.2:
        score_loop += 1
    if zcr_mean < 0.08:
        score_loop += 1
    if rms_mean > 0.03:
        score_loop += 1
    if is_peaky:
        score_loop += 1

    # One-Shot features
    if not is_long_loop(duration):
        score_one_shot += 1
    if not (has_repetition or is_periodic):
        score_one_shot += 1
    if spectral_flatness_mean > 0.2:
        score_one_shot += 1
    if zcr_mean > 0.08:
        score_one_shot += 1
    if rms_mean < 0.03:
        score_one_shot += 1
    if not is_peaky:
        score_one_shot += 1

    # Decide based on higher score
    if score_loop > score_one_shot:
        classification = True
    else:
        classification = False

    return classification, transient_count, duration, transients_detected, autocorr_peaks, {
        'zcr_mean': zcr_mean,
        'spectral_flatness_mean': spectral_flatness_mean,
        'rms_mean': rms_mean
    }

def is_long_loop(duration, threshold=2.0):
    """
    Determine if the duration qualifies as a long loop.
    """
    return duration > threshold

def analyze_transient_periodicity(transient_times, threshold=0.2):
    """
    Analyze the intervals between transients to determine if they are periodic.
    Returns True if transients occur at regular intervals.
    """
    if len(transient_times) < 3:
        return False  # Not enough transients to analyze periodicity

    intervals = np.diff(transient_times)
    mean_interval = np.mean(intervals)
    std_interval = np.std(intervals)

    # If the standard deviation is small relative to the mean, intervals are consistent
    return (std_interval / mean_interval) < threshold

def plot_visualization(file_path, y, sr, y_normalized, transients, autocorr, autocorr_peaks, classification, output_dir, input_folder):
    """
    Generate and save visualization plots for the audio file.
    """
    plt.figure(figsize=(14, 10))

    # 1. Waveform and Normalization
    plt.subplot(3, 1, 1)
    time = np.linspace(0, len(y) / sr, num=len(y))
    plt.plot(time, y, label='Original', alpha=0.5)
    plt.plot(time, y_normalized, label='Normalized', alpha=0.8)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # 2. Transient Detection
    plt.subplot(3, 1, 2)
    plt.plot(time, y_normalized, label='Normalized Waveform')
    plt.vlines(transients, ymin=min(y_normalized), ymax=max(y_normalized), color='r', alpha=0.8, label='Transients')
    plt.title('Transient Detection')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    # 3. Autocorrelation
    plt.subplot(3, 1, 3)
    autocorr_time = np.linspace(0, len(autocorr) / sr, num=len(autocorr))
    plt.plot(autocorr_time, autocorr, label='Autocorrelation')
    autocorr_peaks_times = autocorr_peaks / sr  # Convert sample indices to time
    plt.plot(autocorr_peaks_times, autocorr[autocorr_peaks], 'x', color='red', label='Autocorr Peaks')
    plt.title('Autocorrelation')
    plt.xlabel('Lag Time (s)')
    plt.ylabel('Autocorrelation')
    plt.legend()

    # Add Classification Text
    plt.suptitle(f'File: {os.path.basename(file_path)} | Classification: {"Loop" if classification else "One-Shot"}', fontsize=16, y=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Prepare output path
    relative_path = os.path.relpath(file_path, start=input_folder)
    plot_filename = os.path.splitext(relative_path.replace(os.sep, '_'))[0] + '.png'
    plot_path = os.path.join(output_dir, plot_filename)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Save the plot
    plt.savefig(plot_path)
    plt.close()

def process_audio_file(file_path, logger, buffer_ms=50, hop_length=512):
    """
    Process a single audio file: normalize, classify, and return necessary data.
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=None, mono=True)

        # Normalize audio to 0 dBFS
        y_normalized = normalize_audio(y)

        # Classify
        is_loop_flag, transient_count, duration, transients_detected, autocorr_peaks, features = is_loop(
            y_normalized, sr, buffer_ms=buffer_ms, hop_length=hop_length
        )

        classification = "Loop" if is_loop_flag else "One-Shot"
        logger.info(
            f"{file_path} | Classification: {classification} | Transients: {transient_count} | "
            f"Duration: {duration:.2f}s | ZCR: {features['zcr_mean']:.6f} | "
            f"Spectral Flatness: {features['spectral_flatness_mean']:.6f} | RMS: {features['rms_mean']:.6f}"
        )

        # Compute autocorrelation for plotting
        autocorr = correlate(y_normalized, y_normalized, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        # Find peaks in autocorrelation above 30% of the maximum autocorr value
        peak_height = np.max(autocorr) * 0.3
        autocorr_peaks_plot, _ = find_peaks(autocorr, height=peak_height)

        return is_loop_flag, features, transient_count, y, sr, y_normalized, transients_detected, autocorr, autocorr_peaks_plot
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None, None, None, None, None, None, None, None, None

def setup_logging(output_file):
    """
    Set up logging to the specified output file.
    """
    logging.basicConfig(
        filename=output_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    return logging.getLogger()

def setup_mismatches_logging(mismatches_file):
    """
    Set up logging for mismatched files to a separate file.
    """
    mismatches_logger = logging.getLogger('mismatches')
    mismatches_logger.setLevel(logging.INFO)
    # Create file handler
    fh = logging.FileHandler(mismatches_file, mode='w')
    fh.setLevel(logging.INFO)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # Add the handler to the logger
    mismatches_logger.addHandler(fh)
    return mismatches_logger

def get_ground_truth_label(file_path):
    """
    Extracts the ground truth label from the file path based on 'loop' or 'oneshot' substring.
    """
    file_path_lower = file_path.lower()
    if 'loop' in file_path_lower:
        return 'Loop'
    elif 'oneshot' in file_path_lower or 'one-shot' in file_path_lower:
        return 'One-Shot'
    else:
        return None  # Unknown label

def main():
    """
    Main function to parse arguments and process audio files.
    """
    parser = argparse.ArgumentParser(description="Classify audio files as Loops or One-Shots with Visualization of Mismatches.")
    parser.add_argument('input_folder', type=str, help='Path to the input folder containing audio files.')
    parser.add_argument('--output', type=str, default='output.log', help='Path to the output log file.')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save visualization plots.')
    parser.add_argument('--buffer_ms', type=int, default=50, help='Buffer time in milliseconds between transients.')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length for transient detection.')
    parser.add_argument('--mismatches_log', type=str, default='mismatches.log', help='File to log mismatched files and their features.')
    args = parser.parse_args()

    logger = setup_logging(args.output)
    mismatches_logger = setup_mismatches_logging(args.mismatches_log)

    logger.info("Starting audio classification...")

    supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.aiff', '.aac', '.wma', '.alac')

    # Collect all supported audio files
    audio_files = []
    for root, dirs, files in os.walk(args.input_folder):
        for file in files:
            if file.lower().endswith(supported_extensions):
                file_path = os.path.join(root, file)
                audio_files.append(file_path)

    if not audio_files:
        logger.warning("No audio files found in the specified input folder.")
        print("No audio files found in the specified input folder.")
        return

    # Set a base input folder for plotting
    plot_visualization_base_input = args.input_folder
    # Ensure the plot directory exists
    os.makedirs(args.plot_dir, exist_ok=True)

    # Initialize counters for summary
    loop_count = 0
    one_shot_count = 0
    error_count = 0
    correct_classifications = 0
    unknown_label_count = 0
    total_files = len(audio_files)

    # Process files with a progress bar
    for file_path in tqdm(audio_files, desc="Processing Audio Files", unit="file"):
        classification, features, transient_count, y, sr, y_normalized, transients_detected, autocorr, autocorr_peaks = process_audio_file(
            file_path=file_path,
            logger=logger,
            buffer_ms=args.buffer_ms,
            hop_length=args.hop_length
        )

        if classification is None:
            error_count += 1
            continue  # Skip further processing if there was an error

        predicted_label = "Loop" if classification else "One-Shot"

        if classification:
            loop_count +=1
        else:
            one_shot_count +=1

        # Extract ground truth label from file path
        ground_truth_label = get_ground_truth_label(file_path)

        if ground_truth_label is not None:
            if predicted_label == ground_truth_label:
                correct_classifications += 1
            else:
                # Mismatched file
                # Log mismatched file, features, and number of transients
                mismatches_logger.info(
                    f"{file_path} | Ground Truth: {ground_truth_label} | Predicted: {predicted_label} | "
                    f"Transients: {transient_count} | Features: {features}"
                )

                # Plot and save visualization for mismatched file
                plot_visualization(
                    file_path=file_path,
                    y=y,
                    sr=sr,
                    y_normalized=y_normalized,
                    transients=transients_detected,
                    autocorr=autocorr,
                    autocorr_peaks=autocorr_peaks,
                    classification=classification,
                    output_dir=args.plot_dir,
                    input_folder=plot_visualization_base_input
                )
        else:
            unknown_label_count +=1

    total_files_with_known_labels = total_files - unknown_label_count
    if total_files_with_known_labels > 0:
        accuracy = correct_classifications / total_files_with_known_labels * 100
    else:
        accuracy = 0.0

    # Log summary
    logger.info("========== Summary ==========")
    logger.info(f"Total Files Processed: {total_files}")
    logger.info(f"Total Files with Known Labels: {total_files_with_known_labels}")
    logger.info(f"Total Loops: {loop_count}")
    logger.info(f"Total One-Shots: {one_shot_count}")
    logger.info(f"Total Errors: {error_count}")
    logger.info(f"Total Unknown Labels: {unknown_label_count}")
    logger.info(f"Correct Classifications: {correct_classifications}")
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info("Audio classification completed.")

    # Print summary to console
    print("\nAudio classification completed.")
    print("============================")
    print(f"Total Files Processed: {total_files}")
    print(f"Total Files with Known Labels: {total_files_with_known_labels}")
    print(f"Total Loops: {loop_count}")
    print(f"Total One-Shots: {one_shot_count}")
    print(f"Total Errors: {error_count}")
    print(f"Total Unknown Labels: {unknown_label_count}")
    print(f"Correct Classifications: {correct_classifications}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Visualization plots for mismatched files saved to: {args.plot_dir}")
    print(f"Detailed log saved to: {args.output}")
    print(f"Mismatched files log saved to: {args.mismatches_log}")

if __name__ == "__main__":
    main()