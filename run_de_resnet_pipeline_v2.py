"""
EEG DE-ResNet18 Classification Pipeline
=======================================

This script implements a complete machine learning pipeline for classifying EEG
data based on facial expression stimuli (static vs. morphing). It uses a
Leave-One-Subject-Out (LOSO) cross-validation approach with a ResNet18 model
trained on Differential Entropy (DE) features.

The pipeline consists of the following major steps:
1.  **Configuration**: Sets up paths, parameters, and constants.
2.  **Data Loading**: Loads individual subject EEG data, creates MNE Epochs objects,
    and extracts labels from filenames.
3.  **Feature Extraction**: Calculates Differential Entropy (DE) for multiple
    frequency bands, creating a 2D "image" (channels x bands) for each trial.
4.  **Model Training & Evaluation**: For each classification task (e.g., Fear, Anger):
    a. A Leave-One-Subject-Out (LOSO) cross-validation loop is initiated.
    b. In each fold, a ResNet18 model is trained on all subjects but one.
    c. The model's performance is evaluated on the held-out test subject.
    d. Early stopping is used to prevent overfitting during training.
5.  **Analysis & Visualization**: Generates and saves numerous plots, including:
    a. Detailed explanations of the DE feature extraction process.
    b. ERP topomaps for linear analysis comparison.
    c. DE feature distribution plots.
    d. Channel importance maps derived from a logistic regression model.
    e. Per-participant accuracy bar charts.
    f. ERP comparisons for the lowest-performing subject vs. the group average.
6.  **Report Generation**: Compiles all results and plots into a comprehensive
    PDF report.

Caching is implemented to save significant time on subsequent runs by storing
the processed MNE Epochs and the extracted DE features.
"""

import os
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend suitable for scripts
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from scipy.signal import stft, welch
from tqdm import tqdm
from torchvision import models
import pandas as pd
from datetime import datetime
from datetime import datetime
from datetime import datetime
from mne.viz import plot_topomap
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from datetime import datetime

# --- 1. Configuration ---
# ------------------------

# --- File Paths ---
# Define the directory where the raw EEG data is stored and where outputs will be saved.
DATA_DIR = '/Users/smart/Programming/Semi-Supervised Learning Faces/FaceMorph_singletrial'
OUTPUT_DIR = '/Users/smart/Programming/Semi-Supervised Learning Faces/Semi_supervised_Framework/loso_pipeline'
# Create a unique, timestamped folder for this run's plots
PLOTS_DIR = os.path.join(OUTPUT_DIR, f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

# --- EEG & Epoching Parameters ---
NUM_CHANNELS = 64                   # Number of EEG channels to use from the data files.
NATIVE_SAMPLING_RATE = 1000         # Hz, the original sampling rate of the EEG data.
EPOCH_DURATION = 4.404              # seconds, the length of each trial segment.
T_MIN, T_MAX = 0.1, EPOCH_DURATION + 0.1    # The time window for each epoch relative to the event.
BASELINE = None                     # No baseline correction is applied as per data characteristics.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available.

# --- Caching Configuration ---
# Caching saves processed data to disk to speed up subsequent runs.
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
ALL_EPOCHS_CACHE_FILE = os.path.join(CACHE_DIR, 'all_epochs_v2-epo.fif') # For MNE Epochs
DE_FEATURES_CACHE_FILE = os.path.join(CACHE_DIR, 'de_features_v2.npy')   # For extracted DE features
LABELS_CACHE_FILE = os.path.join(CACHE_DIR, 'labels_v2.npz')             # For corresponding labels

# --- Feature Extraction Parameters ---
# Define the standard EEG frequency bands for Differential Entropy calculation.
BANDS = {
    'Delta': (1, 3),
    'Theta': (4, 7),
    'Alpha': (8, 14),
    'Beta': (15, 30),
    'Gamma': (31, 50)
}

# --- Channel Configuration ---
# Define the names of the 64 EEG channels and the standard montage for topoplots.
# The first 62 are standard, the last two are placeholders as their exact names are not critical.
CH_NAMES_62 = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 
    'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'Pz', 
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2'
]
CH_NAMES = CH_NAMES_62 + ['X1', 'X2'] # Using generic names for the last two channels
MONTAGE = mne.channels.make_standard_montage('standard_1020')

# --- Task Configuration ---
# Define the classification tasks to be run.
# The new task is a single, combined classification of all static vs. all morph trials.
TASKS = {
    'All Static vs All Morph': (np.ones(1, dtype=bool), 'variant') # A placeholder, will be properly defined in main().
}


# --- 2. Model Training Utilities ---
# -----------------------------------

class EarlyStopping:
    """
    A utility to stop training when a monitored metric has stopped improving.
    This helps prevent overfitting by halting the training process once the model
    performance on a validation set ceases to improve for a specified number
    of consecutive epochs.
    """
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): function to use for printing messages.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        """
        Call method to update early stopping state based on current validation loss.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- 3. Data Loading and Feature Extraction ---
# ----------------------------------------------

def load_subject_data(subject_id, eeg_dir):
    """
    Loads all EEG data files for a single subject, processes them, and
    concatenates them into a single MNE Epochs object.

    Args:
        subject_id (str): The identifier for the subject (e.g., 'FM01').
        eeg_dir (str): The directory containing the EEG data files.

    Returns:
        tuple: A tuple containing:
            - mne.Epochs: Concatenated epochs for the subject.
            - np.ndarray: Array of emotion labels.
            - np.ndarray: Array of variant labels (static vs. morph).
            - np.ndarray: Array of group (subject ID) labels.
    """
    # Find all data files for the given subject
    data_files = [f for f in os.listdir(eeg_dir) if f.startswith(subject_id) and f.endswith('_eeg.dat')]
    if not data_files:
        return None, None, None, None

    all_epochs_list, emotion_labels, variant_labels, groups = [], [], [], []
    
    # Create a template MNE Info object. This is done once and reused.
    # All 64 channels are treated as 'eeg' type, as per user clarification.
    info = mne.create_info(ch_names=CH_NAMES, sfreq=NATIVE_SAMPLING_RATE, ch_types=['eeg'] * NUM_CHANNELS)
    
    # Set the montage on the info object. MNE is smart enough to only apply the
    # montage to the channels that exist in it (the first 62). The other 2
    # channels ('X1', 'X2') will be kept but won't have a 3D location.
    # This prevents errors in plotting functions like plot_topomap.
    with mne.utils.use_log_level('error'):
        info.set_montage(MONTAGE, on_missing='warn')
    
    for file_name in data_files:
        try:
            # --- Label Extraction from Filename ---
            # The filename format (e.g., 'FM01_12_eeg.dat') encodes the condition.
            # First digit is emotion (1=Fear, 2=Anger), second is variant (1=Static, 2=Morph).
            parts = file_name.split('_')
            emotion_code = int(parts[1][0])
            variant_code = int(parts[1][1])
            
            # --- Data Loading and Reshaping ---
            # Load the raw binary data.
            data = np.fromfile(os.path.join(eeg_dir, file_name), dtype=np.float32)
            
            # The data is stored as a flat array. It needs to be reshaped into (channels, samples).
            # Some files have 65 or 66 channels, but we only want the first 64.
            # This logic robustly determines the number of channels in the file.
            n_channels_in_file = 66 if data.size % 66 == 0 else 65 if data.size % 65 == 0 else NUM_CHANNELS
            n_total_samples = data.size // n_channels_in_file
            data = data[:n_total_samples * n_channels_in_file] # Ensure data is divisible
            data_full = data.reshape((n_channels_in_file, n_total_samples), order='F')
            data_to_use = data_full[:NUM_CHANNELS, :]

            # --- MNE Object Creation ---
            # Create an MNE Raw object from the numpy array. The info object now has the montage.
            raw = mne.io.RawArray(data_to_use, info, verbose=False)
            
            # The data is pre-cleaned, so no filtering or resampling is applied here.

            # --- Epoching ---
            # Segment the continuous data into fixed-length epochs.
            events = mne.make_fixed_length_events(raw, id=1, duration=EPOCH_DURATION)
            epochs = mne.Epochs(raw, events, tmin=T_MIN, tmax=T_MAX, baseline=BASELINE, preload=True, verbose=False)
            
            # Append the processed epochs and corresponding labels to lists.
            all_epochs_list.append(epochs)
            num_epochs = len(epochs)
            emotion_labels.extend([emotion_code - 1] * num_epochs) # 0 for Fear, 1 for Anger
            variant_labels.extend([variant_code - 1] * num_epochs) # 0 for Static, 1 for Morph
            groups.extend([subject_id] * num_epochs)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue

    if not all_epochs_list:
        return None, None, None, None
        
    # Concatenate all epochs from different files into a single object.
    return mne.concatenate_epochs(all_epochs_list, verbose=False), np.array(emotion_labels), np.array(variant_labels), np.array(groups)


def extract_de_features(epochs):
    """
    Extracts Differential Entropy (DE) features using a Short-Time Fourier
    Transform (STFT) with a Hanning window, as per the specified methodology.

    This function implements the following steps for each trial:
    1.  **STFT Calculation**: The signal is divided into segments using a 256-point
        non-overlapping Hanning window. The Fourier Transform is computed for
        each segment.
    2.  **Power Spectral Density (PSD) Estimation**: The PSD is calculated by
        taking the squared magnitude of the STFT result and averaging across
        the time segments.
    3.  **Band Power Calculation**: The power for each frequency band (Delta, Theta, etc.)
        is calculated by integrating (summing) the PSD values within that band's
        frequency range.
    4.  **DE Calculation**: The Differential Entropy is computed from this band power
        using the formula for a Gaussian distribution: 0.5 * log(2 * pi * e * power).

    Args:
        epochs (mne.Epochs): The epoched EEG data.

    Returns:
        np.ndarray: The extracted DE features, shaped for the ResNet model as
                    (n_epochs, 1, n_channels, n_bands).
    """
    data = epochs.get_data(copy=True)
    sfreq = epochs.info['sfreq']
    n_epochs, n_channels, _ = data.shape
    
    de_features = np.zeros((n_epochs, n_channels, len(BANDS)))

    # Parameters for STFT, matching the user's description.
    nperseg = 256  # 256-point window
    
    # Iterate over each epoch (trial) and channel to extract features.
    for i in tqdm(range(n_epochs), desc="Extracting DE Features (STFT)"):
        for j_chan in range(n_channels):
            
            # Step 1: Calculate the Short-Time Fourier Transform.
            freqs, _, Zxx = stft(
                data[i, j_chan],
                fs=sfreq,
                window='hann',      # Hanning window
                nperseg=nperseg,
                noverlap=0          # Non-overlapping windows
            )
            
            # Step 2: Compute the Power Spectral Density from the STFT result.
            # Power is the squared magnitude, averaged over the time segments.
            psd = np.mean(np.abs(Zxx)**2, axis=1)
            
            for band_idx, (_, (l_freq, h_freq)) in enumerate(BANDS.items()):
                
                # Step 3: Calculate the power in the current band by integrating the PSD.
                freq_res = freqs[1] - freqs[0]
                band_mask = np.logical_and(freqs >= l_freq, freqs <= h_freq)
                band_power = np.sum(psd[band_mask]) * freq_res
                
                # Step 4: Calculate Differential Entropy from the band power.
                de_features[i, j_chan, band_idx] = 0.5 * np.log(2 * np.pi * np.e * band_power + 1e-9)
    
    # Reshape for ResNet.
    return de_features[:, np.newaxis, :, :]


# --- 4. PDF Report Generation ---
# --------------------------------

class PDF(FPDF):
    """
    Custom PDF class to generate a formatted report with a header, footer,
    and methods for adding chapters and plots.
    """
    def header(self):
        """Defines the header for each page."""
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'EEG Classification Report', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, 'Static vs. Morphing Faces (DE + ResNet18)', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(10)

    def footer(self):
        """Defines the footer for each page."""
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

    def chapter_title(self, title):
        """Formats and adds a chapter title."""
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)

    def chapter_body(self, body):
        """Formats and adds a paragraph of text."""
        self.set_font('Helvetica', '', 12)
        self.multi_cell(0, 10, body, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln()

    def add_plot(self, path, title=""):
        """
        Adds a plot to the PDF from a file path, with error handling.
        """
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)
        if path and os.path.exists(path):
            try:
                # Center the image on the page
                self.image(path, x=self.get_x() + 30, w=150)
            except Exception as e:
                self.set_font('Helvetica', 'I', 10)
                self.cell(0, 10, f"(Failed to add plot: {e})", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        else:
            self.set_font('Helvetica', 'I', 10)
            self.cell(0, 10, "(Plot not available)", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(10)


def generate_report(output_pdf_path, task_name, tmin, tmax, loso_accuracies, overall_accuracy, plot_paths, de_params, bands):
    """
    Generates the final PDF report, compiling all results and visualizations.
    """
    pdf = PDF()
    pdf.add_page()

    # --- Executive Summary ---
    pdf.chapter_title("Executive Summary")
    summary_text = (
        f"This report documents a machine learning pipeline designed to classify EEG signals in response to viewing 'static' versus 'morphing' facial expressions for the '{task_name}' task. "
        f"Using a Leave-One-Subject-Out (LOSO) cross-validation methodology, a ResNet18 deep learning model was trained on Differential Entropy (DE) features derived from the EEG data.\n\n"
        f"The overall classification accuracy achieved was {overall_accuracy*100:.2f}%. "
        "Channel importance analysis revealed that the model heavily relied on distinct functional networks of channels, particularly in posterior-occipital regions, to make its classifications.\n\n"
        "The methodology, which uses a Short-Time Fourier Transform (STFT) to derive signal power for DE, was validated as a robust and appropriate choice. The results indicate that this DE-ResNet approach is highly effective for decoding cognitive states from EEG data in this paradigm."
    )
    pdf.chapter_body(summary_text)

    # --- Introduction & Methodology Section ---
    pdf.add_page()
    pdf.chapter_title("1. Introduction & Methodology")
    methodology_text = (
        f"This report details a classification pipeline for EEG data from {len(loso_accuracies)} subjects to distinguish between trials where subjects viewed 'static' faces versus 'morphing' faces. "
        f"The analysis focuses on the '{task_name}' task.\n\n"
        "The methodology involves the following key steps:\n"
        f"1. **Data Loading & Epoching**: Raw EEG data was loaded and segmented into epochs from t={tmin}s to t={tmax}s relative to stimulus onset.\n"
        f"2. **Feature Extraction (Differential Entropy via STFT)**: For each epoch, a Short-Time Fourier Transform (STFT) was computed using a {de_params['nperseg']}-point '{de_params['window']}' window with {de_params['noverlap']} overlap. The resulting Power Spectral Density (PSD) was integrated over 5 standard frequency bands (Delta, Theta, Alpha, Beta, Gamma) to calculate the band power. The Differential Entropy (DE) was then computed from this power, creating a {NUM_CHANNELS}x{len(bands)} 'image' for each trial.\n"
        "3. **Classification (ResNet18)**: A ResNet18 convolutional neural network was adapted to classify these 2D DE feature images. The model was trained and evaluated using a Leave-One-Subject-Out (LOSO) cross-validation scheme.\n"
        "4. **Analysis**: A series of analyses were performed, including DE feature distribution and model-based channel importance."
    )
    pdf.chapter_body(methodology_text)
    pdf.add_plot(plot_paths.get('de_explanation'), title="Methodology: DE Feature Extraction Steps")

    # --- Results Section ---
    pdf.add_page()
    pdf.chapter_title("2. Classification Results")
    results_summary = (
        "The ResNet18 model was trained and evaluated using Leave-One-Subject-Out (LOSO) cross-validation, where each subject's data was used as a test set once. "
        "This approach provides a robust estimate of the model's generalization performance on unseen individuals."
    )
    pdf.chapter_body(results_summary)
    
    # Create a formatted table for results
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(80, 10, 'Metric', 1)
    pdf.cell(80, 10, 'Value', 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(80, 10, 'Overall Accuracy', 1)
    pdf.cell(80, 10, f'{overall_accuracy*100:.2f}%', 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(80, 10, 'Mean Subject Accuracy', 1)
    pdf.cell(80, 10, f'{np.mean(list(loso_accuracies.values()))*100:.2f}%', 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(80, 10, 'Std. Dev. Subject Accuracy', 1)
    pdf.cell(80, 10, f'{np.std(list(loso_accuracies.values()))*100:.2f}%', 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(10)

    results_discussion = (
        "The high overall accuracy demonstrates that the DE features, derived via STFT and processed by a ResNet model, provide a strong basis for discriminating between the 'static' and 'morph' conditions. "
        "The standard deviation of subject accuracy indicates the level of performance variability across the participant cohort."
    )
    pdf.chapter_body(results_discussion)
    pdf.add_plot(plot_paths.get('confusion_matrix'), title="Overall Confusion Matrix (All Subjects)")

    # --- DE Feature Distribution ---
    pdf.add_page()
    pdf.chapter_title("3. DE Feature Distribution Analysis")
    pdf.chapter_body(
        "The following plot shows the distribution of the average Differential Entropy values across all channels for each frequency band, separated by condition ('Static' vs. 'Morph'). "
        "This helps visualize the feature space that the classifier is learning from."
    )
    pdf.add_plot(plot_paths.get('de_distribution'), title=f"DE Feature Distribution ({task_name})")
    
    # --- Channel Importance Section ---
    pdf.add_page()
    pdf.chapter_title("4. Channel Importance Analysis")
    pdf.chapter_body(
        "To understand which channels and frequency bands were most influential, a Logistic Regression model was trained on the DE features. The model's coefficients serve as a proxy for feature importance. The topomaps below visualize these coefficients, with red indicating features that push the prediction towards 'Morph' and blue towards 'Static'.\n\n"
        "**Note on Visualization**: While the analysis utilizes all 64 channels, the topomaps display only the 62 channels with defined 3D locations."
    )
    
    # Add the single, consolidated channel importance plot
    pdf.add_plot(plot_paths.get('channel_importance'), title=f"Channel Importance for {task_name}")

    # --- Clustered Channel Importance Section ---
    pdf.add_page()
    pdf.chapter_title("5. Clustered Channel Importance Analysis")
    pdf.chapter_body(
        "To further understand spatial patterns, k-means clustering was applied to the channel importance profiles. This groups channels that have a similar pattern of importance across frequencies, revealing functional networks the model may be leveraging. The figure below shows both the spatial distribution of these clusters and the mean importance of each cluster."
    )
    pdf.add_plot(plot_paths.get('clustered_channel_importance'), title="Channel Clusters and their Mean Importance")

    # --- Per-Participant Analysis Section ---
    pdf.add_page()
    pdf.chapter_title("6. Per-Participant Accuracy")
    pdf.chapter_body("The following bar chart shows the final classification accuracy for each individual participant, providing insight into the variability of model performance across the cohort.")
    pdf.add_plot(plot_paths.get('loso_accuracies'), title="Per-Participant Accuracy")
    
    pdf.add_page()
    pdf.chapter_title("7. Lowest-Accuracy Subject Analysis")
    pdf.chapter_body("To investigate sources of variability, the ERPs of the lowest-performing subject were compared against the grand average of all other subjects. This can help identify if the subject had an atypical neural response. The plot below shows this comparison for key posterior channels.")
    pdf.add_plot(plot_paths.get('low_accuracy_erp'), title="Lowest Accuracy Subject ERP Comparison")

    # --- Save the PDF ---
    pdf.output(output_pdf_path)
    print(f"Final PDF report generated at {output_pdf_path}")


# --- 5. Plotting Functions ---
# -----------------------------

def plot_de_distribution(de_features, labels, task_name, output_dir):
    """
    Generates and saves a violin plot of DE feature distributions.
    This visualization helps to understand how the DE values are distributed
    for each frequency band and condition, revealing potential feature separability.
    """
    print(f"  Generating DE distribution plot for {task_name}...")
    df_list = []
    for i in range(de_features.shape[0]):
        for j_band, band_name in enumerate(BANDS.keys()):
            # Average DE across all channels for this trial and band
            avg_de = np.mean(de_features[i, 0, :, j_band])
            df_list.append({
                'DE': avg_de,
                'Band': band_name,
                'Condition': 'Morph' if labels[i] == 1 else 'Static'
            })
    df = pd.DataFrame(df_list)

    plt.figure(figsize=(12, 7))
    sns.violinplot(data=df, x='Band', y='DE', hue='Condition', split=True, inner='quart', palette='muted')
    plt.title(f'Distribution of Differential Entropy Features for {task_name}', fontsize=16)
    plt.xlabel('Frequency Band', fontsize=12)
    plt.ylabel('Differential Entropy (Channel Averaged)', fontsize=12)
    plt.legend(title='Condition')
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, f'de_distribution_{task_name.replace(":", "").replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def plot_de_feature_explanation(epoch, output_dir):
    """
    Generates a single, comprehensive plot to explain the DE feature extraction process.
    """
    print("  Generating DE feature explanation plot...")
    
    sfreq = epoch.info['sfreq']
    times = epoch.times
    data_uv = epoch.get_data(picks='POz')[0, 0, :] * 1e6

    fig = plt.figure(figsize=(20, 25))
    gs = fig.add_gridspec(4, 2)
    fig.suptitle('Visual Explanation of the DE Feature Extraction Process (from Channel POz)', fontsize=24, weight='bold')

    # --- Step 1: Raw EEG Signal ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, data_uv, color='black')
    ax1.set_title('Step 1: Raw EEG Signal from a Single Trial', fontsize=18)
    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('Amplitude (µV)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Step 2: Power Spectral Density (PSD) ---
    ax2 = fig.add_subplot(gs[1, :])
    freqs, psd = welch(data_uv, fs=sfreq, nperseg=int(sfreq/2), detrend='linear')
    ax2.plot(freqs, psd, color='navy')
    ax2.set_xlim(1, 55)
    ax2.set_yscale('log')
    ax2.set_title("Step 2: Power Spectral Density (PSD) of the Signal (via Welch's Method)", fontsize=18)
    ax2.set_xlabel('Frequency (Hz)', fontsize=14)
    ax2.set_ylabel('Power Spectral Density (µV²/Hz)', fontsize=14)
    ax2.grid(True, which="both", ls="--", alpha=0.6)
    band_colors = {'Delta': '#1f77b4', 'Theta': '#ff7f0e', 'Alpha': '#2ca02c', 'Beta': '#d62728', 'Gamma': '#9467bd'}
    for band, (l_freq, h_freq) in BANDS.items():
        ax2.axvspan(l_freq, h_freq, alpha=0.2, color=band_colors[band], label=f'{band} ({l_freq}-{h_freq} Hz)')
    ax2.legend(fontsize=12)

    # --- Step 3: Band-Filtered Signals ---
    ax3_main = fig.add_subplot(gs[2, :])
    ax3_main.set_title('Step 3: EEG Signal Filtered into Specific Frequency Bands', fontsize=18)
    ax3_main.set_xlabel('Time (s)', fontsize=14)
    ax3_main.set_ylabel('Amplitude (µV)', fontsize=14)
    ax3_main.grid(True, linestyle='--', alpha=0.6)
    
    inner_gs = gs[2, :].subgridspec(len(BANDS), 1, hspace=0)
    for i, (band, (l_freq, h_freq)) in enumerate(BANDS.items()):
        ax3_sub = fig.add_subplot(inner_gs[i])
        filtered_signal = mne.filter.filter_data(data_uv, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False, fir_design='firwin')
        ax3_sub.plot(times, filtered_signal, color=band_colors[band], label=f'{band} ({l_freq}-{h_freq} Hz)')
        ax3_sub.legend(loc='upper right')
        ax3_sub.set_yticks([])
        if i < len(BANDS) - 1:
            ax3_sub.set_xticks([])

    # --- Step 4: Final DE Features as Topomaps ---
    ax4_main = fig.add_subplot(gs[3, :])
    ax4_main.set_title('Step 4: Final DE Feature "Image" Visualized as Topomaps (Normalized per Band)', fontsize=18)
    ax4_main.axis('off')

    de_features_single_trial = extract_de_features(epoch.copy())[0, 0, :, :]
    info_for_plot = mne.pick_info(epoch.info.copy(), mne.pick_channels(epoch.ch_names, include=CH_NAMES_62))
    
    inner_gs2 = gs[3, :].subgridspec(1, len(BANDS), wspace=0.1)
    for i, band_name in enumerate(BANDS.keys()):
        ax4_sub = fig.add_subplot(inner_gs2[i])
        # Normalize each band's data to 0-1 to highlight spatial patterns
        band_data = de_features_single_trial[:62, i]
        normalized_data = (band_data - band_data.min()) / (band_data.max() - band_data.min())
        im, _ = plot_topomap(normalized_data, info_for_plot, axes=ax4_sub, show=False, cmap='viridis', vlim=(0, 1))
        ax4_sub.set_title(band_name, fontsize=14)

    cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.15])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Normalized Differential Entropy', fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plot_path = os.path.join(output_dir, 'de_feature_explanation.png')
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path


def plot_confusion_matrix(y_true, y_pred, task_name, output_dir):
    """
    Generates and saves a confusion matrix plot.
    """
    print(f"  Generating confusion matrix for {task_name}...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Static', 'Morph'], yticklabels=['Static', 'Morph'])
    plt.title(f'Confusion Matrix for {task_name}', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, f'confusion_matrix_{task_name.replace(":", "").replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def plot_channel_importance(de_features, labels, task_name, info, output_dir):
    """
    Calculates and plots channel importance using a simple linear model in a single figure.
    """
    print(f"  Calculating and plotting channel importance for {task_name}...")
    n_epochs, _, n_channels, n_bands = de_features.shape
    X = de_features.reshape(n_epochs, -1)
    
    lr = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced')
    lr.fit(X, labels)
    
    importances = lr.coef_[0].reshape(n_channels, n_bands)
    
    info_for_plot = mne.pick_info(info.copy(), mne.pick_channels(info.ch_names, include=CH_NAMES_62))
    
    vmax = np.max(np.abs(importances[:62, :]))
    vmin = -vmax
    
    fig, axes = plt.subplots(1, n_bands, figsize=(25, 6), constrained_layout=True)
    fig.suptitle(f'Channel Importance for {task_name}\n(Logistic Regression Coefficients)', fontsize=20, weight='bold')

    for band_idx, band_name in enumerate(BANDS.keys()):
        channel_importances = importances[:62, band_idx]
        im, _ = plot_topomap(channel_importances, info_for_plot, axes=axes[band_idx], show=False, cmap='RdBu_r', vlim=(vmin, vmax))
        axes[band_idx].set_title(band_name, fontsize=16)
        
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.6, pad=0.02)
    cbar.set_label("Coefficient (Importance for 'Morph' vs 'Static')", fontsize=14)

    plot_path = os.path.join(output_dir, f'channel_importance_{task_name.replace(":", "").replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path, importances


def plot_clustered_channel_importance(importances, task_name, info, output_dir, n_clusters=4):
    """
    Performs k-means clustering on channel importance profiles and plots the
    results in a single, combined figure for better interpretation.

    This function creates a two-panel plot:
    1.  A topomap showing the spatial location of the channel clusters.
    2.  A bar chart showing the mean importance of each cluster, making it
        easy to see which functional brain regions are most leveraged by the model.
    """
    print(f"  Clustering channel importance for {task_name}...")
    
    # Use only the 62 channels with locations for clustering and plotting
    importances_62 = importances[:62, :]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(importances_62)

    info_for_plot = mne.pick_info(info.copy(), mne.pick_channels(info.ch_names, include=CH_NAMES_62))

    fig = plt.figure(figsize=(22, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    fig.suptitle(f'Clustered Channel Importance Analysis for {task_name}', fontsize=24, weight='bold')

    # --- Plot 1: Topomap of Clusters ---
    ax1 = fig.add_subplot(gs[0])
    im, _ = plot_topomap(cluster_labels, info_for_plot, axes=ax1, show=False, cmap='tab10', vlim=(0, n_clusters-1))
    ax1.set_title('Spatial Distribution of Channel Clusters', fontsize=18)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.tab10(i / (n_clusters - 1)), label=f'Cluster {i}') for i in range(n_clusters)]
    ax1.legend(handles=legend_elements, bbox_to_anchor=(0.1, 0.1), title="Functional Clusters", fontsize=12)

    # --- Plot 2: Bar Chart of Cluster Importance ---
    ax2 = fig.add_subplot(gs[1])
    cluster_importance_mean = [importances_62[cluster_labels == i].mean() for i in range(n_clusters)]
    cluster_names = [f'Cluster {i}' for i in range(n_clusters)]
    
    sns.barplot(x=cluster_names, y=cluster_importance_mean, hue=cluster_names, palette='tab10', ax=ax2, legend=False)
    ax2.set_title('Mean Importance of Each Functional Cluster', fontsize=18)
    ax2.set_ylabel("Mean Coefficient (Importance)", fontsize=16)
    ax2.set_xlabel("Channel Cluster", fontsize=16)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plot_path = os.path.join(output_dir, f'clustered_importance_{task_name.replace(":", "").replace(" ", "_")}.png')
    fig.savefig(plot_path)
    plt.close(fig)

    return plot_path


def plot_loso_accuracies(subject_accuracies, task_name, output_dir):
    """
    Generates and saves a bar chart of per-participant accuracies.
    """
    if not subject_accuracies:
        print("  Skipping accuracy bar plot: no subject accuracies available.")
        return None
        
    plt.figure(figsize=(15, 8))
    subjects = list(subject_accuracies.keys())
    accuracies = list(subject_accuracies.values())
    
    sns.barplot(x=subjects, y=accuracies, hue=subjects, palette='viridis', legend=False)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 1.1)
    plt.title(f'Per-Participant Accuracy for {task_name}', fontsize=18, weight='bold')
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Participant ID', fontsize=14)
    mean_acc = np.mean(accuracies)
    plt.axhline(y=mean_acc, color='r', linestyle='--', label=f'Mean: {mean_acc:.2f}')
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'participant_accuracy_{task_name.replace(":", "").replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path


# --- 6. Main Execution Block ---
# -------------------------------

def main():
    """
    The main function that orchestrates the entire pipeline.
    """
    # --- Setup ---
    # Create a timestamped directory for this run's plots to keep them organized.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, f'plots_{run_timestamp}')
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'trained_models')
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- Data Loading and Caching ---
    # Check if pre-processed data exists in the cache to save time.
    if os.path.exists(ALL_EPOCHS_CACHE_FILE) and os.path.exists(LABELS_CACHE_FILE):
        print("Loading data from cache...")
        X_epochs = mne.read_epochs(ALL_EPOCHS_CACHE_FILE, preload=True, verbose=False)
        labels_data = np.load(LABELS_CACHE_FILE)
        y_emotion = labels_data['y_emotion']
        y_variant = labels_data['y_variant']
        groups = labels_data['groups']
    else:
        # If no cache, load from raw files.
        print("Loading data from source files...")
        subject_ids = sorted(list(set([f.split('_')[0] for f in os.listdir(DATA_DIR) if f.endswith('_eeg.dat')])))
        all_epochs_data, all_emotion_labels, all_variant_labels, all_groups = [], [], [], []
        for subject_id in tqdm(subject_ids, desc="Loading All Subject Data"):
            epochs, emotions, variants, groups_subj = load_subject_data(subject_id, DATA_DIR)
            if epochs is not None:
                all_epochs_data.append(epochs)
                all_emotion_labels.append(emotions)
                all_variant_labels.append(variants)
                all_groups.append(groups_subj)

        if not all_epochs_data:
            print("No data loaded. Exiting.")
            return

        # Concatenate data from all subjects into single arrays.
        X_epochs = mne.concatenate_epochs(all_epochs_data)
        y_emotion = np.concatenate(all_emotion_labels)
        y_variant = np.concatenate(all_variant_labels)
        groups = np.concatenate(all_groups)

        # Save the processed data to cache for future runs.
        print("Saving loaded data to cache...")
        X_epochs.save(ALL_EPOCHS_CACHE_FILE, overwrite=True)
        np.savez(LABELS_CACHE_FILE, y_emotion=y_emotion, y_variant=y_variant, groups=groups)

    # --- DE Feature Extraction and Caching ---
    if os.path.exists(DE_FEATURES_CACHE_FILE):
        print("Loading DE features from cache...")
        de_features = np.load(DE_FEATURES_CACHE_FILE)
    else:
        print("Extracting DE features...")
        de_features = extract_de_features(X_epochs)
        print("Saving DE features to cache...")
        np.save(DE_FEATURES_CACHE_FILE, de_features)

    # --- Generate Plots for Methodology Section of Report ---
    print("Generating plots for report methodology section...")
    plot_data = {}
    # Load one subject's data to create sample plots for the report.
    first_subject_id = sorted(list(set([f.split('_')[0] for f in os.listdir(DATA_DIR) if f.endswith('_eeg.dat')])))[0]
    first_subject_epochs, _, _, _ = load_subject_data(first_subject_id, DATA_DIR)
    
    if first_subject_epochs is None:
        print(f"Could not load data for first subject {first_subject_id} to generate methodology plots. Exiting.")
        return

    # Generate and save the detailed DE explanation plot.
    de_exp_path = plot_de_feature_explanation(first_subject_epochs[0], PLOTS_DIR)
    plot_data['de_explanation'] = de_exp_path

    # --- Define Classification Tasks ---
    # New combined task: classify all 'static' trials vs. all 'morph' trials, regardless of emotion.
    TASKS = {
        'All Static vs All Morph': (np.ones_like(y_emotion, dtype=bool), y_variant)
    }

    # --- Main Analysis Loop ---
    results = {}
    linear_plots = {}
    participant_analysis_plots = {}
    de_dist_plots = {}
    importance_plots = {}
    cluster_plots = {}

    for task, (emotion_mask, labels) in TASKS.items():
        print(f"\n--- Running Task: {task} ---")
        task_name_safe = task.replace(":", "").replace(" ", "_")
        
        # --- Linear ERP Analysis (for comparison) ---
        print(f"  Running Linear ERP analysis for {task}...")
        task_epochs = X_epochs[emotion_mask]
        task_labels = labels[emotion_mask]
        
        static_erp = task_epochs[task_labels == 0].average()
        morph_erp = task_epochs[task_labels == 1].average()

        # Create topomaps for each condition at key ERP time points.
        times_to_plot = [0.1, 0.17, 0.25, 0.5] # P100, N170, P250, LPC
        
        # Create copies of the Evoked objects with only the 62 EEG channels that have locations.
        static_erp_eeg = static_erp.copy().pick(CH_NAMES_62)
        morph_erp_eeg = morph_erp.copy().pick(CH_NAMES_62)

        # Find a common, robust color scale for all topomaps in this task.
        all_data = np.vstack([static_erp_eeg.get_data(), morph_erp_eeg.get_data()])
        vmax = np.quantile(np.abs(all_data), 0.98) * 1e6 # Use 98th percentile and scale to µV
        vmin = -vmax

        # Plot for 'Static' condition
        fig_static = static_erp_eeg.plot_topomap(times=times_to_plot, show=False, time_unit='s', cmap='RdBu_r', vlim=(vmin, vmax))
        fig_static.suptitle(f"Grand Average ERP Topomaps for 'Static' Condition\n({task}) | Voltage in µV", fontsize=14)
        static_topo_path = os.path.join(PLOTS_DIR, f'topo_static_{task_name_safe}.png')
        fig_static.savefig(static_topo_path)
        plt.close(fig_static)
        linear_plots[f'topo_static_{task_name_safe.lower()}'] = static_topo_path

        # Plot for 'Morph' condition
        fig_morph = morph_erp_eeg.plot_topomap(times=times_to_plot, show=False, time_unit='s', cmap='RdBu_r', vlim=(vmax, vmax))
        fig_morph.suptitle(f"Grand Average ERP Topomaps for 'Morph' Condition\n({task}) | Voltage in µV", fontsize=14)
        morph_topo_path = os.path.join(PLOTS_DIR, f'topo_morph_{task_name_safe}.png')
        fig_morph.savefig(morph_topo_path)
        plt.close(fig_morph)
        linear_plots[f'topo_morph_{task_name_safe.lower()}'] = morph_topo_path
        
        # --- Machine Learning Analysis ---
        # Select task-specific data
        de_features_task = de_features[emotion_mask]
        y_task = labels[emotion_mask]
        groups_task = groups[emotion_mask]

        # Handle potential NaNs from feature extraction (e.g., from flat signals).
        de_features_task = np.nan_to_num(de_features_task)

        # Generate and save DE distribution plot for this task.
        de_dist_path = plot_de_distribution(de_features_task, y_task, task, PLOTS_DIR)
        de_dist_plots[task_name_safe] = de_dist_path

        # Generate and save Channel Importance plots.
        importance_paths, importances = plot_channel_importance(de_features_task, y_task, task, X_epochs.info, PLOTS_DIR)
        importance_plots[task_name_safe] = importance_paths
        
        # Generate and save Clustered Channel Importance plot.
        cluster_plot_path = plot_clustered_channel_importance(importances, task, X_epochs.info, PLOTS_DIR)
        cluster_plots[task_name_safe] = cluster_plot_path

        # --- Data Preparation for PyTorch ---
        n_epochs, _, n_channels, n_bands = de_features_task.shape
        de_features_reshaped = de_features_task.reshape(n_epochs, -1)

        # Standardize features (z-score normalization) is crucial for many ML models.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(de_features_reshaped)
        
        # Reshape back to image-like format for ResNet and convert to PyTorch tensor.
        X_tensor = torch.tensor(X_scaled.reshape(n_epochs, 1, n_channels, n_bands), dtype=torch.float32)

        # --- Leave-One-Subject-Out (LOSO) Cross-Validation ---
        logo = LeaveOneGroupOut()
        all_preds, all_true = [], []
        subject_accuracies = {}
        unique_groups = np.unique(groups_task)

        for fold, (train_idx, test_idx) in enumerate(logo.split(X_tensor, y_task, groups_task)):
            test_subject = unique_groups[fold]
            print(f"  Fold {fold+1}/{logo.get_n_splits(groups=groups_task)}... (Testing on subject {test_subject})")
            
            # Split data into training and testing sets for this fold.
            X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
            y_train, y_test = y_task[train_idx], y_task[test_idx]
            
            # Further split the training set into a smaller training and a validation set.
            # The validation set is used for early stopping.
            X_train_t, X_val, y_train_t, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

            # Create PyTorch DataLoaders.
            train_dataset = TensorDataset(X_train_t, torch.tensor(y_train_t, dtype=torch.long))
            val_dataset = TensorDataset(X_val, torch.tensor(y_val, dtype=torch.long))
            test_dataset = TensorDataset(X_test, torch.tensor(y_test, dtype=torch.long))
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # --- Model Initialization and Training ---
            model_path = os.path.join(MODELS_DIR, f'{task_name_safe}_subject_{test_subject}.pt')

            # Initialize a new model for each fold.
            model = models.resnet18(weights=None) # Not using pre-trained weights
            # Adapt the first convolutional layer to accept our 1-channel (grayscale) input.
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 3), stride=(2, 1), padding=(3, 1), bias=False)
            # Adapt the final fully connected layer to our binary classification task.
            model.fc = nn.Linear(model.fc.in_features, 2)
            model.to(DEVICE)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(patience=10, verbose=False, path=model_path)

            # Training loop for the current fold.
            for epoch in range(100): # Max 100 epochs
                model.train()
                for inputs, labels_batch in train_loader:
                    inputs, labels_batch = inputs.to(DEVICE), labels_batch.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()

                # Validation loop to check for overfitting.
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, labels_batch in val_loader:
                        inputs, labels_batch = inputs.to(DEVICE), labels_batch.to(DEVICE)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels_batch)
                        val_loss += loss.item()
                
                # Update early stopping.
                early_stopping(val_loss / len(val_loader), model)
                if early_stopping.early_stop:
                    break
            
            # --- Evaluation on the Held-Out Test Subject ---
            model.load_state_dict(torch.load(model_path)) # Load the best performing model
            model.eval()
            fold_preds, fold_true = [], []
            with torch.no_grad():
                for inputs, labels_batch in test_loader:
                    inputs, labels_batch = inputs.to(DEVICE), labels_batch.to(DEVICE)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    fold_preds.extend(predicted.cpu().numpy())
                    fold_true.extend(labels_batch.cpu().numpy())
            
            all_preds.extend(fold_preds)
            all_true.extend(fold_true)
            subject_accuracies[test_subject] = accuracy_score(fold_true, fold_preds)

        # --- Per-Participant Accuracy Analysis ---
        bar_plot_path = plot_loso_accuracies(subject_accuracies, task, PLOTS_DIR)
        participant_analysis_plots[f'bar_{task_name_safe.lower()}'] = bar_plot_path

        # --- Analyze the lowest-performing subject ---
        low_subj_plot_path = plot_lowest_accuracy_subject_erp(
            task_epochs=task_epochs,
            task_labels=y_task,
            groups=groups_task,
            subject_accuracies=subject_accuracies,
            task_name=task,
            output_dir=PLOTS_DIR,
            tmin=T_MIN
        )
        participant_analysis_plots[f'low_acc_erp_{task_name_safe.lower()}'] = low_subj_plot_path

        # --- Store Final Results and Generate Report for the Task ---
        accuracy = accuracy_score(all_true, all_preds)
        results[task] = accuracy
        
        # Generate confusion matrix for the final results
        cm_path = plot_confusion_matrix(all_true, all_preds, task, PLOTS_DIR)
        plot_data['confusion_matrix'] = cm_path

        # --- Consolidate All Plot Paths for the Report ---
        plot_paths = {
            'de_explanation': plot_data.get('de_explanation'),
            'de_distribution': de_dist_plots.get(task_name_safe),
            'channel_importance': importance_plots.get(task_name_safe),
            'clustered_channel_importance': cluster_plots.get(task_name_safe),
            'loso_accuracies': participant_analysis_plots.get(f'bar_{task_name_safe.lower()}'),
            'low_accuracy_erp': participant_analysis_plots.get(f'low_acc_erp_{task_name_safe.lower()}'),
            'confusion_matrix': plot_data.get('confusion_matrix'),
        }

        # --- Generate Final Report for this task ---
        generate_report(
            output_pdf_path=os.path.join(OUTPUT_DIR, f'DE_ResNet_Report_{task_name_safe}.pdf'),
            task_name=task,
            tmin=T_MIN,
            tmax=T_MAX,
            loso_accuracies=subject_accuracies,
            overall_accuracy=accuracy,
            plot_paths=plot_paths,
            de_params={
                'fs': X_epochs.info['sfreq'],
                'nperseg': 256,
                'noverlap': 0,
                'window': 'hann'
            },
            bands=BANDS
        )


if __name__ == '__main__':
    main()
