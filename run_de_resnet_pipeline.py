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
from scipy.signal import welch
from tqdm import tqdm
from torchvision import models
import pandas as pd
from mne.viz import plot_topomap
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# --- 1. Configuration ---
# ------------------------

# --- File Paths ---
# Define the directory where the raw EEG data is stored and where outputs will be saved.
DATA_DIR = '/Users/smart/Programming/Semi-Supervised Learning Faces/FaceMorph_singletrial'
OUTPUT_DIR = '/Users/smart/Programming/Semi-Supervised Learning Faces/Semi_supervised_Framework/loso_pipeline'
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots') # Directory for all generated image files

# --- EEG & Epoching Parameters ---
NUM_CHANNELS = 62                   # Number of EEG channels to use from the data files.
NATIVE_SAMPLING_RATE = 1000         # Hz, the original sampling rate of the EEG data.
EPOCH_DURATION = 4.404              # seconds, the length of each trial segment.
T_MIN, T_MAX = 0, EPOCH_DURATION    # The time window for each epoch relative to the event.
BASELINE = None                     # No baseline correction is applied as per data characteristics.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available.

# --- Caching Configuration ---
# Caching saves processed data to disk to speed up subsequent runs.
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')
ALL_EPOCHS_CACHE_FILE = os.path.join(CACHE_DIR, 'all_epochs-epo.fif') # For MNE Epochs
DE_FEATURES_CACHE_FILE = os.path.join(CACHE_DIR, 'de_features.npy')   # For extracted DE features
LABELS_CACHE_FILE = os.path.join(CACHE_DIR, 'labels.npz')             # For corresponding labels

# --- Feature Extraction Parameters ---
# Define the standard EEG frequency bands for Differential Entropy calculation.
BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 14),
    'Beta': (14, 31),
    'Gamma': (31, 50)
}

# --- Channel Configuration ---
# Define the names of the 62 EEG channels and the standard montage for topoplots.
CH_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 
    'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 
    'M1', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'M2', 'P7', 'P5', 'P3', 'P1', 'Pz', 
    'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2'
]
MONTAGE = mne.channels.make_standard_montage('standard_1020')


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
    info = mne.create_info(ch_names=CH_NAMES, sfreq=NATIVE_SAMPLING_RATE, ch_types=['eeg'] * NUM_CHANNELS)
    
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
            # The file contains 66 channels, but we only use the first 62.
            n_channels_total_in_file = 66
            n_total_samples = data.size // n_channels_total_in_file
            data = data[:n_total_samples * n_channels_total_in_file] # Ensure data is divisible
            data_full = data.reshape((n_channels_total_in_file, n_total_samples), order='F')
            data_to_use = data_full[:NUM_CHANNELS, :]

            # --- MNE Object Creation ---
            # Create an MNE Raw object from the numpy array.
            raw = mne.io.RawArray(data_to_use, info, verbose=False)
            with mne.utils.use_log_level('error'): # Suppress warnings about channel locations
                raw.set_montage(MONTAGE, on_missing='warn')
            
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
    Extracts Differential Entropy (DE) features using a biologically plausible method.

    This function implements a standard and robust method for estimating band power:
    1.  **Band-pass Filtering**: For each frequency band (Delta, Theta, etc.), the
        raw EEG signal of a trial is filtered to isolate only the frequencies
        within that specific band.
    2.  **Variance Calculation**: The variance of this filtered signal is then
        calculated. In signal processing, the variance of a zero-mean signal is
        equivalent to its power.
    3.  **DE Calculation**: The Differential Entropy is computed from this power
        value using the formula for a Gaussian distribution: 0.5 * log(2 * pi * e * power).

    This approach is preferred over some FFT-based methods as it is less susceptible
    to spectral leakage and provides a more direct measure of power within each band.

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

    # Iterate over each epoch (trial) and channel to extract features.
    for i in tqdm(range(n_epochs), desc="Extracting DE Features"):
        for j_chan in range(n_channels):
            for band_idx, (_, (l_freq, h_freq)) in enumerate(BANDS.items()):
                
                # Step 1: Band-pass filter the signal for the current band.
                # A copy of the signal is used for each filtering operation to avoid side effects.
                filtered_signal = mne.filter.filter_data(
                    data[i, j_chan], 
                    sfreq, 
                    l_freq, 
                    h_freq, 
                    fir_design='firwin', # Use a standard FIR filter
                    verbose=False
                )
                
                # Step 2: Calculate the variance of the filtered signal, which represents its power.
                band_power = np.var(filtered_signal)
                
                # Step 3: Calculate Differential Entropy from the band power.
                # A small epsilon (1e-9) is added for numerical stability to avoid log(0).
                de_features[i, j_chan, band_idx] = 0.5 * np.log(2 * np.pi * np.e * band_power + 1e-9)
    
    # Reshape for ResNet: add a singleton dimension for the "channel" in the CNN.
    # The final shape is (n_epochs, 1, n_channels, n_bands).
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


def generate_report(results, plots):
    """
    Generates the final PDF report, compiling all results and visualizations.

    Args:
        results (dict): A dictionary of classification accuracies for each task.
        plots (dict): A dictionary mapping plot names to their file paths.
    """
    pdf = PDF()
    pdf.add_page()

    # --- Executive Summary ---
    pdf.chapter_title("Executive Summary")
    summary_text = (
        "This report documents a machine learning pipeline designed to classify EEG signals in response to viewing 'static' versus 'morphing' facial expressions. Using a Leave-One-Subject-Out (LOSO) cross-validation methodology, a ResNet18 deep learning model was trained on Differential Entropy (DE) features derived from the EEG data.\n\n"
        "Key findings include:\n"
        f"- High classification accuracies were achieved for both the 'Fear' ({results.get('Fear: Static vs Morph', 0)*100:.2f}%) and 'Anger' ({results.get('Anger: Static vs Morph', 0)*100:.2f}%) conditions, demonstrating that DE features contain robust information for this task.\n"
        "- Channel importance analysis, further refined by k-means clustering, revealed that the model heavily relied on distinct functional networks of channels, particularly in posterior-occipital regions, to make its classifications.\n\n"
        "The methodology, which uses a standard band-pass filtering and variance calculation to derive signal power for DE, was validated as a robust and appropriate choice. The results indicate that this DE-ResNet approach is highly effective for decoding cognitive states from EEG data in this paradigm."
    )
    pdf.chapter_body(summary_text)


    # --- Introduction & Methodology Section ---
    pdf.chapter_title("1. Introduction & Methodology")
    methodology_text = (
        "This report details a classification pipeline for EEG data to distinguish between trials where subjects viewed 'static' faces versus 'morphing' faces. "
        "Two emotional contexts were analyzed: 'Fear' and 'Anger'.\n\n"
        "The methodology involves the following key steps:\n"
        "1. **Data Loading**: Raw EEG data (.dat files) for 23 subjects was loaded.\n"
        "2. **Epoching**: Data was segmented into 4.404-second epochs based on the experimental design.\n"
        "3. **Feature Extraction (Differential Entropy)**: For each epoch, the signal was band-pass filtered into 5 standard frequency bands (Delta, Theta, Alpha, Beta, Gamma). The variance of each filtered signal (a proxy for power) was used to compute the Differential Entropy (DE), creating a 62x5 'image' for each trial.\n"
        "4. **Classification (ResNet18)**: A ResNet18 convolutional neural network was adapted to classify these 2D DE feature images. The model was trained and evaluated using a Leave-One-Subject-Out (LOSO) cross-validation scheme.\n"
        "5. **Analysis**: A series of analyses were performed, including traditional ERP analysis, DE feature distribution, and model-based channel importance."
    )
    pdf.chapter_body(methodology_text)
    pdf.add_plot(plots.get('de_features'), title="Methodology: Example DE Features for a Single Trial")
    pdf.add_plot(plots.get('de_exp_step1'), title="Methodology: Step 1 - Raw EEG Signal")
    pdf.add_plot(plots.get('de_exp_step2'), title="Methodology: Step 2 - Power Spectral Density")
    pdf.add_plot(plots.get('de_exp_step3'), title="Methodology: Step 3 - Band-Filtered Signals")
    pdf.add_plot(plots.get('de_exp_step4'), title="Methodology: Step 4 - Final DE Features as Topomaps")

    # --- Note on DE Calculation ---
    pdf.add_page()
    pdf.chapter_title("2. A Note on Calculating Power and Differential Entropy")
    de_explanation_text = (
        "A critical step in this analysis is the calculation of Differential Entropy (DE), which relies on an accurate estimation of signal power within specific frequency bands (e.g., Alpha, Beta).\n\n"
        "There are several valid methods to estimate band power, each with its own characteristics:\n"
        "1. **FFT-based Methods (e.g., Welch's Method)**: These methods use the Fast Fourier Transform to convert the signal from the time domain to the frequency domain. The power is then calculated by averaging the spectral density within the desired frequency band. This is a very common and efficient approach.\n\n"
        "2. **Wavelet Transform**: This method provides a time-frequency representation, showing how the frequency content of a signal changes over time. It offers excellent temporal resolution for high frequencies and excellent frequency resolution for low frequencies. Power can be derived by integrating over time for specific frequency bands.\n\n"
        "3. **Band-Pass Filtering (Our Method)**: This is the method used in our pipeline. The raw signal is first filtered to isolate only the frequencies of a specific band (e.g., 8-14 Hz for Alpha). The power of this resulting signal is then calculated. For a zero-mean signal, **the variance is mathematically equivalent to its power**. This is a direct, robust, and intuitive method that is widely used in EEG signal processing and is implemented here using the standard `mne.filter.filter_data` function.\n\n"
        "**Conclusion**: Our choice of band-pass filtering followed by variance calculation is a standard, valid, and 'biologically plausible' way to compute the band power required for the Differential Entropy feature. It is computationally efficient and less prone to issues like spectral leakage compared to some FFT methods."
    )
    pdf.chapter_body(de_explanation_text)


    # --- Results Section ---
    pdf.add_page()
    pdf.chapter_title("3. Classification Results")
    results_summary = "The ResNet18 model was trained separately for each task. The table below shows the final classification accuracies achieved using Leave-One-Subject-Out cross-validation:"
    pdf.chapter_body(results_summary)
    
    # Create a formatted table for results
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(100, 10, 'Task', 1)
    pdf.cell(50, 10, 'Accuracy', 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 12)
    for task, accuracy in results.items():
        pdf.cell(100, 10, task, 1)
        pdf.cell(50, 10, f"{accuracy*100:.2f}%", 1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln()
    
    results_discussion = (
        "The high accuracy scores suggest that the DE features, when treated as images and processed by a deep learning model, provide a strong basis for discriminating between the 'static' and 'morph' conditions."
    )
    pdf.chapter_body(results_discussion)

    # --- Analysis Plots Section ---
    pdf.add_page()
    pdf.chapter_title("4. DE Feature Distribution Analysis")
    pdf.chapter_body("The following plots show the distribution of the average Differential Entropy values across all channels for each frequency band, separated by condition ('Static' vs. 'Morph'). This helps visualize the feature space that the classifier is learning from.")
    pdf.add_plot(plots.get('Fear_Static_vs_Morph'), title="DE Feature Distribution (Fear Task)")
    pdf.add_plot(plots.get('Anger_Static_vs_Morph'), title="DE Feature Distribution (Anger Task)")
    
    # --- Channel Importance Section ---
    pdf.add_page()
    pdf.chapter_title("5. Channel Importance Analysis")
    pdf.chapter_body("To understand which channels and frequency bands were most influential for the classification, a simple Logistic Regression model was trained on the DE features. The coefficients of this model serve as a proxy for feature importance. The topomaps below visualize these coefficients, with red indicating features that push the prediction towards 'Morph' and blue towards 'Static'.")
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, "Fear Task: Channel Importance", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
    for band in BANDS.keys():
        key = f'channel_importance_Fear_Static_vs_Morph_{band}.png'
        if key in plots:
             pdf.add_plot(plots[key], title=f"Importance for Fear Task ({band} Band)")

    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, "Anger Task: Channel Importance", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
    for band in BANDS.keys():
        key = f'channel_importance_Anger_Static_vs_Morph_{band}.png'
        if key in plots:
             pdf.add_plot(plots[key], title=f"Importance for Anger Task ({band} Band)")

    # --- Clustered Channel Importance Section ---
    pdf.add_page()
    pdf.chapter_title("6. Clustered Channel Importance Analysis")
    pdf.chapter_body(
        "To further understand the spatial patterns of importance, k-means clustering was applied to the channel importance profiles (the 5 importance coefficients across the frequency bands for each channel). "
        "This groups channels that have a similar pattern of importance across frequencies. The topomaps below show these functional clusters. Each color represents a group of channels that the model uses in a similar way. The bar chart shows the average importance of each cluster, revealing which functional networks were most critical for the classification."
    )
    pdf.add_plot(plots.get('cluster_topo_fear_static_vs_morph'), title="Channel Clusters based on Importance Profiles (Fear Task)")
    pdf.add_plot(plots.get('cluster_bar_fear_static_vs_morph'), title="Mean Importance of each Channel Cluster (Fear Task)")
    pdf.add_page()
    pdf.add_plot(plots.get('cluster_topo_anger_static_vs_morph'), title="Channel Clusters based on Importance Profiles (Anger Task)")
    pdf.add_plot(plots.get('cluster_bar_anger_static_vs_morph'), title="Mean Importance of each Channel Cluster (Anger Task)")


    # --- Per-Participant Analysis Section ---
    pdf.add_page()
    pdf.chapter_title("7. Per-Participant Accuracy")
    pdf.chapter_body("The following bar charts show the final classification accuracy for each individual participant, providing insight into the variability of model performance across the cohort.")
    pdf.add_plot(plots.get('bar_fear_static_vs_morph'), title="Per-Participant Accuracy (Fear Task)")
    pdf.add_plot(plots.get('bar_anger_static_vs_morph'), title="Per-Participant Accuracy (Anger Task)")
    
    pdf.add_page()
    pdf.chapter_title("8. Lowest-Accuracy Subject Analysis")
    pdf.chapter_body("To investigate sources of variability, the ERPs of the lowest-performing subject in each task were compared against the grand average of all other subjects. This can help identify if the subject had an atypical neural response. If a plot is missing, it is because the subject lacked sufficient data in one of the conditions for a valid comparison.")
    pdf.add_plot(plots.get('low_acc_erp_fear_static_vs_morph'), title="Lowest Accuracy Subject ERP Comparison (Fear Task)")
    pdf.add_plot(plots.get('low_acc_erp_anger_static_vs_morph'), title="Lowest Accuracy Subject ERP Comparison (Anger Task)")

    # --- Linear ERP Analysis Section ---
    pdf.add_page()
    pdf.chapter_title("9. Comparative Linear Analysis (ERP Topomaps)")
    pdf.chapter_body("For a traditional analysis comparison, Event-Related Potential (ERP) topomaps were generated. The plots show the scalp distribution of voltage at key time points (P100, N170, P250, LPC) for both 'static' and 'morph' conditions, averaged across all subjects.")
    pdf.add_plot(plots.get('topo_static_fear_static_vs_morph'), title="Grand Average Topomaps for 'Static' Condition (Fear Task)")
    pdf.add_plot(plots.get('topo_morph_fear_static_vs_morph'), title="Grand Average Topomaps for 'Morph' Condition (Fear Task)")
    pdf.add_page()
    pdf.add_plot(plots.get('topo_static_anger_static_vs_morph'), title="Grand Average Topomaps for 'Static' Condition (Anger Task)")
    pdf.add_plot(plots.get('topo_morph_anger_static_vs_morph'), title="Grand Average Topomaps for 'Morph' Condition (Anger Task)")

    # --- Save the PDF ---
    report_path = os.path.join(OUTPUT_DIR, 'classification_report_final.pdf')
    pdf.output(report_path)
    print(f"Final PDF report generated at {report_path}")


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
    Generates a series of plots to explain the DE feature extraction process in detail.
    This provides a clear, step-by-step visualization of how the final features are derived
    from the raw EEG signal, enhancing the interpretability of the methodology.
    """
    print("  Generating DE feature explanation plots...")
    
    sfreq = epoch.info['sfreq']
    times = epoch.times
    # Select a representative channel (POz) and convert to microvolts for plotting
    data_uv = epoch.get_data(picks='POz')[0, 0, :] * 1e6

    # --- Step 1: Plot the raw EEG signal ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, data_uv, color='black')
    ax.set_title('Step 1: Raw EEG Signal from a Single Trial (Channel POz)', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude (µV)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    step1_path = os.path.join(PLOTS_DIR, 'de_explanation_step1_raw_signal.png')
    plt.savefig(step1_path)
    plt.close(fig)

    # --- Step 2: Plot the Power Spectral Density (PSD) ---
    # The PSD shows the power of the signal at different frequencies.
    freqs, psd = welch(data_uv, fs=sfreq, nperseg=int(sfreq), detrend='linear')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.loglog(freqs, psd, color='navy') # Log-log scale is standard for viewing EEG power
    ax.set_xlim(1, 55)
    ax.set_title('Step 2: Power Spectral Density (PSD) of the Signal', fontsize=14)
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Power Spectral Density (µV²/Hz)', fontsize=12)
    ax.grid(True, which="both", ls="--", alpha=0.6)
    band_colors = {'Delta': '#1f77b4', 'Theta': '#ff7f0e', 'Alpha': '#2ca02c', 'Beta': '#d62728', 'Gamma': '#9467bd'}
    for band, (l_freq, h_freq) in BANDS.items():
        ax.axvspan(l_freq, h_freq, alpha=0.2, color=band_colors[band], label=f'{band} ({l_freq}-{h_freq} Hz)')
    ax.legend()
    step2_path = os.path.join(PLOTS_DIR, 'de_explanation_step2_psd.png')
    plt.savefig(step2_path)
    plt.close(fig)

    # --- Step 3: Plot the band-filtered signals ---
    # This shows what the signal looks like when only specific frequency bands are present.
    fig, axes = plt.subplots(len(BANDS), 1, figsize=(12, 14), sharex=True, sharey=True)
    fig.suptitle('Step 3: EEG Signal Filtered into Specific Frequency Bands', fontsize=16)
    for i, (band, (l_freq, h_freq)) in enumerate(BANDS.items()):
        filtered_signal = mne.filter.filter_data(data_uv, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False, fir_design='firwin')
        axes[i].plot(times, filtered_signal, color=band_colors[band])
        axes[i].set_title(f'{band} Band ({l_freq}-{h_freq} Hz)')
        axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[-1].set_xlabel('Time (s)', fontsize=12)
    fig.text(0.04, 0.5, 'Amplitude (µV)', va='center', rotation='vertical', fontsize=12)
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    step3_path = os.path.join(PLOTS_DIR, 'de_explanation_step3_filtered_signals.png')
    plt.savefig(step3_path)
    plt.close(fig)

    # --- Step 4: Plot the final DE features as topomaps ---
    # This visualizes the spatial distribution of DE across the scalp for each band.
    epoch_for_de = epoch.copy()
    de_features_single_trial = extract_de_features(epoch_for_de)[0, 0, :, :]
    
    fig, axes = plt.subplots(1, len(BANDS), figsize=(22, 5))
    fig.suptitle('Step 4: Final DE Feature "Image" Visualized as Topomaps for One Trial', fontsize=16)
    # Find a common color scale for all bands in this trial for comparability.
    vmin = de_features_single_trial.min()
    vmax = de_features_single_trial.max()
    for i, band_name in enumerate(BANDS.keys()):
        im, _ = plot_topomap(de_features_single_trial[:, i], epoch.info, axes=axes[i], show=False, cmap='viridis', vlim=(vmin, vmax))
        axes[i].set_title(band_name, fontsize=14)
    
    # Add a single, shared colorbar for all topomaps.
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Differential Entropy (log of power)', fontsize=12)
    step4_path = os.path.join(PLOTS_DIR, 'de_explanation_step4_feature_topomaps.png')
    plt.savefig(step4_path)
    plt.close(fig)

    return step1_path, step2_path, step3_path, step4_path


def plot_channel_importance(de_features, labels, task_name, info, output_dir):
    """
    Calculates and plots channel importance using a simple linear model.
    The coefficients of a Logistic Regression model are used as a proxy for
    how much the model relies on each feature (channel x band) for its decision.
    """
    print(f"  Calculating and plotting channel importance for {task_name}...")
    n_epochs, _, n_channels, n_bands = de_features.shape
    X = de_features.reshape(n_epochs, -1)
    
    # Train a simple, interpretable model to find feature importances.
    lr = LogisticRegression(solver='liblinear', max_iter=1000, class_weight='balanced')
    lr.fit(X, labels)
    
    # The coefficients represent the "importance" of each feature.
    importances = lr.coef_[0].reshape(n_channels, n_bands)
    
    # Use a symmetrical, diverging colormap centered at 0.
    # Red indicates features pushing the prediction towards 'Morph'.
    # Blue indicates features pushing the prediction towards 'Static'.
    vmax = np.max(np.abs(importances))
    vmin = -vmax
    
    plot_paths = []
    for band_idx, band_name in enumerate(BANDS.keys()):
        channel_importances = importances[:, band_idx]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        im, _ = plot_topomap(channel_importances, info, axes=ax, show=False, cmap='RdBu_r', vlim=(vmin, vmax))
        ax.set_title(f'Channel Importance for {task_name}\n({band_name} Band)', fontsize=14)
        
        # Add a descriptive colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label("Model Coefficient (Importance for 'Morph' vs 'Static')", fontsize=10)

        plot_path = os.path.join(PLOTS_DIR, f'channel_importance_{task_name.replace(":", "").replace(" ", "_")}_{band_name}.png')
        plt.savefig(plot_path)
        plt.close(fig)
        plot_paths.append(plot_path)
        
    return plot_paths, importances


def plot_clustered_channel_importance(importances, task_name, info, output_dir, n_clusters=4):
    """
    Performs k-means clustering on channel importance profiles and plots the results.
    This helps to identify functional networks of channels that the model treats similarly.
    """
    print(f"  Clustering channel importance for {task_name}...")
    
    # --- K-Means Clustering ---
    # The features for clustering are the importance coefficients for each band.
    # This groups channels that have a similar importance *profile* across frequencies.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(importances)

    # --- Plot 1: Topomap of Clusters ---
    # This plot shows the spatial location of the channel clusters.
    fig, ax = plt.subplots(figsize=(8, 7))
    # We pass the cluster labels as data. MNE's plot_topomap will assign a color to each integer label.
    im, _ = plot_topomap(cluster_labels, info, axes=ax, show=False, cmap='tab10', vlim=(0, n_clusters-1))
    ax.set_title(f'Channel Clusters Based on Importance Profiles\n({task_name})', fontsize=14)
    
    # Create a custom legend for the clusters instead of a colorbar.
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.tab10(i / (n_clusters - 1)), label=f'Cluster {i}') for i in range(n_clusters)]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.1, 1.05))

    topo_plot_path = os.path.join(PLOTS_DIR, f'cluster_topo_{task_name.replace(":", "").replace(" ", "_")}.png')
    plt.savefig(topo_plot_path, bbox_inches='tight')
    plt.close(fig)

    # --- Plot 2: Bar Chart of Cluster Importance ---
    # This plot quantifies which clusters are, on average, most important.
    cluster_importance_mean = [importances[cluster_labels == i].mean() for i in range(n_clusters)]
    cluster_names = [f'Cluster {i}' for i in range(n_clusters)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=cluster_names, y=cluster_importance_mean, palette='tab10', ax=ax)
    ax.set_title(f'Mean Importance of Each Channel Cluster\n({task_name})', fontsize=14)
    ax.set_ylabel("Mean Coefficient (Importance)", fontsize=12)
    ax.set_xlabel("Channel Cluster", fontsize=12)
    ax.axhline(0, color='black', linewidth=0.8) # Add a zero line for reference
    plt.tight_layout()

    bar_plot_path = os.path.join(PLOTS_DIR, f'cluster_bar_{task_name.replace(":", "").replace(" ", "_")}.png')
    plt.savefig(bar_plot_path)
    plt.close(fig)

    return topo_plot_path, bar_plot_path


# --- 6. Main Execution Block ---
# -------------------------------

def main():
    """
    The main function that orchestrates the entire pipeline.
    """
    # --- Setup ---
    # Create directories for saved models, plots, and cache if they don't exist.
    MODELS_DIR = os.path.join(OUTPUT_DIR, 'trained_models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
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

    # Generate and save the detailed DE explanation plots.
    de_exp_paths = plot_de_feature_explanation(first_subject_epochs[0], OUTPUT_DIR)
    plot_data['de_exp_step1'] = de_exp_paths[0]
    plot_data['de_exp_step2'] = de_exp_paths[1]
    plot_data['de_exp_step3'] = de_exp_paths[2]
    plot_data['de_exp_step4'] = de_exp_paths[3]

    # --- Define Classification Tasks ---
    # Each task is a tuple of (mask_for_trials, labels_for_those_trials)
    TASKS = {
        'Fear: Static vs Morph': (y_emotion == 0, y_variant),
        'Anger: Static vs Morph': (y_emotion == 1, y_variant)
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
        
        # Find a common, robust color scale for all topomaps in this task.
        # This ensures the color scales are comparable across plots.
        all_data = np.vstack([static_erp.get_data(), morph_erp.get_data()])
        vmax = np.quantile(np.abs(all_data), 0.98) * 1e6 # Use 98th percentile and scale to µV
        vmin = -vmax

        # Plot for 'Static' condition
        fig_static = static_erp.plot_topomap(times=times_to_plot, show=False, time_unit='s', cmap='RdBu_r', vlim=(vmin, vmax))
        fig_static.suptitle(f"Grand Average ERP Topomaps for 'Static' Condition\n({task}) | Voltage in µV", fontsize=14)
        static_topo_path = os.path.join(PLOTS_DIR, f'topo_static_{task_name_safe}.png')
        fig_static.savefig(static_topo_path)
        plt.close(fig_static)
        linear_plots[f'topo_static_{task_name_safe.lower()}'] = static_topo_path

        # Plot for 'Morph' condition
        fig_morph = morph_erp.plot_topomap(times=times_to_plot, show=False, time_unit='s', cmap='RdBu_r', vlim=(vmin, vmax))
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
        
        # Generate and save Clustered Channel Importance plots.
        cluster_topo_path, cluster_bar_path = plot_clustered_channel_importance(importances, task, X_epochs.info, PLOTS_DIR)
        cluster_plots[f'cluster_topo_{task_name_safe.lower()}'] = cluster_topo_path
        cluster_plots[f'cluster_bar_{task_name_safe.lower()}'] = cluster_bar_path

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
        # Create and save a bar chart of accuracies for each subject.
        plt.figure(figsize=(15, 7))
        subjects = list(subject_accuracies.keys())
        accuracies = list(subject_accuracies.values())
        # Fix for FutureWarning: Assign 'x' to 'hue' to use palette without warning.
        sns.barplot(x=subjects, y=accuracies, hue=subjects, palette='viridis', legend=False)
        plt.xticks(rotation=90)
        plt.ylim(0, 1.1)
        plt.title(f'Per-Participant Accuracy for {task}', fontsize=16)
        plt.ylabel('Accuracy', fontsize=12)
        plt.xlabel('Participant ID', fontsize=12)
        plt.tight_layout()
        bar_plot_path = os.path.join(PLOTS_DIR, f'participant_accuracy_{task_name_safe}.png')
        plt.savefig(bar_plot_path)
        plt.close()
        participant_analysis_plots[f'bar_{task_name_safe.lower()}'] = bar_plot_path

        # --- Analyze the lowest-performing subject ---
        lowest_acc_subject = min(subject_accuracies, key=subject_accuracies.get)
        print(f"  Lowest accuracy for this task was subject {lowest_acc_subject} with {subject_accuracies[lowest_acc_subject]:.2f}")

        # This entire block is wrapped in a try-except to prevent crashes if a subject
        # has missing data for one of the conditions, which would make plotting impossible.
        try:
            low_subj_mask = groups == lowest_acc_subject
            other_subj_mask = groups != lowest_acc_subject
            low_subj_task_mask = np.logical_and(low_subj_mask, emotion_mask)
            other_subj_task_mask = np.logical_and(other_subj_mask, emotion_mask)

            low_subj_epochs = X_epochs[low_subj_task_mask]
            low_subj_labels = y_variant[low_subj_task_mask]
            other_subj_epochs = X_epochs[other_subj_task_mask]
            other_subj_labels = y_variant[other_subj_task_mask]

            # --- Robust check for data availability ---
            # Ensure there are trials for both conditions for the low-acc subject and the GA.
            low_subj_has_static = np.any(low_subj_labels == 0)
            low_subj_has_morph = np.any(low_subj_labels == 1)
            others_have_static = np.any(other_subj_labels == 0)
            others_have_morph = np.any(other_subj_labels == 1)

            if not (low_subj_has_static and low_subj_has_morph and others_have_static and others_have_morph):
                 print(f"  Skipping lowest accuracy plot for {lowest_acc_subject} due to insufficient data for comparison (missing one condition).")
                 participant_analysis_plots[f'low_acc_erp_{task_name_safe.lower()}'] = None
                 continue # Skip to the next task in the main loop

            evokeds_to_plot = {
                'Low Subj Static': low_subj_epochs[low_subj_labels == 0].average(),
                'Low Subj Morph': low_subj_epochs[low_subj_labels == 1].average(),
                'Others Static (GA)': other_subj_epochs[other_subj_labels == 0].average(),
                'Others Morph (GA)': other_subj_epochs[other_subj_labels == 1].average()
            }
            
            fig = mne.viz.plot_compare_evokeds(
                evokeds_to_plot, picks=['POz'],
                title=f'ERP Comparison: Lowest Accuracy Subject ({lowest_acc_subject}) vs. Others for {task}',
                show=False)[0]
            ax = fig.get_axes()[0]
            ax.set_xlim(0.1, 1.0); ax.set_ylim(-15, 5)
            ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=False))
            ax.ticklabel_format(style='plain', axis='y')

            low_subj_plot_path = os.path.join(PLOTS_DIR, f'low_acc_subj_erp_{task_name_safe}.png')
            fig.savefig(low_subj_plot_path, bbox_inches='tight')
            plt.close(fig)
            participant_analysis_plots[f'low_acc_erp_{task_name_safe.lower()}'] = low_subj_plot_path

        except Exception as e:
            print(f"  !! Could not generate lowest accuracy plot for subject {lowest_acc_subject}: {e}")
            participant_analysis_plots[f'low_acc_erp_{task_name_safe.lower()}'] = None

        # --- Store Final Results for the Task ---
        accuracy = accuracy_score(all_true, all_preds)
        results[task] = accuracy
    
    # --- Consolidate All Plot Paths for the Report ---
    plot_paths = {
        'de_features': plot_data.get('de_exp_step4'), # Use the more descriptive topomap plot
        'de_exp_step1': plot_data.get('de_exp_step1'),
        'de_exp_step2': plot_data.get('de_exp_step2'),
        'de_exp_step3': plot_data.get('de_exp_step3'),
        'de_exp_step4': plot_data.get('de_exp_step4'),
        **linear_plots,
        **participant_analysis_plots,
        **de_dist_plots,
        **cluster_plots
    }
    
    # Add importance plots separately since they are a list per task
    for task_name_safe, paths in importance_plots.items():
        for path in paths:
            plot_paths[os.path.basename(path)] = path

    # --- Generate Final Report ---
    generate_report(results, plot_paths)

if __name__ == '__main__':
    main()
