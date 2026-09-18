import numpy as np
import mne
from scipy import signal, fftpack
from scipy.stats import pearsonr

def load_eeg_data(file_path):
    raw_data = mne.io.read_raw_bdf(file_path, preload=True)
    data = raw_data.get_data()
    # events = mne.events_from_annotations(raw, event_id=lambda x: int(x), verbose=False)[0]
    # ch_picks = mne.pick_channels(raw.ch_names, chs, ordered=True)
    # epochs = mne.Epochs(raw, events, event_id=labels, tmin=delay, tmax=delay + stimlen, baseline=None, picks=ch_picks,
    #                         preload=True, verbose=False)
    return data

def generate_sine_wave(length, frequency=10, phase=1.05, sample_rate=250):
    t = np.arange(length) / sample_rate  # Time vector
    sine_wave = np.sin(2 * np.pi * frequency * t + phase)
    return sine_wave

def calculate_pearson_correlation(X1, X2):
    X1_flat = X1.flatten()
    X2_flat = X2.flatten()

    correlation, _ = pearsonr(X1_flat, X2_flat)
    return correlation


if __name__ == "__main__":

    file_path = "C:\\Users\\11402\\Desktop\\Light_text2.bdf"
    X1 = load_eeg_data(file_path)
    print("X1:", X1.shape)

    srate = 250
    f0 = 50
    Q = 30
    b_notch, a_notch = signal.iirnotch(f0, Q, fs=srate)
    X1 = signal.filtfilt(b_notch, a_notch, X1)

    filtered_data = mne.filter.filter_data(X1, sfreq=srate, l_freq=5, h_freq=20)
    print("filtered_data.shape:", filtered_data.shape)

    length = filtered_data.shape[1]
    X2 = generate_sine_wave(length)

    correlation = calculate_pearson_correlation(filtered_data[0], X2)
    # correlation = calculate_pearson_correlation(X2, X2)

    
    print(f"Pearson correlation between X1 and X2: {correlation:.4f}")
