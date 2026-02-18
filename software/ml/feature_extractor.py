#!/usr/bin/env python3
"""
feature_extractor.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Extracts a 26-feature vector from each signal window.
Wraps signal_processor.py for all DSP — no duplication.
"""

import numpy as np
from scipy.stats import skew, kurtosis
import sys
import os

# Add parent directory to path for signal_processor import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'processing'))
from signal_processor import SignalProcessor


# Feature names in extraction order
FEATURE_NAMES = [
    # Time domain (6)
    'rms', 'variance', 'zero_crossing_rate', 'peak_to_peak', 'skewness', 'kurtosis',
    # Spike features (5)
    'spike_count', 'mean_spike_amplitude', 'spike_rate_per_min', 'mean_isi', 'isi_std',
    # Frequency domain (7)
    'power_ultra_low', 'power_low', 'power_mid', 'power_high',
    'total_power', 'dominant_freq', 'spectral_centroid',
    # STFT-derived (4)
    'stft_max_power', 'stft_mean_power', 'stft_power_std', 'spectral_entropy',
    # Statistical shape (4)
    'median', 'iqr', 'hurst_estimate', 'autocorr_lag10',
]

# Cache processors by sample_rate to avoid re-creating filters every call
_processor_cache = {}


def _get_processor(sample_rate: float) -> SignalProcessor:
    """Get or create a cached SignalProcessor for the given sample rate."""
    if sample_rate not in _processor_cache:
        _processor_cache[sample_rate] = SignalProcessor(sample_rate=sample_rate)
    return _processor_cache[sample_rate]


def extract_features(window: np.ndarray, sample_rate: float = 10.0,
                     processor: SignalProcessor = None) -> np.ndarray:
    """
    Extract 26 features from a single signal window.

    Args:
        window: 1D array of voltage samples (e.g., 600 samples = 60 sec at 10 Hz)
        sample_rate: Sampling rate in Hz
        processor: Optional pre-created SignalProcessor (avoids cache lookup)

    Returns:
        1D numpy array of 26 features
    """
    if processor is None:
        processor = _get_processor(sample_rate)
    features = np.zeros(len(FEATURE_NAMES))

    # --- Time domain (6) ---
    features[0] = np.sqrt(np.mean(window ** 2))           # RMS
    features[1] = np.var(window)                           # Variance
    zero_crossings = np.sum(np.diff(np.sign(window - np.mean(window))) != 0)
    features[2] = zero_crossings / len(window)             # Zero-crossing rate
    features[3] = np.ptp(window)                           # Peak-to-peak
    features[4] = skew(window) if len(window) > 2 else 0.0  # Skewness
    features[5] = kurtosis(window) if len(window) > 2 else 0.0  # Kurtosis

    # --- Spike features (5) ---
    filtered = processor.process(window)
    spikes, spike_info = processor.detect_spikes(filtered)

    features[6] = spike_info['count']                      # Spike count
    if spike_info['count'] > 0:
        features[7] = np.mean(spike_info['amplitudes'])    # Mean spike amplitude
    else:
        features[7] = 0.0
    features[8] = spike_info['rate_per_minute']            # Spike rate/min

    if len(spikes) > 1:
        isis = np.diff(spikes) / sample_rate               # Inter-spike intervals (sec)
        features[9] = np.mean(isis)                        # Mean ISI
        features[10] = np.std(isis)                        # ISI std
    else:
        features[9] = 0.0
        features[10] = 0.0

    # --- Frequency domain (7) + STFT-derived (4) ---
    # Compute spectrogram ONCE and derive all spectral features from it
    if len(window) >= 64:
        f, t, power = processor.compute_spectrogram(window)
        avg_power = np.mean(power, axis=1)
        total_power = np.sum(avg_power)

        # Band powers (inline — avoids get_frequency_bands calling compute_spectrogram again)
        bands = {
            'ultra_low': (0.01, 0.05),
            'low': (0.05, 0.2),
            'mid': (0.2, 0.5),
            'high': (0.5, 2.0),
        }
        for idx, (band_name, (f_low, f_high)) in enumerate(bands.items()):
            mask = (f >= f_low) & (f < f_high)
            features[11 + idx] = np.sum(avg_power[mask]) if np.any(mask) else 0.0

        features[15] = total_power                          # Total power

        # Dominant frequency (from same avg_power — avoids get_dominant_frequency STFT)
        peak_idx = np.argmax(avg_power)
        features[16] = f[peak_idx]                          # Dominant frequency

        # Spectral centroid
        if total_power > 0:
            features[17] = np.sum(f * avg_power) / total_power
        else:
            features[17] = 0.0

        # STFT-derived features (from same power array)
        features[18] = np.max(power)                        # Max STFT power
        features[19] = np.mean(power)                       # Mean STFT power
        features[20] = np.std(power)                        # STFT power std

        # Spectral entropy (normalized)
        flat_power = power.flatten()
        flat_power = flat_power + 1e-12  # avoid log(0)
        p_norm = flat_power / np.sum(flat_power)
        entropy = -np.sum(p_norm * np.log2(p_norm))
        max_entropy = np.log2(len(p_norm))
        features[21] = entropy / max_entropy if max_entropy > 0 else 0.0
    # else: features 11-21 stay 0.0

    # --- Statistical shape (4) ---
    features[22] = np.median(window)                       # Median
    features[23] = np.percentile(window, 75) - np.percentile(window, 25)  # IQR

    # Hurst exponent estimate (rescaled range method, simplified)
    features[24] = _estimate_hurst(window)

    # Autocorrelation at lag 10
    if len(window) > 10:
        w_centered = window - np.mean(window)
        var = np.var(window)
        if var > 0:
            features[25] = np.correlate(w_centered[:-10], w_centered[10:])[0] / (len(w_centered[:-10]) * var)
        else:
            features[25] = 0.0
    else:
        features[25] = 0.0

    return features


def _estimate_hurst(ts: np.ndarray) -> float:
    """Estimate Hurst exponent using rescaled range (R/S) method."""
    n = len(ts)
    if n < 20:
        return 0.5  # default for too-short series

    max_k = min(n // 2, 128)
    sizes = []
    rs_values = []

    for k in [16, 32, 64, 128]:
        if k > max_k:
            break
        n_chunks = n // k
        if n_chunks < 1:
            break

        rs_list = []
        for i in range(n_chunks):
            chunk = ts[i * k:(i + 1) * k]
            mean_c = np.mean(chunk)
            devs = np.cumsum(chunk - mean_c)
            r = np.max(devs) - np.min(devs)
            s = np.std(chunk, ddof=1) if np.std(chunk, ddof=1) > 0 else 1e-12
            rs_list.append(r / s)

        sizes.append(k)
        rs_values.append(np.mean(rs_list))

    if len(sizes) < 2:
        return 0.5

    log_sizes = np.log(sizes)
    log_rs = np.log(np.array(rs_values) + 1e-12)
    # Linear regression slope = Hurst exponent
    coeffs = np.polyfit(log_sizes, log_rs, 1)
    hurst = np.clip(coeffs[0], 0.0, 1.0)
    return hurst


def extract_features_batch(windows: np.ndarray, sample_rate: float = 10.0) -> np.ndarray:
    """
    Extract features from multiple windows.

    Args:
        windows: 2D array of shape (n_windows, window_length)
        sample_rate: Sampling rate in Hz

    Returns:
        2D numpy array of shape (n_windows, 26)
    """
    # Create one processor for the entire batch
    processor = SignalProcessor(sample_rate=sample_rate)
    n = windows.shape[0]
    features = np.zeros((n, len(FEATURE_NAMES)))
    for i in range(n):
        features[i] = extract_features(windows[i], sample_rate, processor=processor)
    return features
