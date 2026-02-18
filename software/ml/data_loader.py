#!/usr/bin/env python3
"""
data_loader.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Multi-format data loader for ML pipeline.
Handles our CSV format, Adamatzky .txt, Buffi HDF5, and synthetic manifest files.
Segments recordings into fixed-length windows with labels.

Usage:
    from data_loader import load_all_data
    X, y = load_all_data(synthetic_dir='data/synthetic',
                         adamatzky_dir='data/external/adamatzky',
                         buffi_dir='data/external/buffi')
"""

import numpy as np
import csv
import os
import glob
from typing import Tuple, Optional, List

from augmentation import clean_signal, normalize_signal

SAMPLE_RATE = 10.0  # Hz (our standard)
WINDOW_SECONDS = 60  # 60-second windows
WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)  # 600 samples
OVERLAP = 0.5  # 50% overlap


def load_our_csv(filepath: str) -> np.ndarray:
    """
    Load a recording in our CSV format.

    Format: timestamp_ms, adc_raw, voltage_mV

    Returns:
        1D numpy array of voltage values (mV)
    """
    voltages = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith('#') or row[0].startswith('timestamp'):
                continue
            try:
                if len(row) >= 3:
                    voltages.append(float(row[2]))  # voltage_mV column
                elif len(row) >= 2:
                    voltages.append(float(row[1]))
            except (ValueError, IndexError):
                continue
    return np.array(voltages)


def load_adamatzky_txt(filepath: str, target_rate: float = SAMPLE_RATE,
                       max_rows: int = 0) -> List[np.ndarray]:
    """
    Load an Adamatzky dataset .txt file.

    These files have tab/space-delimited columns with 8 electrode pair readings.
    Native sample rate is ~1 Hz. We resample to our target rate via interpolation.

    Each column is treated as a separate recording channel.

    Args:
        max_rows: Maximum rows to read per file (0 = unlimited).
                  36,000 rows = 10 hours at 1 Hz (MacBook-safe default).

    Returns:
        List of 1D arrays (one per electrode channel)
    """
    data = []
    n_cols = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('Time') or line.startswith('Differential'):
                continue
            try:
                values = [float(x) for x in line.split()]
                if not values:
                    continue
                # Lock column count to first valid row
                if n_cols is None:
                    n_cols = len(values)
                if len(values) == n_cols:
                    data.append(values)
                    if max_rows > 0 and len(data) >= max_rows:
                        break
            except ValueError:
                continue

    if not data:
        return []

    data = np.array(data)
    native_rate = 1.0  # Adamatzky data is ~1 Hz

    channels = []
    for col in range(data.shape[1]):
        channel = data[:, col]

        # Clean before resampling (handles NaN, outliers, flat-lines)
        channel = clean_signal(channel)

        # Resample from native_rate to target_rate via linear interpolation
        if native_rate != target_rate:
            n_original = len(channel)
            duration = n_original / native_rate
            n_target = int(duration * target_rate)
            t_original = np.linspace(0, duration, n_original)
            t_target = np.linspace(0, duration, n_target)
            channel = np.interp(t_target, t_original, channel)

        # Normalize to z-score (different species have different voltage scales)
        channel = normalize_signal(channel)

        channels.append(channel)

    return channels


def segment_windows(signal: np.ndarray,
                    window_samples: int = WINDOW_SAMPLES,
                    overlap: float = OVERLAP) -> np.ndarray:
    """
    Segment a signal into overlapping windows.

    Args:
        signal: 1D array
        window_samples: Samples per window
        overlap: Fraction of overlap between windows (0-1)

    Returns:
        2D array of shape (n_windows, window_samples)
    """
    if len(signal) < window_samples:
        return np.array([]).reshape(0, window_samples)

    step = int(window_samples * (1 - overlap))
    if step < 1:
        step = 1

    n_windows = (len(signal) - window_samples) // step + 1
    windows = np.zeros((n_windows, window_samples))

    for i in range(n_windows):
        start = i * step
        windows[i] = signal[start:start + window_samples]

    return windows


def load_synthetic_data(synthetic_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load synthetic dataset using the manifest file.

    Returns:
        X: 2D array of windows (n_windows, window_samples)
        y: 1D array of labels (0 or 1)
    """
    manifest_path = os.path.join(synthetic_dir, 'manifest.csv')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"No manifest.csv found in {synthetic_dir}")

    all_windows = []
    all_labels = []

    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = os.path.join(synthetic_dir, row['filename'])
            label = int(row['label'])

            if not os.path.exists(filepath):
                print(f"  Warning: {filepath} not found, skipping")
                continue

            signal = load_our_csv(filepath)
            signal = clean_signal(signal)
            signal = normalize_signal(signal)
            windows = segment_windows(signal)

            if windows.shape[0] > 0:
                all_windows.append(windows)
                all_labels.extend([label] * windows.shape[0])

    if not all_windows:
        return np.array([]).reshape(0, WINDOW_SAMPLES), np.array([])

    X = np.vstack(all_windows)
    y = np.array(all_labels)
    return X, y


def load_adamatzky_data(adamatzky_dir: str,
                        max_rows: int = 36000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all Adamatzky dataset files from a directory.

    All Adamatzky data is labeled as active fungal signal (label=1).

    Args:
        max_rows: Max rows per .txt file (0 = unlimited).
                  Default 36,000 = 10 hours at 1 Hz (MacBook-safe).

    Returns:
        X: 2D array of windows (n_windows, window_samples)
        y: 1D array of labels (all 1s)
    """
    all_windows = []

    # Find all .txt files
    patterns = [os.path.join(adamatzky_dir, '**', '*.txt')]
    txt_files = []
    for pattern in patterns:
        txt_files.extend(glob.glob(pattern, recursive=True))

    if not txt_files:
        print(f"  No .txt files found in {adamatzky_dir}")
        return np.array([]).reshape(0, WINDOW_SAMPLES), np.array([])

    # Skip __MACOSX metadata files
    txt_files = [f for f in txt_files if '__MACOSX' not in f]

    if max_rows > 0:
        print(f"  Row cap per file: {max_rows:,} ({max_rows/3600:.0f} hours at 1 Hz)")

    for filepath in sorted(txt_files):
        print(f"  Loading {os.path.basename(filepath)}...")
        channels = load_adamatzky_txt(filepath, max_rows=max_rows)
        for channel in channels:
            windows = segment_windows(channel)
            if windows.shape[0] > 0:
                all_windows.append(windows)

    if not all_windows:
        return np.array([]).reshape(0, WINDOW_SAMPLES), np.array([])

    X = np.vstack(all_windows)
    y = np.ones(X.shape[0], dtype=int)  # All label=1 (active fungal)
    return X, y


def load_buffi_hdf5(filepath: str, target_rate: float = SAMPLE_RATE,
                    max_duration_sec: float = 3600.0) -> List[np.ndarray]:
    """
    Load a Buffi et al. 2025 HDF5 file (from PicoLog 6).

    Each file contains multi-channel recordings of Fusarium oxysporum
    at ~17 Hz (60 ms conversion time). Files are HUGE (~17M samples each),
    so we take a random segment of max_duration_sec per channel to stay
    within memory limits.

    Returns:
        List of 1D arrays (one per channel), cleaned and normalized
    """
    try:
        import h5py
    except ImportError:
        print("  Warning: h5py not installed. Run: pip install h5py")
        return []

    # Buffi native rate: ~17 Hz (60 ms per sample)
    native_rate = 17.0
    max_native_samples = int(max_duration_sec * native_rate)

    result = []
    with h5py.File(filepath, 'r') as f:
        for key in sorted(f.keys()):
            item = f[key]
            if not (isinstance(item, h5py.Dataset) and item.ndim == 1
                    and item.dtype.kind in ('f', 'i', 'u')):
                continue
            if item.shape[0] < 1000:
                continue

            # Take a random segment to avoid loading entire 17M-sample array
            total = item.shape[0]
            if total > max_native_samples:
                start = np.random.randint(0, total - max_native_samples)
                data = np.array(item[start:start + max_native_samples], dtype=float)
            else:
                data = np.array(item, dtype=float)

            # Clean raw signal
            data = clean_signal(data)

            # Resample to target rate
            n_original = len(data)
            duration = n_original / native_rate
            n_target = int(duration * target_rate)
            t_original = np.linspace(0, duration, n_original)
            t_target = np.linspace(0, duration, n_target)
            resampled = np.interp(t_target, t_original, data)

            # Normalize to z-score (consistent with Adamatzky handling)
            resampled = normalize_signal(resampled)

            result.append(resampled)

    if not result:
        print(f"  Warning: No numeric datasets found in {filepath}")

    return result


def load_buffi_data(buffi_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all Buffi HDF5 files from a directory.

    All Buffi data is labeled as active fungal signal (label=1).

    Returns:
        X: 2D array of windows (n_windows, window_samples)
        y: 1D array of labels (all 1s)
    """
    all_windows = []

    hdf5_files = glob.glob(os.path.join(buffi_dir, '*.hdf5'))
    hdf5_files += glob.glob(os.path.join(buffi_dir, '*.h5'))

    if not hdf5_files:
        print(f"  No HDF5 files found in {buffi_dir}")
        return np.array([]).reshape(0, WINDOW_SAMPLES), np.array([])

    for filepath in sorted(hdf5_files):
        print(f"  Loading {os.path.basename(filepath)}...")
        channels = load_buffi_hdf5(filepath)
        for channel in channels:
            windows = segment_windows(channel)
            if windows.shape[0] > 0:
                all_windows.append(windows)
        if channels:
            print(f"    {len(channels)} channels extracted")

    if not all_windows:
        return np.array([]).reshape(0, WINDOW_SAMPLES), np.array([])

    X = np.vstack(all_windows)
    y = np.ones(X.shape[0], dtype=int)  # All label=1
    return X, y


def load_all_data(synthetic_dir: Optional[str] = None,
                  adamatzky_dir: Optional[str] = None,
                  buffi_dir: Optional[str] = None,
                  our_csv_files: Optional[List[Tuple[str, int]]] = None,
                  adamatzky_max_rows: int = 36000
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and combine all available data sources.

    Args:
        synthetic_dir: Path to synthetic data directory (with manifest.csv)
        adamatzky_dir: Path to Adamatzky dataset directory
        buffi_dir: Path to Buffi HDF5 dataset directory
        our_csv_files: List of (filepath, label) tuples for our recordings
        adamatzky_max_rows: Max rows per Adamatzky file (0 = unlimited)

    Returns:
        X: 2D array of windows (n_total_windows, window_samples)
        y: 1D array of labels
    """
    all_X = []
    all_y = []

    # 1. Synthetic data
    if synthetic_dir and os.path.exists(synthetic_dir):
        print(f"Loading synthetic data from {synthetic_dir}...")
        X_syn, y_syn = load_synthetic_data(synthetic_dir)
        if X_syn.shape[0] > 0:
            print(f"  {X_syn.shape[0]} windows (label 0: {np.sum(y_syn == 0)}, label 1: {np.sum(y_syn == 1)})")
            all_X.append(X_syn)
            all_y.append(y_syn)

    # 2. Adamatzky data
    if adamatzky_dir and os.path.exists(adamatzky_dir):
        print(f"Loading Adamatzky data from {adamatzky_dir}...")
        X_adam, y_adam = load_adamatzky_data(adamatzky_dir, max_rows=adamatzky_max_rows)
        if X_adam.shape[0] > 0:
            print(f"  {X_adam.shape[0]} windows (all label 1)")
            all_X.append(X_adam)
            all_y.append(y_adam)

    # 3. Buffi data
    if buffi_dir and os.path.exists(buffi_dir):
        print(f"Loading Buffi data from {buffi_dir}...")
        X_buffi, y_buffi = load_buffi_data(buffi_dir)
        if X_buffi.shape[0] > 0:
            print(f"  {X_buffi.shape[0]} windows (all label 1)")
            all_X.append(X_buffi)
            all_y.append(y_buffi)

    # 4. Our own recordings
    if our_csv_files:
        print("Loading our recordings...")
        for filepath, label in our_csv_files:
            signal = load_our_csv(filepath)
            signal = clean_signal(signal)
            signal = normalize_signal(signal)
            windows = segment_windows(signal)
            if windows.shape[0] > 0:
                all_X.append(windows)
                all_y.append(np.full(windows.shape[0], label, dtype=int))
                print(f"  {filepath}: {windows.shape[0]} windows (label={label})")

    if not all_X:
        print("Warning: No data loaded!")
        return np.array([]).reshape(0, WINDOW_SAMPLES), np.array([])

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    print(f"\nTotal: {X.shape[0]} windows")
    print(f"  Label 0 (inactive): {np.sum(y == 0)}")
    print(f"  Label 1 (active):   {np.sum(y == 1)}")

    return X, y
