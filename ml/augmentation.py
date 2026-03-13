#!/usr/bin/env python3
"""
augmentation.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Data augmentation and signal cleaning for the ML pipeline.
Applies noise injection, time-shifting, amplitude scaling, and proper
NaN/outlier handling to make models robust to real-world signal variation.
"""

import numpy as np
from typing import Tuple


# ============================================================
# SIGNAL CLEANING
# ============================================================

def clean_signal(signal: np.ndarray, max_sigma: float = 5.0) -> np.ndarray:
    """
    Clean a raw 1D signal:
      1. Interpolate over NaN/Inf values
      2. Cap outliers beyond max_sigma standard deviations
      3. Detect and interpolate flat-line segments (>50 consecutive identical values)

    Args:
        signal: 1D array of voltage samples
        max_sigma: Outlier threshold in standard deviations

    Returns:
        Cleaned 1D array (same length)
    """
    sig = signal.copy().astype(float)

    # 1. Handle NaN / Inf — linear interpolation
    bad_mask = ~np.isfinite(sig)
    if np.all(bad_mask):
        return np.zeros_like(sig)  # entirely bad signal
    if np.any(bad_mask):
        good_idx = np.where(~bad_mask)[0]
        bad_idx = np.where(bad_mask)[0]
        sig[bad_idx] = np.interp(bad_idx, good_idx, sig[good_idx])

    # 2. Cap outliers
    mu = np.mean(sig)
    sigma = np.std(sig)
    if sigma > 0:
        lower = mu - max_sigma * sigma
        upper = mu + max_sigma * sigma
        sig = np.clip(sig, lower, upper)

    # 3. Detect flat-line segments (>50 identical values in a row)
    flat_threshold = 50
    diffs = np.diff(sig)
    flat_runs = np.zeros(len(sig), dtype=bool)
    run_start = 0
    for i in range(1, len(diffs)):
        if diffs[i] == diffs[run_start] == 0:
            if (i - run_start + 1) >= flat_threshold:
                flat_runs[run_start:i + 2] = True
        else:
            run_start = i

    if np.any(flat_runs):
        good_idx = np.where(~flat_runs)[0]
        if len(good_idx) > 1:
            flat_idx = np.where(flat_runs)[0]
            sig[flat_idx] = np.interp(flat_idx, good_idx, sig[good_idx])

    return sig


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalize a signal to zero mean, unit variance.
    Robust to constant signals (returns zeros).
    """
    mu = np.mean(signal)
    sigma = np.std(signal)
    if sigma < 1e-12:
        return np.zeros_like(signal)
    return (signal - mu) / sigma


# ============================================================
# DATA AUGMENTATION
# ============================================================

def add_gaussian_noise(window: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """
    Add Gaussian noise at a given SNR level.

    Think of it like adding background hum to your mushroom recording —
    the signal is still there, just a bit noisier.

    Args:
        window: 1D signal window
        snr_db: Signal-to-noise ratio in dB (lower = more noise)

    Returns:
        Noisy copy of the window
    """
    sig_power = np.mean(window ** 2)
    if sig_power < 1e-12:
        return window.copy()
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(window))
    return window + noise


def time_shift(window: np.ndarray, max_shift_frac: float = 0.1) -> np.ndarray:
    """
    Circular time-shift the window by a random amount.

    Like if your recording started a few seconds earlier or later —
    the signal pattern is the same, just shifted.

    Args:
        window: 1D signal window
        max_shift_frac: Maximum shift as fraction of window length

    Returns:
        Shifted copy
    """
    max_shift = int(len(window) * max_shift_frac)
    if max_shift < 1:
        return window.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(window, shift)


def amplitude_scale(window: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
    """
    Randomly scale the amplitude.

    Like if your amplifier gain drifted slightly — same shape, different size.

    Args:
        window: 1D signal window
        scale_range: (min_scale, max_scale) uniform range

    Returns:
        Scaled copy
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return window * scale


def add_baseline_wander(window: np.ndarray, max_amplitude: float = 0.3,
                        max_freq: float = 0.05) -> np.ndarray:
    """
    Add slow baseline wander (simulates electrode drift).

    Like the ground slowly shifting under your measurement — a slow wobble
    that doesn't affect the fast spikes but shifts the whole signal up/down.

    Args:
        window: 1D signal window
        max_amplitude: Maximum wander amplitude (relative to signal std)
        max_freq: Maximum wander frequency (Hz, at 10 Hz sample rate)

    Returns:
        Window with added baseline wander
    """
    t = np.linspace(0, len(window) / 10.0, len(window))  # assume 10 Hz
    freq = np.random.uniform(0.005, max_freq)
    phase = np.random.uniform(0, 2 * np.pi)
    amplitude = np.random.uniform(0, max_amplitude) * np.std(window)
    wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return window + wander


def augment_window(window: np.ndarray, n_augmented: int = 3) -> np.ndarray:
    """
    Generate augmented copies of a single window by applying random
    combinations of transformations.

    Args:
        window: 1D signal window (e.g., 600 samples)
        n_augmented: Number of augmented copies to generate

    Returns:
        2D array of shape (n_augmented, window_length)
    """
    augmented = np.zeros((n_augmented, len(window)))

    for i in range(n_augmented):
        w = window.copy()

        # Each augmentation has a probability of being applied
        if np.random.random() < 0.7:
            snr = np.random.uniform(15, 30)  # 15-30 dB SNR
            w = add_gaussian_noise(w, snr_db=snr)

        if np.random.random() < 0.5:
            w = time_shift(w, max_shift_frac=0.1)

        if np.random.random() < 0.5:
            w = amplitude_scale(w, scale_range=(0.8, 1.2))

        if np.random.random() < 0.4:
            w = add_baseline_wander(w, max_amplitude=0.3)

        augmented[i] = w

    return augmented


def augment_dataset(X: np.ndarray, y: np.ndarray,
                    n_augmented_per_sample: int = 2,
                    balance_classes: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment an entire dataset.

    If balance_classes is True, augments the minority class more heavily
    so both classes end up with roughly equal counts.

    Args:
        X: 2D array (n_samples, window_length) — original windows
        y: 1D array of labels
        n_augmented_per_sample: Base number of augmented copies per sample
        balance_classes: Whether to oversample the minority class

    Returns:
        X_aug: Original + augmented windows
        y_aug: Corresponding labels
    """
    n_orig = X.shape[0]
    window_length = X.shape[1]
    classes, counts = np.unique(y, return_counts=True)

    # --- Pre-calculate total augmented count per original sample ---
    # Build an array of how many augmented copies each sample gets
    aug_counts = np.full(n_orig, n_augmented_per_sample, dtype=int)

    if balance_classes and len(classes) >= 2:
        majority_count = counts.max()
        for cls, count in zip(classes, counts):
            if count < majority_count:
                deficit = majority_count - count
                n_aug = max(n_augmented_per_sample, int(np.ceil(deficit / count)) + 1)
                mask = y == cls
                aug_counts[mask] = n_aug

    total_aug = int(np.sum(aug_counts))
    total_out = n_orig + total_aug

    # --- Pre-allocate output arrays ---
    X_aug = np.empty((total_out, window_length), dtype=X.dtype)
    y_aug = np.empty(total_out, dtype=y.dtype)

    # Copy originals into the front
    X_aug[:n_orig] = X
    y_aug[:n_orig] = y

    # Fill augmented samples with a write cursor
    cursor = n_orig
    for j in range(n_orig):
        n_aug = aug_counts[j]
        aug_windows = augment_window(X[j], n_augmented=n_aug)
        X_aug[cursor:cursor + n_aug] = aug_windows
        y_aug[cursor:cursor + n_aug] = y[j]
        cursor += n_aug

    return X_aug, y_aug


# ============================================================
# MIXUP AUGMENTATION (in-batch, applied during training)
# ============================================================

def mixup_batch(X_batch, y_batch, alpha=0.2):
    """
    Apply Mixup augmentation to a training batch.

    Creates convex combinations of pairs of windows and their labels:
      x_mix = λ·x_i + (1-λ)·x_j
      loss  = λ·criterion(logits, y_i) + (1-λ)·criterion(logits, y_j)

    Forces the model to learn smoother decision boundaries — especially
    useful for similar stimuli (cycloheximide ↔ sodiumazide confusion).

    Args:
        X_batch: FloatTensor (batch, 1, window_len) — input windows
        y_batch: LongTensor (batch,) — class labels
        alpha:   Beta distribution parameter. Higher = more mixing.
                 alpha=0.2 gives λ mostly near 0 or 1 (mild mixing).

    Returns:
        X_mix:  Mixed input batch (same shape as X_batch)
        y_a:    Original labels (LongTensor)
        y_b:    Shuffled labels for the mix targets (LongTensor)
        lam:    Scalar mixing coefficient in [0, 1]
    """
    import torch
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(X_batch.size(0), device=X_batch.device)
    X_mix = lam * X_batch + (1.0 - lam) * X_batch[idx]
    return X_mix, y_batch, y_batch[idx], lam
