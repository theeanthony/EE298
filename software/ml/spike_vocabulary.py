#!/usr/bin/env python3
"""
spike_vocabulary.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Discover fungal electrical "vocabulary" using Adamatzky's methodology:
  1. Detect spikes in raw signals
  2. Group consecutive spikes into "words" (spike trains) using temporal threshold
  3. Extract per-word features (length, amplitude, duration, ISI, etc.)
  4. Cluster words into vocabulary types via k-means
  5. Label signal windows by their dominant word type

Based on:
  - Adamatzky (2022) "Language of fungi derived from their electrical spiking activity"
  - MIT CSAIL decipherment approach (embed patterns → discover structure)

Usage:
    python spike_vocabulary.py                    # Discover vocabulary (k=50)
    python spike_vocabulary.py --n-clusters 30    # Try different k
    python spike_vocabulary.py --theta 2.0        # Alternate temporal threshold
    python spike_vocabulary.py --analyze          # Show stats from saved model
    python spike_vocabulary.py --elbow            # Plot elbow curve for k selection
"""

import numpy as np
import os
import sys
import json
import argparse
import gc

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent dir so we can import signal_processor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'processing'))

from signal_processor import SignalProcessor
from data_loader import load_adamatzky_txt, segment_windows, WINDOW_SAMPLES, SAMPLE_RATE

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
ADAMATZKY_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'adamatzky')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'ml_results')

VOCAB_KMEANS_PATH = os.path.join(MODELS_DIR, 'vocabulary_kmeans.joblib')
VOCAB_SCALER_PATH = os.path.join(MODELS_DIR, 'vocabulary_scaler.joblib')
VOCAB_LABELS_PATH = os.path.join(MODELS_DIR, 'vocabulary_labels.npz')
VOCAB_STATS_PATH = os.path.join(PLOTS_DIR, 'vocabulary_stats.json')
VOCAB_PLOT_PATH = os.path.join(PLOTS_DIR, 'vocabulary_analysis.png')


# ============================================================
# SPIKE EXTRACTION (reuses SignalProcessor)
# ============================================================

def extract_spikes(signal, sample_rate=SAMPLE_RATE, processor=None):
    """
    Extract and characterize spikes from a signal.

    Args:
        signal: 1D array of (already normalized) signal values
        sample_rate: Sampling rate in Hz
        processor: Reusable SignalProcessor instance (created if None)

    Returns:
        List of spike dicts with: index, time, amplitude, rise_time,
        fall_time, duration, area
    """
    if processor is None:
        processor = SignalProcessor(sample_rate=sample_rate)

    # Bandpass filter to isolate fungal band
    filtered = processor.process(signal)

    # Detect spike indices
    spike_indices, _ = processor.detect_spikes(filtered)

    if len(spike_indices) == 0:
        return []

    # Characterize each spike
    spikes = processor.characterize_spikes(filtered, spike_indices)
    return spikes


# ============================================================
# WORD GROUPING (Adamatzky's temporal threshold method)
# ============================================================

def group_spikes_into_words(spikes, theta_multiplier=1.0):
    """
    Group consecutive spikes into "words" using Adamatzky's method.

    A word = a train of closely-spaced spikes. If the gap between
    consecutive spikes > theta (average ISI × multiplier), that's
    a word boundary.

    Args:
        spikes: List of spike dicts (from extract_spikes)
        theta_multiplier: Multiplier for average ISI threshold.
                         Adamatzky used 1.0 and 2.0.

    Returns:
        List of words, where each word is a list of spike dicts.
    """
    if len(spikes) < 2:
        # Single spike = single word
        return [spikes] if spikes else []

    # Compute inter-spike intervals
    times = np.array([s['time'] for s in spikes])
    isis = np.diff(times)

    if len(isis) == 0:
        return [spikes]

    # Temporal threshold: average ISI × multiplier
    theta = np.mean(isis) * theta_multiplier

    # Group spikes into words
    words = []
    current_word = [spikes[0]]

    for i in range(1, len(spikes)):
        gap = spikes[i]['time'] - spikes[i - 1]['time']
        if gap <= theta:
            current_word.append(spikes[i])
        else:
            words.append(current_word)
            current_word = [spikes[i]]

    # Don't forget the last word
    words.append(current_word)

    return words


# ============================================================
# PER-WORD FEATURE EXTRACTION
# ============================================================

def extract_word_features(word):
    """
    Extract features for a single word (spike train).

    Returns 7 features per word:
        n_spikes, total_duration, mean_amplitude, max_amplitude,
        amplitude_std, mean_isi, rise_fall_ratio

    Args:
        word: List of spike dicts (one "word")

    Returns:
        1D numpy array of 7 features
    """
    n_spikes = len(word)
    amplitudes = np.array([s['amplitude'] for s in word])
    mean_amplitude = np.mean(amplitudes)
    max_amplitude = np.max(amplitudes)
    amplitude_std = np.std(amplitudes) if n_spikes > 1 else 0.0

    # Duration: time from first to last spike
    if n_spikes > 1:
        total_duration = word[-1]['time'] - word[0]['time']
        times = np.array([s['time'] for s in word])
        isis = np.diff(times)
        mean_isi = np.mean(isis)
    else:
        total_duration = word[0].get('duration', 0.0)
        mean_isi = 0.0

    # Rise/fall asymmetry
    rise_times = np.array([s['rise_time'] for s in word])
    fall_times = np.array([s['fall_time'] for s in word])
    # Avoid division by zero
    denom = fall_times + 1e-12
    rise_fall_ratio = np.mean(rise_times / denom)

    return np.array([
        n_spikes,
        total_duration,
        mean_amplitude,
        max_amplitude,
        amplitude_std,
        mean_isi,
        rise_fall_ratio,
    ])


N_WORD_FEATURES = 7
WORD_FEATURE_NAMES = [
    'n_spikes', 'total_duration', 'mean_amplitude', 'max_amplitude',
    'amplitude_std', 'mean_isi', 'rise_fall_ratio',
]


# ============================================================
# VOCABULARY DISCOVERY (k-means clustering)
# ============================================================

def discover_vocabulary(adamatzky_dir=ADAMATZKY_DIR, n_clusters=50,
                        max_rows=36000, theta_multiplier=1.0):
    """
    Discover fungal signal vocabulary from Adamatzky data.

    Pipeline: load channels → extract spikes → group into words →
    extract features → standardize → k-means cluster.

    Args:
        adamatzky_dir: Path to Adamatzky .txt files
        n_clusters: Number of word clusters (vocabulary size)
        max_rows: Max rows per file (MacBook safety)
        theta_multiplier: Temporal threshold multiplier

    Returns:
        (scaler, kmeans, all_word_features, all_word_labels)
    """
    import glob as globmod

    processor = SignalProcessor(sample_rate=SAMPLE_RATE)

    # Find all .txt files
    txt_files = []
    for pattern in [os.path.join(adamatzky_dir, '**', '*.txt')]:
        txt_files.extend(globmod.glob(pattern, recursive=True))
    txt_files = [f for f in txt_files if '__MACOSX' not in f]

    if not txt_files:
        print(f"  No .txt files found in {adamatzky_dir}")
        return None, None, None, None

    print(f"  Found {len(txt_files)} Adamatzky files")
    if max_rows > 0:
        print(f"  Row cap per file: {max_rows:,}")

    # Collect words across all channels
    all_word_features = []
    total_spikes = 0
    total_words = 0

    for filepath in sorted(txt_files):
        basename = os.path.basename(filepath)
        channels = load_adamatzky_txt(filepath, max_rows=max_rows)

        for ch_idx, channel in enumerate(channels):
            spikes = extract_spikes(channel, processor=processor)
            total_spikes += len(spikes)

            if len(spikes) < 2:
                continue

            words = group_spikes_into_words(spikes, theta_multiplier)
            total_words += len(words)

            for word in words:
                features = extract_word_features(word)
                all_word_features.append(features)

        print(f"    {basename}: {len(channels)} channels, "
              f"running total: {total_spikes} spikes, {total_words} words")

    if not all_word_features:
        print("  No words found — check spike detection parameters")
        return None, None, None, None

    all_word_features = np.array(all_word_features)
    print(f"\n  Total: {total_spikes} spikes → {total_words} words "
          f"→ {all_word_features.shape[0]} feature vectors")

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_word_features)

    # Cap k at number of words
    actual_k = min(n_clusters, all_word_features.shape[0])
    if actual_k < n_clusters:
        print(f"  Capping k from {n_clusters} to {actual_k} (only {all_word_features.shape[0]} words)")

    # K-means clustering
    print(f"  Running k-means with k={actual_k}...")
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10, max_iter=300)
    word_labels = kmeans.fit_predict(features_scaled)

    print(f"  Vocabulary discovered: {actual_k} word types")

    return scaler, kmeans, all_word_features, word_labels


def run_elbow(adamatzky_dir=ADAMATZKY_DIR, max_k=80, step=5,
              max_rows=36000, theta_multiplier=1.0):
    """
    Run elbow method to find optimal k for vocabulary size.

    Plots inertia vs k and saves to vocabulary_analysis.png.
    """
    import glob as globmod

    processor = SignalProcessor(sample_rate=SAMPLE_RATE)

    # Collect word features (same as discover_vocabulary)
    txt_files = []
    for pattern in [os.path.join(adamatzky_dir, '**', '*.txt')]:
        txt_files.extend(globmod.glob(pattern, recursive=True))
    txt_files = [f for f in txt_files if '__MACOSX' not in f]

    all_word_features = []
    for filepath in sorted(txt_files):
        channels = load_adamatzky_txt(filepath, max_rows=max_rows)
        for channel in channels:
            spikes = extract_spikes(channel, processor=processor)
            if len(spikes) < 2:
                continue
            words = group_spikes_into_words(spikes, theta_multiplier)
            for word in words:
                all_word_features.append(extract_word_features(word))

    if not all_word_features:
        print("  No words found for elbow analysis")
        return

    all_word_features = np.array(all_word_features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_word_features)

    # Test k values
    k_values = list(range(5, min(max_k + 1, all_word_features.shape[0]), step))
    inertias = []

    print(f"  Testing k values: {k_values[0]}..{k_values[-1]} (step={step})")
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=200)
        km.fit(features_scaled)
        inertias.append(km.inertia_)
        print(f"    k={k}: inertia={km.inertia_:.1f}")

    # Plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertias, 'bo-', linewidth=2)
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Inertia (within-cluster sum of squares)')
    ax.set_title('Elbow Method for Fungal Vocabulary Size')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'vocabulary_elbow.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Elbow plot saved: {plot_path}")


# ============================================================
# WINDOW LABELING
# ============================================================

def label_windows_by_vocabulary(signal, words, word_labels,
                                window_samples=WINDOW_SAMPLES,
                                silence_label=None):
    """
    Label fixed-length windows by their dominant word type.

    For each 60s window, find which words overlap it.
    Window label = most common word type. Windows with no words
    get the silence label (K = max_label + 1).

    Args:
        signal: 1D signal array
        words: List of word lists (from group_spikes_into_words)
        word_labels: Per-word cluster labels (from k-means)
        window_samples: Samples per window
        silence_label: Label for silent windows. If None, uses max(word_labels)+1.

    Returns:
        windows: 2D array (n_windows, window_samples)
        labels: 1D array of word-type labels per window
    """
    windows = segment_windows(signal, window_samples, overlap=0.5)
    if windows.shape[0] == 0:
        return windows, np.array([], dtype=int)

    if silence_label is None:
        silence_label = int(np.max(word_labels)) + 1 if len(word_labels) > 0 else 0

    step = int(window_samples * 0.5)
    n_windows = windows.shape[0]
    labels = np.full(n_windows, silence_label, dtype=int)

    # Pre-compute word time ranges and labels
    word_times = []
    for i, word in enumerate(words):
        t_start = word[0]['time']
        t_end = word[-1]['time']
        word_times.append((t_start, t_end, word_labels[i]))

    # For each window, find overlapping words
    for w_idx in range(n_windows):
        w_start = (w_idx * step) / SAMPLE_RATE
        w_end = w_start + window_samples / SAMPLE_RATE

        overlapping_types = []
        for t_start, t_end, wtype in word_times:
            # Word overlaps window if word_end >= window_start and word_start <= window_end
            if t_end >= w_start and t_start <= w_end:
                overlapping_types.append(wtype)

        if overlapping_types:
            # Most common word type in this window
            types, counts = np.unique(overlapping_types, return_counts=True)
            labels[w_idx] = types[np.argmax(counts)]

    return windows, labels


# ============================================================
# BUILD LABELED DATASET (for TCN training)
# ============================================================

def build_labeled_dataset(adamatzky_dir=ADAMATZKY_DIR, kmeans=None,
                          scaler=None, max_rows=36000, theta_multiplier=1.0):
    """
    Build vocabulary-labeled windows for TCN Phase 2 training.

    Loads Adamatzky data, runs full pipeline (spikes → words → cluster labels
    → window labels), returns training-ready arrays.

    Args:
        adamatzky_dir: Path to Adamatzky .txt files
        kmeans: Fitted KMeans model
        scaler: Fitted StandardScaler
        max_rows: Max rows per file
        theta_multiplier: Temporal threshold multiplier

    Returns:
        X: 2D array (n_windows, 600)
        y: 1D array of word-type labels
    """
    import glob as globmod

    processor = SignalProcessor(sample_rate=SAMPLE_RATE)
    silence_label = kmeans.n_clusters  # K = silence class

    txt_files = []
    for pattern in [os.path.join(adamatzky_dir, '**', '*.txt')]:
        txt_files.extend(globmod.glob(pattern, recursive=True))
    txt_files = [f for f in txt_files if '__MACOSX' not in f]

    all_windows = []
    all_labels = []

    for filepath in sorted(txt_files):
        channels = load_adamatzky_txt(filepath, max_rows=max_rows)

        for channel in channels:
            # Extract spikes and group into words
            spikes = extract_spikes(channel, processor=processor)
            if len(spikes) < 2:
                # No meaningful words — label all windows as silence
                windows = segment_windows(channel)
                if windows.shape[0] > 0:
                    all_windows.append(windows)
                    all_labels.extend([silence_label] * windows.shape[0])
                continue

            words = group_spikes_into_words(spikes, theta_multiplier)

            # Get per-word features and cluster labels
            word_features = np.array([extract_word_features(w) for w in words])
            features_scaled = scaler.transform(word_features)
            word_labels = kmeans.predict(features_scaled)

            # Label windows
            windows, labels = label_windows_by_vocabulary(
                channel, words, word_labels, silence_label=silence_label
            )

            if windows.shape[0] > 0:
                all_windows.append(windows)
                all_labels.extend(labels)

        print(f"    {os.path.basename(filepath)}: "
              f"{sum(w.shape[0] for w in all_windows)} windows so far")

    if not all_windows:
        return np.array([]).reshape(0, WINDOW_SAMPLES), np.array([], dtype=int)

    X = np.vstack(all_windows)
    y = np.array(all_labels, dtype=int)

    print(f"\n  Labeled dataset: {X.shape[0]} windows, "
          f"{len(np.unique(y))} classes (including silence)")
    for c in sorted(np.unique(y)):
        name = f"word_{c}" if c < silence_label else "silence"
        print(f"    {name}: {np.sum(y == c)} windows")

    return X, y


# ============================================================
# VOCABULARY ANALYSIS
# ============================================================

def analyze_vocabulary(word_labels, word_features, save_dir=PLOTS_DIR):
    """
    Analyze discovered vocabulary — word frequency, Zipf's law,
    Shannon entropy, per-type stats.

    Args:
        word_labels: 1D array of cluster labels per word
        word_features: 2D array (n_words, 7) of per-word features
        save_dir: Directory to save plots and stats
    """
    os.makedirs(save_dir, exist_ok=True)

    types, counts = np.unique(word_labels, return_counts=True)
    n_types = len(types)
    n_words = len(word_labels)

    # Sort by frequency (most common first)
    sort_idx = np.argsort(-counts)
    types_sorted = types[sort_idx]
    counts_sorted = counts[sort_idx]

    print(f"\n  Vocabulary Analysis")
    print(f"  {'='*40}")
    print(f"  Total words: {n_words}")
    print(f"  Vocabulary size: {n_types} word types")

    # Core lexicon: words appearing >1% of the time
    freq = counts_sorted / n_words
    core_mask = freq >= 0.01
    core_count = np.sum(core_mask)
    print(f"  Core lexicon (>1%): {core_count} word types")

    # Average word length (in spikes)
    avg_length = np.mean(word_features[:, 0])
    print(f"  Average word length: {avg_length:.2f} spikes "
          f"(Adamatzky found 5.97, English = 4.8 letters)")

    # Shannon entropy
    probs = counts / n_words
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = np.log2(n_types)
    print(f"  Shannon entropy: {entropy:.2f} bits (max possible: {max_entropy:.2f})")

    # Per-type feature averages
    type_stats = {}
    for t in types:
        mask = word_labels == t
        type_features = word_features[mask]
        stats = {}
        for i, name in enumerate(WORD_FEATURE_NAMES):
            stats[name] = float(np.mean(type_features[:, i]))
        stats['count'] = int(np.sum(mask))
        stats['frequency'] = float(np.sum(mask) / n_words)
        type_stats[str(int(t))] = stats

    # Save stats JSON
    stats_output = {
        'vocabulary_size': n_types,
        'total_words': n_words,
        'core_lexicon_size': int(core_count),
        'avg_word_length_spikes': float(avg_length),
        'shannon_entropy': float(entropy),
        'max_entropy': float(max_entropy),
        'word_types': type_stats,
    }
    stats_path = os.path.join(save_dir, 'vocabulary_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Fungal Signal Vocabulary Analysis (Adamatzky Method)', fontsize=14)

    # 1. Word frequency distribution
    ax = axes[0, 0]
    ax.bar(range(min(30, n_types)), counts_sorted[:30], color='forestgreen', alpha=0.7)
    ax.set_xlabel('Word type (ranked by frequency)')
    ax.set_ylabel('Count')
    ax.set_title('Word Frequency Distribution (top 30)')
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Zipf's law check (log-log rank vs frequency)
    ax = axes[0, 1]
    ranks = np.arange(1, n_types + 1)
    ax.loglog(ranks, counts_sorted, 'go-', markersize=3, linewidth=1)
    # Reference Zipf line (1/rank)
    zipf_ref = counts_sorted[0] / ranks
    ax.loglog(ranks, zipf_ref, 'r--', alpha=0.5, label='Zipf (1/rank)')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.set_title("Zipf's Law Check")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Word length distribution
    ax = axes[1, 0]
    word_lengths = word_features[:, 0].astype(int)
    max_len = min(int(np.max(word_lengths)), 30)
    ax.hist(word_lengths, bins=range(1, max_len + 2), color='teal',
            alpha=0.7, edgecolor='black', align='left')
    ax.set_xlabel('Word length (number of spikes)')
    ax.set_ylabel('Count')
    ax.set_title(f'Word Length Distribution (mean={avg_length:.1f})')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Feature scatter (mean amplitude vs duration)
    ax = axes[1, 1]
    scatter = ax.scatter(
        word_features[:, 1],  # total_duration
        word_features[:, 2],  # mean_amplitude
        c=word_labels, cmap='tab20', s=10, alpha=0.5
    )
    ax.set_xlabel('Word duration (seconds)')
    ax.set_ylabel('Mean amplitude')
    ax.set_title('Words in Feature Space (colored by cluster)')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(save_dir, 'vocabulary_analysis.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


# ============================================================
# MAIN CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Discover fungal signal vocabulary from Adamatzky data'
    )
    parser.add_argument('--n-clusters', type=int, default=50,
                        help='Number of word clusters (default: 50)')
    parser.add_argument('--theta', type=float, default=1.0,
                        help='ISI threshold multiplier (default: 1.0, Adamatzky also used 2.0)')
    parser.add_argument('--max-rows', type=int, default=36000,
                        help='Max rows per Adamatzky file (default: 36000)')
    parser.add_argument('--adamatzky-dir', type=str, default=ADAMATZKY_DIR)
    parser.add_argument('--analyze', action='store_true',
                        help='Load saved model and show vocabulary analysis')
    parser.add_argument('--elbow', action='store_true',
                        help='Run elbow method for k selection')

    args = parser.parse_args()

    print("=" * 60)
    print("FUNGAL SIGNAL VOCABULARY DISCOVERY")
    print("EE297B Research Project — SJSU")
    print("=" * 60)

    if args.elbow:
        print("\nRunning elbow method for k selection...")
        run_elbow(args.adamatzky_dir, max_rows=args.max_rows,
                  theta_multiplier=args.theta)
        return

    if args.analyze:
        # Load saved results and show analysis
        if not os.path.exists(VOCAB_STATS_PATH):
            print(f"  No saved stats found at {VOCAB_STATS_PATH}")
            print("  Run without --analyze first to discover vocabulary")
            return

        with open(VOCAB_STATS_PATH, 'r') as f:
            stats = json.load(f)

        print(f"\n  Saved Vocabulary Stats:")
        print(f"  {'='*40}")
        print(f"  Vocabulary size: {stats['vocabulary_size']} word types")
        print(f"  Total words: {stats['total_words']}")
        print(f"  Core lexicon: {stats['core_lexicon_size']} types")
        print(f"  Avg word length: {stats['avg_word_length_spikes']:.2f} spikes")
        print(f"  Shannon entropy: {stats['shannon_entropy']:.2f} / {stats['max_entropy']:.2f} bits")

        # Top 10 most frequent types
        word_types = stats['word_types']
        sorted_types = sorted(word_types.items(),
                              key=lambda x: x[1]['count'], reverse=True)
        print(f"\n  Top 10 word types:")
        for wtype, wstats in sorted_types[:10]:
            print(f"    word_{wtype}: count={wstats['count']}, "
                  f"freq={wstats['frequency']:.3f}, "
                  f"avg_spikes={wstats['n_spikes']:.1f}, "
                  f"avg_amp={wstats['mean_amplitude']:.3f}")
        return

    # --- Full discovery pipeline ---
    import joblib

    print(f"\n  Parameters:")
    print(f"    k = {args.n_clusters}")
    print(f"    theta multiplier = {args.theta}")
    print(f"    max rows/file = {args.max_rows}")
    print(f"    data dir = {args.adamatzky_dir}")

    # Step 1: Discover vocabulary
    print("\n  Step 1: Discovering vocabulary...")
    scaler, kmeans, word_features, word_labels = discover_vocabulary(
        args.adamatzky_dir,
        n_clusters=args.n_clusters,
        max_rows=args.max_rows,
        theta_multiplier=args.theta,
    )

    if kmeans is None:
        print("  Failed — no vocabulary discovered")
        return

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(kmeans, VOCAB_KMEANS_PATH)
    joblib.dump(scaler, VOCAB_SCALER_PATH)
    print(f"\n  K-means saved: {VOCAB_KMEANS_PATH}")
    print(f"  Scaler saved: {VOCAB_SCALER_PATH}")

    # Step 2: Analyze vocabulary
    print("\n  Step 2: Analyzing vocabulary...")
    analyze_vocabulary(word_labels, word_features)

    # Step 3: Build labeled dataset for TCN training
    print("\n  Step 3: Building labeled dataset for TCN Phase 2...")
    X, y = build_labeled_dataset(
        args.adamatzky_dir, kmeans, scaler,
        max_rows=args.max_rows, theta_multiplier=args.theta,
    )

    if X.shape[0] > 0:
        os.makedirs(MODELS_DIR, exist_ok=True)
        np.savez_compressed(VOCAB_LABELS_PATH, X=X, y=y)
        size_mb = os.path.getsize(VOCAB_LABELS_PATH) / (1024 * 1024)
        print(f"\n  Labeled dataset saved: {VOCAB_LABELS_PATH} ({size_mb:.1f} MB)")
    else:
        print("  Warning: No labeled windows produced")

    gc.collect()
    print("\n" + "=" * 60)
    print("VOCABULARY DISCOVERY COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
