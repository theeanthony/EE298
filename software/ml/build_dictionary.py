#!/usr/bin/env python3
"""
build_dictionary.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Map fungal signal "words" to stimulus meanings.

Like watching someone in context — "they touched fire and said THIS word,
so that word must mean pain." Runs the vocabulary-trained TCN on Buffi
stimulus-response data where we KNOW what chemical triggered each signal,
then builds a word→stimulus dictionary.

Usage:
    python build_dictionary.py
    python build_dictionary.py --tcn-model models/tcn_phase2_vocabulary.pt
"""

import numpy as np
import os
import sys
import json
import argparse

import torch

from tcn_model import build_tcn
from data_loader import (
    load_buffi_stimulus_labeled, BUFFI_STIMULI,
    WINDOW_SAMPLES, SAMPLE_RATE
)

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
BUFFI_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'buffi')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'ml_results')

DEFAULT_TCN_PATH = os.path.join(MODELS_DIR, 'tcn_phase2_vocabulary.pt')
VOCAB_STATS_PATH = os.path.join(PLOTS_DIR, 'vocabulary_stats.json')
VOCAB_LABELS_PATH = os.path.join(MODELS_DIR, 'vocabulary_labels.npz')
DICTIONARY_PATH = os.path.join(MODELS_DIR, 'fungal_dictionary.json')


def infer_n_classes(checkpoint_path):
    """Infer number of classes from a saved TCN checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    # Last linear layer: head.3.weight has shape (n_classes, hidden)
    return state_dict['head.3.weight'].shape[0]


def predict_word_types(model, X, device, batch_size=64):
    """
    Run TCN on windows and return predicted word-type labels.

    Args:
        model: Loaded TCN model in eval mode
        X: 2D array (n_windows, 600)
        device: torch device
        batch_size: Inference batch size

    Returns:
        1D array of predicted word-type IDs
    """
    model.eval()
    all_preds = []

    X_tensor = torch.FloatTensor(X).unsqueeze(1)  # (N, 1, 600)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            logits = model(batch)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return np.array(all_preds)


def build_dictionary(tcn_model_path=DEFAULT_TCN_PATH, buffi_dir=BUFFI_DIR):
    """
    Build the fungal word → stimulus meaning dictionary.

    Pipeline:
    1. Load vocabulary-trained TCN (Phase 2)
    2. Load Buffi data BY STIMULUS (4 separate label groups)
    3. Run TCN on each stimulus group → get predicted word types
    4. For each word type, compute which stimulus it appears under most
    5. Output dictionary mapping word types → stimulus meanings

    Returns:
        dictionary: Dict mapping word type ID → stimulus info
    """
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"  Device: {device}")

    # Load TCN
    if not os.path.exists(tcn_model_path):
        print(f"  ERROR: TCN checkpoint not found: {tcn_model_path}")
        print("  Run: python train_tcn.py --mode vocabulary --phase 2")
        return None

    n_classes = infer_n_classes(tcn_model_path)
    print(f"  Loading TCN ({n_classes} classes): {tcn_model_path}")
    model = build_tcn(n_classes=n_classes)
    model.load_state_dict(torch.load(tcn_model_path, map_location='cpu', weights_only=True))
    model = model.to(device)
    model.eval()

    # Load Buffi data with stimulus labels
    if not os.path.exists(buffi_dir):
        print(f"  ERROR: Buffi dir not found: {buffi_dir}")
        return None

    print("\n  Loading Buffi stimulus-labeled data...")
    X_buffi, y_stim = load_buffi_stimulus_labeled(buffi_dir)

    if X_buffi.shape[0] == 0:
        print("  No Buffi data loaded")
        return None

    # Load unique_labels to map remapped TCN outputs back to original k-means IDs.
    # Phase 2 trained with contiguous labels 0..K-1 (remapped), but
    # vocabulary_stats.json uses the original k-means cluster IDs as keys.
    if os.path.exists(VOCAB_LABELS_PATH):
        vocab_data = np.load(VOCAB_LABELS_PATH)
        unique_labels = vocab_data['unique_labels'] if 'unique_labels' in vocab_data \
            else np.sort(np.unique(vocab_data['y']))
        print(f"  Loaded unique_labels from vocabulary_labels.npz: {unique_labels}")
    else:
        unique_labels = None
        print("  WARNING: vocabulary_labels.npz not found — "
              "dictionary keys will use remapped IDs and may not match vocabulary_stats.json")

    # Run TCN on all Buffi windows
    print(f"\n  Running TCN inference on {X_buffi.shape[0]} windows...")
    word_preds = predict_word_types(model, X_buffi, device)

    # Build word-type → stimulus distribution
    stimulus_names = {v: k for k, v in BUFFI_STIMULI.items()}
    word_types = sorted(np.unique(word_preds))

    print(f"\n  Building dictionary ({len(word_types)} word types detected)...")

    dictionary = {}
    for wtype_remapped in word_types:
        # Convert remapped label back to original k-means cluster ID
        if unique_labels is not None and wtype_remapped < len(unique_labels):
            original_id = int(unique_labels[wtype_remapped])
        else:
            original_id = int(wtype_remapped)

        wtype_mask = word_preds == wtype_remapped
        stim_labels_for_wtype = y_stim[wtype_mask]

        # Count per stimulus
        stim_distribution = {}
        for stim_label, stim_name in sorted(stimulus_names.items()):
            count = int(np.sum(stim_labels_for_wtype == stim_label))
            if count > 0:
                stim_distribution[stim_name] = count

        total_count = int(np.sum(wtype_mask))

        # Primary stimulus = most frequent
        if stim_distribution:
            primary_stim = max(stim_distribution, key=stim_distribution.get)
            confidence = stim_distribution[primary_stim] / total_count
        else:
            primary_stim = "unknown"
            confidence = 0.0

        # Normalize distribution to fractions
        stim_fractions = {k: v / total_count for k, v in stim_distribution.items()}

        dictionary[str(original_id)] = {
            'total_occurrences': total_count,
            'primary_stimulus': primary_stim,
            'confidence': round(confidence, 3),
            'stimulus_distribution': {k: round(v, 3) for k, v in stim_fractions.items()},
        }

    # Load vocabulary stats to enrich with feature info.
    # dictionary keys are now original k-means IDs, matching vocabulary_stats.json.
    if os.path.exists(VOCAB_STATS_PATH):
        with open(VOCAB_STATS_PATH, 'r') as f:
            vocab_stats = json.load(f)
        word_type_stats = vocab_stats.get('word_types', {})
        for wtype_str, entry in dictionary.items():
            if wtype_str in word_type_stats:
                wstats = word_type_stats[wtype_str]
                entry['avg_amplitude'] = round(wstats.get('mean_amplitude', 0), 4)
                entry['avg_duration'] = round(wstats.get('total_duration', 0), 4)
                entry['avg_n_spikes'] = round(wstats.get('n_spikes', 0), 1)

    # Summary
    stimulus_interpretations = {
        'calcimycin': 'Calcium ionophore stress response',
        'cycloheximide': 'Protein synthesis inhibition response',
        'sodiumazide': 'Respiratory chain disruption response',
        'voriconazole': 'Antifungal (ergosterol synthesis inhibition) response',
    }

    for wtype_str, entry in dictionary.items():
        primary = entry['primary_stimulus']
        entry['interpretation'] = stimulus_interpretations.get(primary, 'Unknown stimulus')

    return dictionary


def main():
    parser = argparse.ArgumentParser(
        description='Build fungal word→stimulus dictionary'
    )
    parser.add_argument('--tcn-model', type=str, default=DEFAULT_TCN_PATH,
                        help='Path to vocabulary-trained TCN checkpoint')
    parser.add_argument('--buffi-dir', type=str, default=BUFFI_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("FUNGAL SIGNAL DICTIONARY BUILDER")
    print("EE297B Research Project — SJSU")
    print("=" * 60)

    dictionary = build_dictionary(args.tcn_model, args.buffi_dir)

    if dictionary is None:
        print("\n  Dictionary build failed")
        return

    # Save dictionary
    os.makedirs(MODELS_DIR, exist_ok=True)
    output = {
        'vocabulary_size': len(dictionary),
        'stimuli': list(BUFFI_STIMULI.keys()),
        'words': dictionary,
    }

    with open(DICTIONARY_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Dictionary saved: {DICTIONARY_PATH}")

    # Print summary table
    print(f"\n  {'='*70}")
    print(f"  FUNGAL DICTIONARY — {len(dictionary)} word types")
    print(f"  {'='*70}")
    print(f"  {'Word':<8} {'Count':<8} {'Primary Stimulus':<20} {'Conf':<8} {'Interp'}")
    print(f"  {'-'*70}")

    # Sort by confidence
    sorted_words = sorted(dictionary.items(),
                          key=lambda x: x[1]['confidence'], reverse=True)
    for wtype, entry in sorted_words[:20]:
        print(f"  word_{wtype:<4} {entry['total_occurrences']:<8} "
              f"{entry['primary_stimulus']:<20} {entry['confidence']:<8.3f} "
              f"{entry['interpretation']}")

    if len(sorted_words) > 20:
        print(f"  ... and {len(sorted_words) - 20} more word types")

    print(f"\n  {'='*70}")
    print("DICTIONARY BUILD COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
