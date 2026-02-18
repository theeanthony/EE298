#!/usr/bin/env python3
"""
train_tcn.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

3-phase transfer learning for TCN classifier:
  Phase 1: Pre-train on ECG heartbeat data (5-class, learn temporal patterns)
  Phase 2: Domain-adapt on plant + Adamatzky fungal signals (2-class, shift domain)
  Phase 3: Fine-tune on Buffi + synthetic fungal data (2-class, target task)

Vocabulary mode (--mode vocabulary):
  Phase 1: Same ECG pre-training
  Phase 2: Train on vocabulary-labeled Adamatzky data (k-class spike word types)
  Phase 3: Fine-tune on stimulus-labeled Buffi data (5-class: 4 stimuli + baseline)

Usage:
    python train_tcn.py --phase 1                      # Pre-train on ECG
    python train_tcn.py --phase 2                      # Domain-adapt (binary)
    python train_tcn.py --phase 3                      # Fine-tune (binary)
    python train_tcn.py --phase all                    # Run all 3 sequentially
    python train_tcn.py --phase 3 --full               # Colab mode
    python train_tcn.py --evaluate                     # Evaluate binary model
    python train_tcn.py --mode vocabulary --phase 2    # Vocabulary Phase 2
    python train_tcn.py --mode vocabulary --phase 3    # Stimulus Phase 3
    python train_tcn.py --mode vocabulary --phase all  # Full vocabulary pipeline
"""

import numpy as np
import os
import sys
import gc
import argparse
import time
import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    roc_curve, auc, ConfusionMatrixDisplay
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local imports
from tcn_model import build_tcn, build_tcn_with_attention, count_parameters
from data_loader import (
    load_all_data, load_adamatzky_data, load_synthetic_data,
    load_buffi_stimulus_labeled, load_vocabulary_labeled,
    segment_windows, WINDOW_SAMPLES, SAMPLE_RATE
)
from augmentation import augment_dataset, normalize_signal, mixup_batch

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, 'data', 'synthetic')
ADAMATZKY_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'adamatzky')
BUFFI_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'buffi')
ECG_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'ecg_heartbeat')
PLANT_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'plant_electrophys')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'ml_results')

CHECKPOINT_PHASE1 = os.path.join(MODELS_DIR, 'tcn_phase1_ecg.pt')
CHECKPOINT_PHASE2 = os.path.join(MODELS_DIR, 'tcn_phase2_adapted.pt')
CHECKPOINT_FINAL = os.path.join(MODELS_DIR, 'tcn_final.pt')

# Vocabulary mode checkpoints
CHECKPOINT_PHASE2_VOCAB = os.path.join(MODELS_DIR, 'tcn_phase2_vocabulary.pt')
CHECKPOINT_PHASE3_STIMULUS = os.path.join(MODELS_DIR, 'tcn_phase3_stimulus.pt')
VOCAB_LABELS_PATH = os.path.join(MODELS_DIR, 'vocabulary_labels.npz')

# Phase 0: plant pre-training checkpoint
CHECKPOINT_PHASE0_PLANT = os.path.join(MODELS_DIR, 'tcn_phase0_plant.pt')

# Track start time
_t0 = time.time()


# ============================================================
# FOCAL LOSS (Tier 1: replaces weighted CrossEntropyLoss)
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification (Lin et al. 2017).

    Down-weights *easy* examples (those the model already predicts with high
    confidence) and concentrates gradient on hard misclassified windows.
    Critical for the silence class (word_50 = 93% of training data) where
    correct "silence" predictions would otherwise drown the gradient.

    Loss = (1 - p_t)^gamma * CE(logits, targets)

    With gamma=2.0, examples predicted with p=0.9 are down-weighted 100x
    vs focal loss with gamma=0. CE (gamma=0) gives no such down-weighting.

    Args:
        gamma:  Focusing parameter. 0 = standard CE, 2 = standard focal loss.
        weight: Per-class weight tensor (same as CrossEntropyLoss weight).
    """

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def log_memory(stage: str):
    """Print RSS memory usage and elapsed time."""
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    elapsed = time.time() - _t0
    print(f"  [{elapsed:6.1f}s] {stage}: {rss_mb:.0f} MB RSS")


def get_device():
    """Auto-detect best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  Device: Apple MPS")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU")
    return device


# ============================================================
# SHARED TRAINING LOOP
# ============================================================

def train_phase(model, train_loader, val_loader, epochs, lr, device,
                patience=3, phase_name="", class_weights=None,
                use_focal=True, focal_gamma=2.0,
                use_mixup=True, mixup_alpha=0.2,
                warmup_epochs=2, max_grad_norm=1.0):
    """
    Shared training loop for all phases — with Tier 1 enhancements.

    Enhancements over baseline:
      - AdamW (weight_decay=1e-4): decouples L2 from adaptive LR,
        reduces overfitting on sparse word-type classes.
      - FocalLoss (gamma=2): down-weights easy examples so minority
        word types get more gradient. Disable with use_focal=False.
      - Gradient clipping (max_norm=1.0): prevents exploding gradients
        during Phase 2 with many imbalanced classes.
      - LR warmup (first warmup_epochs): linearly ramps LR from
        lr/warmup_steps to lr per step — avoids corrupting pre-trained
        weights with large early updates.
      - Mixup (alpha=0.2): convex batch combinations smooth decision
        boundaries between similar stimuli. Disable with use_mixup=False.

    Args:
        class_weights: Optional FloatTensor of per-class weights.
        use_focal:     If True, use FocalLoss; else CrossEntropyLoss.
        focal_gamma:   Focusing exponent for FocalLoss (default 2.0).
        use_mixup:     If True, apply Mixup to each training batch.
        mixup_alpha:   Beta distribution parameter for Mixup (0.2 = mild).
        warmup_epochs: Number of epochs for linear LR warmup.
        max_grad_norm: Gradient clipping threshold (0 = disabled).

    Returns:
        best_val_loss: Best validation loss achieved
        history: Dict of training metrics per epoch
    """
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    if use_focal:
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Pre-compute warmup schedule (step-based, linear ramp)
    total_warmup_steps = warmup_epochs * max(len(train_loader), 1)
    step_count = 0

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # LR warmup: linearly ramp from 0 → lr over total_warmup_steps
            if total_warmup_steps > 0 and step_count < total_warmup_steps:
                warmup_factor = (step_count + 1) / total_warmup_steps
                for pg in optimizer.param_groups:
                    pg['lr'] = lr * warmup_factor

            optimizer.zero_grad()

            # Mixup augmentation (on-device, no extra memory allocation)
            if use_mixup and model.training:
                X_mix, y_a, y_b, lam = mixup_batch(X_batch, y_batch,
                                                     alpha=mixup_alpha)
                logits = model(X_mix)
                loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
            else:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

            loss.backward()

            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            step_count += 1

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # --- Validate ---
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)

        # ReduceLROnPlateau kicks in after warmup completes
        if step_count >= total_warmup_steps:
            scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        lr_current = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}/{epochs} — "
              f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
              f"val_acc: {val_acc:.3f}, lr: {lr_current:.1e}")

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

        log_memory(f"{phase_name} epoch {epoch+1}")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss, history


def evaluate_model(model, loader, criterion, device):
    """Evaluate model on a DataLoader, returning loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * y_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def save_checkpoint(model, path):
    """Save full model state dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Checkpoint saved: {path} ({size_kb:.0f} KB)")


def make_loader(X, y, batch_size, shuffle=True, num_workers=0):
    """Create a DataLoader from numpy arrays. Reshapes X to (N, 1, 600)."""
    X_tensor = torch.FloatTensor(X).unsqueeze(1)  # (N, 600) -> (N, 1, 600)
    y_tensor = torch.LongTensor(y)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)


# ============================================================
# PHASE 0: PRE-TRAIN ON PLANT ELECTROPHYSIOLOGY (Tier 2)
# ============================================================

def run_phase0_plant(device, batch_size=32, epochs=10, lr=1e-3, num_workers=0,
                     plant_dir=None, use_focal=True, use_mixup=True,
                     warmup_epochs=2, num_blocks=5, use_attention=False):
    """
    Phase 0: Pre-train on plant electrophysiology data (binary: AP spike vs silence).

    Scientific rationale: plant action potentials share key properties with fungal
    signals — both are slow (0.01-1 Hz), bioelectrical, and aperiodic — making plant
    data much closer to the target domain than ECG heartbeats.

    Paper angle: "Cross-kingdom transfer learning — pre-training on vascular plant
    action potentials improves fungal signal classification."

    Data: .wav files at ~10 kHz in data/external/plant_electrophys/
    If missing, place Mancuso-lab style AP recordings there.
    Phase 1 (ECG) will automatically load this encoder as its starting point.
    """
    if plant_dir is None:
        plant_dir = PLANT_DIR

    print("\n" + "=" * 60)
    print("PHASE 0: PRE-TRAIN ON PLANT ELECTROPHYSIOLOGY")
    print("=" * 60)

    print("\n  Loading plant electrophysiology data...")
    X, y = load_plant_data(plant_dir)

    if X is None or X.shape[0] == 0:
        print("  No plant data found — skipping Phase 0")
        print(f"  (Place plant .wav recordings in: {plant_dir})")
        return None

    print(f"  Plant: {X.shape[0]} windows "
          f"(label 0: {np.sum(y == 0)}, label 1: {np.sum(y == 1)})")
    log_memory("After Phase 0 plant data load")

    # Augment and balance
    X, y = augment_dataset(X, y, n_augmented_per_sample=2, balance_classes=True)
    print(f"  After augmentation: {X.shape[0]} windows "
          f"(label 0: {np.sum(y == 0)}, label 1: {np.sum(y == 1)})")
    log_memory("After Phase 0 augment")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_loader = make_loader(X_train, y_train, batch_size, num_workers=num_workers)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False,
                             num_workers=num_workers)

    del X, y, X_train, X_val, y_train, y_val
    gc.collect()

    # Build fresh 2-class model
    if use_attention:
        model = build_tcn_with_attention(n_classes=2, num_blocks=num_blocks)
    else:
        model = build_tcn(n_classes=2, num_blocks=num_blocks)

    total, trainable = count_parameters(model)
    print(f"  Model: {total:,} total params, {trainable:,} trainable")
    model = model.to(device)

    best_loss, history = train_phase(
        model, train_loader, val_loader, epochs, lr, device,
        patience=3, phase_name="Phase0-Plant",
        use_focal=use_focal, use_mixup=use_mixup,
        warmup_epochs=warmup_epochs,
    )
    print(f"  Best val loss: {best_loss:.4f}")

    save_checkpoint(model, CHECKPOINT_PHASE0_PLANT)
    log_memory("Phase 0 complete")

    return model


# ============================================================
# PHASE 1: PRE-TRAIN ON ECG
# ============================================================

def load_ecg_data(ecg_dir):
    """
    Load ECG heartbeat data and convert to 600-sample windows at 10 Hz.

    Each ECG row is 187 samples at 125 Hz (~1.5 sec per heartbeat).
    Strategy: concatenate consecutive same-class heartbeats, resample to 10 Hz,
    segment into 600-sample windows.
    """
    import csv

    train_path = os.path.join(ecg_dir, 'mitbih_train.csv')
    test_path = os.path.join(ecg_dir, 'mitbih_test.csv')

    if not os.path.exists(train_path):
        print(f"  ERROR: {train_path} not found")
        print("  Run: python download_pretrain_data.py --ecg")
        return None, None

    print("  Loading ECG CSVs...")
    all_windows = []
    all_labels = []

    for csv_path in [train_path, test_path]:
        if not os.path.exists(csv_path):
            continue
        print(f"    {os.path.basename(csv_path)}...")

        # Read all rows: 187 signal values + 1 label
        rows_by_class = {}
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 188:
                    continue
                label = int(float(row[-1]))
                signal = np.array([float(x) for x in row[:-1]])
                if label not in rows_by_class:
                    rows_by_class[label] = []
                rows_by_class[label].append(signal)

        # Concatenate same-class heartbeats, resample, and segment
        for label, signals in rows_by_class.items():
            # Concatenate all heartbeats of this class
            concat = np.concatenate(signals)  # at 125 Hz

            # Resample from 125 Hz to 10 Hz
            n_original = len(concat)
            duration = n_original / 125.0
            n_target = int(duration * SAMPLE_RATE)
            if n_target < WINDOW_SAMPLES:
                continue
            t_orig = np.linspace(0, duration, n_original)
            t_target = np.linspace(0, duration, n_target)
            resampled = np.interp(t_target, t_orig, concat)

            # Segment into windows
            windows = segment_windows(resampled, WINDOW_SAMPLES, overlap=0.5)
            if windows.shape[0] > 0:
                # Z-score normalize each window
                for i in range(windows.shape[0]):
                    windows[i] = normalize_signal(windows[i])
                all_windows.append(windows)
                all_labels.extend([label] * windows.shape[0])

    if not all_windows:
        return None, None

    X = np.vstack(all_windows)
    y = np.array(all_labels)

    print(f"  ECG data: {X.shape[0]} windows, {len(np.unique(y))} classes")
    for c in sorted(np.unique(y)):
        print(f"    Class {c}: {np.sum(y == c)} windows")

    return X, y


def run_phase1(device, batch_size=32, epochs=10, lr=1e-3, num_workers=0,
               ecg_dir=None, use_focal=True, use_mixup=True,
               warmup_epochs=2, num_blocks=5, use_attention=False):
    """Phase 1: Pre-train on ECG heartbeat data (5-class).
    Loads Phase 0 (plant) encoder as starting point if checkpoint exists.
    """
    if ecg_dir is None:
        ecg_dir = ECG_DIR
    print("\n" + "=" * 60)
    print("PHASE 1: PRE-TRAIN ON ECG HEARTBEAT DATA")
    print("=" * 60)

    X, y = load_ecg_data(ecg_dir)
    if X is None:
        print("  Skipping Phase 1 — no ECG data found")
        return None
    log_memory("After ECG data load")

    # Cap at 50K windows to keep training fast
    if X.shape[0] > 50000:
        indices = np.random.RandomState(42).choice(X.shape[0], 50000, replace=False)
        X, y = X[indices], y[indices]
        print(f"  Capped to {X.shape[0]} windows for training speed")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    train_loader = make_loader(X_train, y_train, batch_size, num_workers=num_workers)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False,
                             num_workers=num_workers)

    del X, y, X_train, X_val, y_train, y_val
    gc.collect()

    # Build fresh 5-class model
    if use_attention:
        model = build_tcn_with_attention(n_classes=5, num_blocks=num_blocks)
    else:
        model = build_tcn(n_classes=5, num_blocks=num_blocks)

    # Load Phase 0 (plant) encoder as starting point if available
    # Cross-kingdom transfer: plant APs → ECG features → fungal domain
    if os.path.exists(CHECKPOINT_PHASE0_PLANT):
        print(f"\n  Loading Phase 0 (plant) encoder: {CHECKPOINT_PHASE0_PLANT}")
        phase0_model = (build_tcn_with_attention(n_classes=2, num_blocks=num_blocks)
                        if use_attention
                        else build_tcn(n_classes=2, num_blocks=num_blocks))
        phase0_state = torch.load(CHECKPOINT_PHASE0_PLANT, map_location='cpu',
                                  weights_only=True)
        phase0_model.load_state_dict(phase0_state, strict=False)
        model.load_encoder_state_dict(phase0_model.get_encoder_state_dict())
        del phase0_model, phase0_state
        gc.collect()
        print("  Plant pre-training encoder transferred to Phase 1 (cross-kingdom transfer)")
    else:
        print("\n  No Phase 0 plant checkpoint — Phase 1 starts from scratch")

    total, trainable = count_parameters(model)
    print(f"  Model: {total:,} total params, {trainable:,} trainable")
    model = model.to(device)

    best_loss, history = train_phase(
        model, train_loader, val_loader, epochs, lr, device,
        patience=3, phase_name="Phase1",
        use_focal=use_focal, use_mixup=use_mixup,
        warmup_epochs=warmup_epochs,
    )
    print(f"  Best val loss: {best_loss:.4f}")

    save_checkpoint(model, CHECKPOINT_PHASE1)
    log_memory("Phase 1 complete")

    return model


# ============================================================
# PHASE 2: DOMAIN ADAPTATION
# ============================================================

def load_plant_data(plant_dir):
    """
    Load plant electrophysiology data and convert to 600-sample windows.

    .wav files at 10 kHz — decimate to 10 Hz, bandpass, segment.
    .txt files contain event markers for stimulus labels.
    """
    from scipy.io import wavfile
    from scipy.signal import decimate

    if not os.path.exists(plant_dir):
        print(f"  Plant data dir not found: {plant_dir}")
        return None, None

    wav_files = sorted([f for f in os.listdir(plant_dir) if f.endswith('.wav')])
    if not wav_files:
        print(f"  No .wav files found in {plant_dir}")
        return None, None

    all_windows = []
    all_labels = []

    for wav_file in wav_files:
        wav_path = os.path.join(plant_dir, wav_file)
        txt_file = wav_file.replace('.wav', '.txt')
        txt_path = os.path.join(plant_dir, txt_file)

        print(f"    Loading {wav_file}...")
        try:
            sr, data = wavfile.read(wav_path)
        except Exception as e:
            print(f"      Error reading {wav_file}: {e}")
            continue

        # Handle stereo by taking first channel
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(float)

        # Decimate from source rate to 10 Hz in stages
        # Each stage decimates by up to 10x
        current_rate = sr
        while current_rate > SAMPLE_RATE * 10:
            factor = min(10, int(current_rate / (SAMPLE_RATE * 10)))
            if factor < 2:
                break
            data = decimate(data, factor)
            current_rate /= factor

        # Final decimation to reach 10 Hz
        final_factor = int(round(current_rate / SAMPLE_RATE))
        if final_factor >= 2:
            data = decimate(data, final_factor)

        # Normalize
        data = normalize_signal(data)

        # Segment into windows
        windows = segment_windows(data, WINDOW_SAMPLES, overlap=0.5)
        if windows.shape[0] == 0:
            continue

        # Load event markers if available
        event_times = []
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = line.split()
                        if parts:
                            try:
                                event_times.append(float(parts[0]))
                            except ValueError:
                                continue
            except Exception:
                pass

        # Label windows: if event markers exist, windows near events = 1, else = 0
        if event_times:
            labels = np.zeros(windows.shape[0], dtype=int)
            step = int(WINDOW_SAMPLES * 0.5) / SAMPLE_RATE  # step in seconds
            for i in range(windows.shape[0]):
                window_start = i * step
                window_end = window_start + WINDOW_SAMPLES / SAMPLE_RATE
                for evt in event_times:
                    if window_start <= evt <= window_end:
                        labels[i] = 1
                        break
        else:
            # No markers — label all as active (plant signal present)
            labels = np.ones(windows.shape[0], dtype=int)

        all_windows.append(windows)
        all_labels.extend(labels)

    if not all_windows:
        return None, None

    X = np.vstack(all_windows)
    y = np.array(all_labels)
    return X, y


def run_phase2(device, batch_size=32, epochs=10, lr=1e-4, num_workers=0,
               plant_dir=None, use_focal=True, use_mixup=True,
               warmup_epochs=2, num_blocks=5, use_attention=False):
    """Phase 2: Domain-adapt on plant + Adamatzky data (2-class)."""
    if plant_dir is None:
        plant_dir = PLANT_DIR
    print("\n" + "=" * 60)
    print("PHASE 2: DOMAIN ADAPTATION (Plant + Fungal signals)")
    print("=" * 60)

    all_X = []
    all_y = []

    # Load plant electrophysiology data
    print("\n  Loading plant electrophysiology data...")
    X_plant, y_plant = load_plant_data(plant_dir)
    if X_plant is not None and X_plant.shape[0] > 0:
        print(f"    Plant: {X_plant.shape[0]} windows "
              f"(label 0: {np.sum(y_plant == 0)}, label 1: {np.sum(y_plant == 1)})")
        all_X.append(X_plant)
        all_y.append(y_plant)
    else:
        print("    No plant data available — skipping")

    # Load Adamatzky fungal data (all label=1)
    print("\n  Loading Adamatzky fungal data...")
    if os.path.exists(ADAMATZKY_DIR):
        X_adam, y_adam = load_adamatzky_data(ADAMATZKY_DIR, max_rows=36000)
        if X_adam.shape[0] > 0:
            print(f"    Adamatzky: {X_adam.shape[0]} windows (all label 1)")
            all_X.append(X_adam)
            all_y.append(y_adam)
    else:
        print(f"    Adamatzky dir not found: {ADAMATZKY_DIR}")

    if not all_X:
        print("  No Phase 2 data available — skipping")
        return None

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    # Augment to balance classes
    print(f"\n  Combined: {X.shape[0]} windows (label 0: {np.sum(y == 0)}, label 1: {np.sum(y == 1)})")
    X, y = augment_dataset(X, y, n_augmented_per_sample=2, balance_classes=True)
    print(f"  After augmentation: {X.shape[0]} windows (label 0: {np.sum(y == 0)}, label 1: {np.sum(y == 1)})")
    log_memory("After Phase 2 data load + augment")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_loader = make_loader(X_train, y_train, batch_size, num_workers=num_workers)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False, num_workers=num_workers)

    del X, y, X_train, X_val, y_train, y_val, all_X, all_y
    gc.collect()

    # Load Phase 1 checkpoint or build fresh
    if use_attention:
        model = build_tcn_with_attention(n_classes=5, num_blocks=num_blocks)
    else:
        model = build_tcn(n_classes=5, num_blocks=num_blocks)

    if os.path.exists(CHECKPOINT_PHASE1):
        print(f"\n  Loading Phase 1 checkpoint: {CHECKPOINT_PHASE1}")
        model.load_state_dict(
            torch.load(CHECKPOINT_PHASE1, map_location='cpu', weights_only=True),
            strict=not use_attention  # attention layers missing from ECG checkpoint
        )
    else:
        print("\n  No Phase 1 checkpoint found — starting from scratch")

    # Replace head for 2-class and freeze early blocks
    model.replace_head(new_n_classes=2)
    model.freeze_early_blocks(n=2)
    total, trainable = count_parameters(model)
    print(f"  Model: {total:,} total, {trainable:,} trainable (later blocks + head)")
    model = model.to(device)

    best_loss, history = train_phase(
        model, train_loader, val_loader, epochs, lr, device,
        patience=3, phase_name="Phase2",
        use_focal=use_focal, use_mixup=use_mixup,
        warmup_epochs=warmup_epochs,
    )
    print(f"  Best val loss: {best_loss:.4f}")

    save_checkpoint(model, CHECKPOINT_PHASE2)
    log_memory("Phase 2 complete")

    return model


# ============================================================
# PHASE 3: FINE-TUNE ON TARGET DATA
# ============================================================

def run_phase3(device, batch_size=32, epochs=10, lr=1e-4, num_workers=0,
               use_focal=True, use_mixup=True, warmup_epochs=2,
               num_blocks=5, use_attention=False):
    """Phase 3: Fine-tune on Buffi + synthetic fungal data (2-class)."""
    print("\n" + "=" * 60)
    print("PHASE 3: FINE-TUNE ON TARGET FUNGAL DATA")
    print("=" * 60)

    # Load fungal data via existing pipeline
    print("\n  Loading fungal data (synthetic + Adamatzky + Buffi)...")
    X, y = load_all_data(
        synthetic_dir=SYNTHETIC_DIR,
        adamatzky_dir=ADAMATZKY_DIR if os.path.exists(ADAMATZKY_DIR) else None,
        buffi_dir=BUFFI_DIR if os.path.exists(BUFFI_DIR) else None,
        adamatzky_max_rows=36000,
    )

    if X.shape[0] == 0:
        print("  No fungal data available — run synthetic_data.py first")
        return None

    # Augment
    X, y = augment_dataset(X, y, n_augmented_per_sample=2, balance_classes=True)
    print(f"  After augmentation: {X.shape[0]} windows "
          f"(label 0: {np.sum(y == 0)}, label 1: {np.sum(y == 1)})")
    log_memory("After Phase 3 data load + augment")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_loader = make_loader(X_train, y_train, batch_size, num_workers=num_workers)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False, num_workers=num_workers)

    # Keep val data for final evaluation
    X_val_np, y_val_np = X_val.copy(), y_val.copy()

    del X, y, X_train, X_val, y_train, y_val
    gc.collect()

    # Load Phase 2 checkpoint or Phase 1 or fresh
    if use_attention:
        model = build_tcn_with_attention(n_classes=2, num_blocks=num_blocks)
    else:
        model = build_tcn(n_classes=2, num_blocks=num_blocks)

    if os.path.exists(CHECKPOINT_PHASE2):
        print(f"\n  Loading Phase 2 checkpoint: {CHECKPOINT_PHASE2}")
        model.load_state_dict(
            torch.load(CHECKPOINT_PHASE2, map_location='cpu', weights_only=True),
            strict=not use_attention
        )
    elif os.path.exists(CHECKPOINT_PHASE1):
        print(f"\n  No Phase 2 checkpoint — loading Phase 1: {CHECKPOINT_PHASE1}")
        src = (build_tcn_with_attention(n_classes=5, num_blocks=num_blocks)
               if use_attention else build_tcn(n_classes=5, num_blocks=num_blocks))
        src.load_state_dict(
            torch.load(CHECKPOINT_PHASE1, map_location='cpu', weights_only=True),
            strict=not use_attention
        )
        model.load_encoder_state_dict(src.get_encoder_state_dict())
        del src
    else:
        print("\n  No prior checkpoint — starting from scratch")

    model.freeze_encoder()
    total, trainable = count_parameters(model)
    print(f"  Model: {total:,} total, {trainable:,} trainable (head only)")
    model = model.to(device)

    best_loss, history = train_phase(
        model, train_loader, val_loader, epochs, lr, device,
        patience=3, phase_name="Phase3",
        use_focal=use_focal, use_mixup=use_mixup,
        warmup_epochs=warmup_epochs,
    )
    print(f"  Best val loss: {best_loss:.4f}")

    save_checkpoint(model, CHECKPOINT_FINAL)
    log_memory("Phase 3 complete")

    # Run evaluation
    run_evaluation(model, X_val_np, y_val_np, device)

    return model


# ============================================================
# VOCABULARY MODE — PHASE 2: VOCABULARY CLASSIFICATION
# ============================================================

def run_phase2_vocabulary(device, batch_size=32, epochs=10, lr=1e-4, num_workers=0,
                          use_focal=True, use_mixup=True, warmup_epochs=2,
                          num_blocks=5, use_attention=False):
    """Phase 2 (vocabulary): Train on k-class vocabulary labels from spike clustering."""
    print("\n" + "=" * 60)
    print("PHASE 2 (VOCABULARY): CLASSIFY SPIKE WORD TYPES")
    print("=" * 60)

    # Load vocabulary-labeled data (from spike_vocabulary.py)
    print("\n  Loading vocabulary-labeled windows...")
    X, y, unique_labels = load_vocabulary_labeled(VOCAB_LABELS_PATH)

    if X.shape[0] == 0:
        print("  No vocabulary data — run spike_vocabulary.py first")
        return None

    # Remap labels to contiguous 0..K-1.
    # k-means labels are sparse (e.g. {0,1,5,11,38,50}) when the window cap
    # stops early — CrossEntropyLoss requires contiguous 0-based labels.
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in y], dtype=int)
    K = len(unique_labels)
    print(f"  {X.shape[0]} windows, {K} classes (word types + silence)")
    print(f"  Label remap: {dict(list(label_map.items())[:8])}{'...' if len(label_map) > 8 else ''}")

    # Compute inverse-frequency class weights to handle Zipfian imbalance
    # without oversampling (which would OOM on large vocabularies).
    counts = np.bincount(y, minlength=K).astype(float)
    raw_weights = 1.0 / (counts + 1e-6)
    class_weights = torch.FloatTensor(raw_weights / raw_weights.sum() * K).to(device)
    print(f"  Class weights (min={class_weights.min():.3f}, "
          f"max={class_weights.max():.3f})")

    # Light augmentation — no class balancing for vocabulary mode.
    # Vocabulary classes are inherently imbalanced (Zipf's law); upsampling
    # minority word types to match silence would create 900k+ windows and OOM.
    X, y = augment_dataset(X, y, n_augmented_per_sample=1, balance_classes=False)
    print(f"  After augmentation: {X.shape[0]} windows")
    log_memory("After vocab data load + augment")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_loader = make_loader(X_train, y_train, batch_size, num_workers=num_workers)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False, num_workers=num_workers)

    del X, y, X_train, X_val, y_train, y_val
    gc.collect()

    # Load Phase 1 checkpoint or build fresh
    if use_attention:
        model = build_tcn_with_attention(n_classes=5, num_blocks=num_blocks)
    else:
        model = build_tcn(n_classes=5, num_blocks=num_blocks)

    if os.path.exists(CHECKPOINT_PHASE1):
        print(f"\n  Loading Phase 1 checkpoint: {CHECKPOINT_PHASE1}")
        model.load_state_dict(
            torch.load(CHECKPOINT_PHASE1, map_location='cpu', weights_only=True),
            strict=not use_attention
        )
    else:
        print("\n  No Phase 1 checkpoint — starting from scratch")

    # Replace head for K classes and freeze early blocks
    model.replace_head(new_n_classes=K)
    model.freeze_early_blocks(n=2)
    total, trainable = count_parameters(model)
    print(f"  Model: {total:,} total, {trainable:,} trainable (later blocks + head)")
    model = model.to(device)

    # FocalLoss + class_weights: focal handles intra-class hard examples,
    # class_weights handles inter-class Zipfian imbalance (silence vs. rare words).
    best_loss, history = train_phase(
        model, train_loader, val_loader, epochs, lr, device,
        patience=3, phase_name="Phase2-Vocab", class_weights=class_weights,
        use_focal=use_focal, use_mixup=use_mixup,
        warmup_epochs=warmup_epochs,
    )
    print(f"  Best val loss: {best_loss:.4f}")

    save_checkpoint(model, CHECKPOINT_PHASE2_VOCAB)
    log_memory("Phase 2 (vocabulary) complete")

    return model


# ============================================================
# VOCABULARY MODE — PHASE 3: STIMULUS RESPONSE
# ============================================================

def run_phase3_stimulus(device, batch_size=32, epochs=10, lr=1e-4, num_workers=0,
                        use_focal=True, use_mixup=True, warmup_epochs=2,
                        num_blocks=5, use_attention=False):
    """Phase 3 (vocabulary): Fine-tune on stimulus-labeled Buffi data (5-class)."""
    print("\n" + "=" * 60)
    print("PHASE 3 (VOCABULARY): STIMULUS RESPONSE CLASSIFICATION")
    print("=" * 60)

    all_X = []
    all_y = []

    # Load Buffi data with per-stimulus labels (0-3)
    if os.path.exists(BUFFI_DIR):
        print("\n  Loading Buffi data with stimulus labels...")
        X_buffi, y_buffi = load_buffi_stimulus_labeled(BUFFI_DIR)
        if X_buffi.shape[0] > 0:
            all_X.append(X_buffi)
            all_y.append(y_buffi)
    else:
        print(f"  Buffi dir not found: {BUFFI_DIR}")

    # Add synthetic baseline as class 4
    if os.path.exists(SYNTHETIC_DIR):
        print("\n  Loading synthetic baseline data...")
        X_syn, y_syn_orig = load_synthetic_data(SYNTHETIC_DIR)
        if X_syn.shape[0] > 0:
            # Only take inactive (label=0) synthetic windows as baseline class 4
            baseline_mask = y_syn_orig == 0
            X_baseline = X_syn[baseline_mask]
            if X_baseline.shape[0] > 0:
                y_baseline = np.full(X_baseline.shape[0], 4, dtype=int)
                all_X.append(X_baseline)
                all_y.append(y_baseline)
                print(f"    Baseline (class 4): {X_baseline.shape[0]} windows")

    if not all_X:
        print("  No Phase 3 stimulus data available")
        return None

    X = np.vstack(all_X)
    y = np.concatenate(all_y)

    n_classes = len(np.unique(y))
    print(f"\n  Combined: {X.shape[0]} windows, {n_classes} classes")
    for c in sorted(np.unique(y)):
        print(f"    Class {c}: {np.sum(y == c)} windows")

    # Augment to balance classes
    X, y = augment_dataset(X, y, n_augmented_per_sample=2, balance_classes=True)
    print(f"  After augmentation: {X.shape[0]} windows")
    log_memory("After stimulus data load + augment")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    train_loader = make_loader(X_train, y_train, batch_size, num_workers=num_workers)
    val_loader = make_loader(X_val, y_val, batch_size, shuffle=False, num_workers=num_workers)

    X_val_np, y_val_np = X_val.copy(), y_val.copy()

    del X, y, X_train, X_val, y_train, y_val, all_X, all_y
    gc.collect()

    # Load Phase 2 vocabulary checkpoint, or Phase 1, or fresh
    if os.path.exists(CHECKPOINT_PHASE2_VOCAB):
        print(f"\n  Loading Phase 2 vocabulary checkpoint: {CHECKPOINT_PHASE2_VOCAB}")
        state_dict = torch.load(CHECKPOINT_PHASE2_VOCAB, map_location='cpu', weights_only=True)
        # Infer K from last linear layer weight shape
        phase2_k = state_dict['head.3.weight'].shape[0]
        if use_attention:
            model = build_tcn_with_attention(n_classes=phase2_k, num_blocks=num_blocks)
        else:
            model = build_tcn(n_classes=phase2_k, num_blocks=num_blocks)
        model.load_state_dict(state_dict, strict=not use_attention)
    elif os.path.exists(CHECKPOINT_PHASE1):
        print(f"\n  No Phase 2 vocab checkpoint — loading Phase 1: {CHECKPOINT_PHASE1}")
        if use_attention:
            model = build_tcn_with_attention(n_classes=5, num_blocks=num_blocks)
        else:
            model = build_tcn(n_classes=5, num_blocks=num_blocks)
        model.load_state_dict(
            torch.load(CHECKPOINT_PHASE1, map_location='cpu', weights_only=True),
            strict=not use_attention
        )
    else:
        print("\n  No prior checkpoint — starting from scratch")
        if use_attention:
            model = build_tcn_with_attention(n_classes=5, num_blocks=num_blocks)
        else:
            model = build_tcn(n_classes=5, num_blocks=num_blocks)

    # Replace head for stimulus classes and freeze encoder
    model.replace_head(new_n_classes=n_classes)
    model.freeze_encoder()
    total, trainable = count_parameters(model)
    print(f"  Model: {total:,} total, {trainable:,} trainable (head only)")
    model = model.to(device)

    # Mixup between stimulus classes is most impactful here:
    # cycloheximide ↔ sodiumazide confusion should decrease.
    best_loss, history = train_phase(
        model, train_loader, val_loader, epochs, lr, device,
        patience=3, phase_name="Phase3-Stimulus",
        use_focal=use_focal, use_mixup=use_mixup,
        warmup_epochs=warmup_epochs,
    )
    print(f"  Best val loss: {best_loss:.4f}")

    save_checkpoint(model, CHECKPOINT_PHASE3_STIMULUS)
    log_memory("Phase 3 (stimulus) complete")

    # Evaluate
    run_evaluation_multiclass(model, X_val_np, y_val_np, device, n_classes)

    return model


def run_evaluation_multiclass(model, X_val, y_val, device, n_classes):
    """Evaluate multi-class TCN (vocabulary or stimulus mode)."""
    from data_loader import BUFFI_STIMULI

    print("\n" + "-" * 60)
    print("MULTI-CLASS EVALUATION")
    print("-" * 60)

    model.eval()
    val_loader = make_loader(X_val, y_val, batch_size=64, shuffle=False)
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())

    preds = np.array(all_preds)

    # Build class names
    stimulus_names = {v: k for k, v in BUFFI_STIMULI.items()}
    class_names = []
    for c in range(n_classes):
        if c in stimulus_names:
            class_names.append(stimulus_names[c])
        elif c == 4:
            class_names.append('baseline')
        else:
            class_names.append(f'class_{c}')

    print(f"\n  Classification Report:")
    print(classification_report(y_val, preds, target_names=class_names,
                                zero_division=0))

    # Per-class F1
    from sklearn.metrics import f1_score as f1_multi
    f1_per = f1_multi(y_val, preds, average=None, zero_division=0)
    f1_macro = f1_multi(y_val, preds, average='macro', zero_division=0)
    print(f"  Macro F1: {f1_macro:.4f}")
    for i, name in enumerate(class_names):
        if i < len(f1_per):
            print(f"    {name}: F1={f1_per[i]:.4f}")

    # Confusion matrix plot
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_val, preds)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, cmap='Greens'
    )
    ax.set_title(f'Stimulus Response Confusion Matrix (Macro F1: {f1_macro:.3f})')
    fig.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'tcn_stimulus_results.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Results plot saved: {plot_path}")


# ============================================================
# EVALUATION
# ============================================================

def run_evaluation(model=None, X_val=None, y_val=None, device=None):
    """Evaluate TCN and compare with classical RF/SVM baselines."""
    print("\n" + "=" * 60)
    print("EVALUATION: TCN vs Classical ML Baselines")
    print("=" * 60)

    if device is None:
        device = get_device()

    # Load model if not provided
    if model is None:
        if not os.path.exists(CHECKPOINT_FINAL):
            print(f"  No TCN checkpoint found: {CHECKPOINT_FINAL}")
            print("  Run --phase 3 or --phase all first")
            return
        model = build_tcn(n_classes=2)
        model.load_state_dict(torch.load(CHECKPOINT_FINAL, map_location='cpu', weights_only=True))
        model = model.to(device)

    # Load validation data if not provided
    if X_val is None or y_val is None:
        print("\n  Loading evaluation data...")
        X, y = load_all_data(
            synthetic_dir=SYNTHETIC_DIR,
            adamatzky_dir=ADAMATZKY_DIR if os.path.exists(ADAMATZKY_DIR) else None,
            buffi_dir=BUFFI_DIR if os.path.exists(BUFFI_DIR) else None,
            adamatzky_max_rows=36000,
        )
        if X.shape[0] == 0:
            print("  No data available for evaluation")
            return
        _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # TCN predictions
    model.eval()
    val_loader = make_loader(X_val, y_val, batch_size=64, shuffle=False)
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    tcn_preds = np.array(all_preds)
    tcn_probs = np.array(all_probs)
    tcn_f1 = f1_score(y_val, tcn_preds)

    print(f"\n  TCN Classification Report:")
    print(classification_report(y_val, tcn_preds, target_names=['Inactive', 'Active']))

    # Load classical baselines for comparison
    import joblib
    rf_path = os.path.join(MODELS_DIR, 'rf_baseline.joblib')
    svm_path = os.path.join(MODELS_DIR, 'svm_baseline.joblib')

    print("\n" + "-" * 60)
    print("MODEL COMPARISON")
    print("-" * 60)

    rf_f1 = None
    svm_f1 = None

    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        # RF needs features, not raw windows — note this
        print(f"  RF baseline:  (loaded from {rf_path})")
        print(f"    Note: RF operates on 26 hand-crafted features, not raw windows.")
        print(f"    Use train.py --from-cache for RF F1 score.")
    else:
        print("  RF baseline:  not found (run train.py first)")

    if os.path.exists(svm_path):
        print(f"  SVM baseline: (loaded from {svm_path})")
        print(f"    Note: SVM operates on 26 hand-crafted features, not raw windows.")
    else:
        print("  SVM baseline: not found (run train.py first)")

    print(f"\n  TCN F1 Score: {tcn_f1:.4f}")
    print(f"  (Compare with RF/SVM F1 from train.py output)")

    # Plot results
    _plot_tcn_results(y_val, tcn_preds, tcn_probs, tcn_f1)


def _plot_tcn_results(y_true, y_pred, y_prob, f1):
    """Generate confusion matrix and ROC curve for TCN."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'TCN Classifier — Fungal Signal Detection (F1: {f1:.3f})', fontsize=13)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=['Inactive', 'Active']).plot(
        ax=axes[0], cmap='Greens'
    )
    axes[0].set_title('TCN Confusion Matrix')

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='green', linewidth=2, label=f'TCN (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'tcn_results.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Results plot saved: {plot_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    global _t0
    _t0 = time.time()

    parser = argparse.ArgumentParser(
        description='TCN transfer learning for fungal signal classification'
    )
    parser.add_argument('--phase', type=str, default='all',
                        choices=['0', '1', '2', '3', 'all'],
                        help='Phase: 0=plant pre-train, 1=ECG, 2=adapt, 3=fine-tune, all=sequential')
    parser.add_argument('--mode', type=str, default='binary',
                        choices=['binary', 'vocabulary'],
                        help='binary: detect activity | vocabulary: classify word types + stimuli')
    parser.add_argument('--evaluate', action='store_true',
                        help='Load tcn_final.pt and evaluate (skip training)')
    parser.add_argument('--full', action='store_true',
                        help='Colab mode: bigger batch, more epochs')

    # Architecture flags
    parser.add_argument('--num-blocks', type=int, default=5,
                        help='Number of TCN dilated blocks (default: 5, RF=37s). '
                             'Use 4 to reproduce baseline (RF=18s).')
    parser.add_argument('--attention', action='store_true',
                        help='Use self-attention pooling instead of global average pool. '
                             'Adds ~3k params; produces interpretable attention maps.')

    # Loss and optimizer flags
    parser.add_argument('--no-focal-loss', action='store_true',
                        help='Use CrossEntropyLoss instead of FocalLoss (disables Tier 1 focal)')
    parser.add_argument('--no-mixup', action='store_true',
                        help='Disable Mixup batch augmentation (disables Tier 1 mixup)')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='LR warmup epochs (0 = disabled). Default 2.')

    # Data paths
    parser.add_argument('--ecg-dir', type=str, default=ECG_DIR)
    parser.add_argument('--plant-dir', type=str, default=PLANT_DIR)

    args = parser.parse_args()

    ecg_dir = args.ecg_dir
    plant_dir = args.plant_dir
    use_focal = not args.no_focal_loss
    use_mixup = not args.no_mixup
    warmup_epochs = args.warmup_epochs
    num_blocks = args.num_blocks
    use_attention = args.attention

    print("=" * 60)
    print("TCN TRANSFER LEARNING — ENHANCED PIPELINE")
    print(f"Mode: {args.mode.upper()}")
    print(f"Arch: {num_blocks}-block TCN"
          f"{' + Attention' if use_attention else ''}"
          f" | FocalLoss: {use_focal}"
          f" | Mixup: {use_mixup}"
          f" | Warmup: {warmup_epochs} epochs")
    print("EE297B Research Project — SJSU")
    print("=" * 60)

    device = get_device()

    # MacBook vs Colab settings
    if args.full:
        batch_size = 64
        epochs = 20
        num_workers = 2
        print("  Power: Full (Colab)")
    else:
        batch_size = 32
        epochs = 10
        num_workers = 0
        print("  Power: MacBook-safe")

    log_memory("Start")

    if args.evaluate:
        run_evaluation(device=device)
        return

    # Shared kwargs passed to all run_phase* functions
    phase_kw = dict(
        batch_size=batch_size, epochs=epochs, num_workers=num_workers,
        use_focal=use_focal, use_mixup=use_mixup, warmup_epochs=warmup_epochs,
        num_blocks=num_blocks, use_attention=use_attention,
    )

    # --- BINARY MODE ---
    if args.mode == 'binary':
        if args.phase in ('0', 'all'):
            run_phase0_plant(device, lr=1e-3, plant_dir=plant_dir, **phase_kw)
            gc.collect()

        if args.phase in ('1', 'all'):
            run_phase1(device, lr=1e-3, ecg_dir=ecg_dir, **phase_kw)
            gc.collect()

        if args.phase in ('2', 'all'):
            run_phase2(device, lr=1e-4, plant_dir=plant_dir, **phase_kw)
            gc.collect()

        if args.phase in ('3', 'all'):
            run_phase3(device, lr=1e-4, **phase_kw)
            gc.collect()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE (BINARY)")
        print("=" * 60)
        checkpoints = [
            ('Phase 0 (Plant)', CHECKPOINT_PHASE0_PLANT),
            ('Phase 1 (ECG)', CHECKPOINT_PHASE1),
            ('Phase 2 (Adapted)', CHECKPOINT_PHASE2),
            ('Phase 3 (Final)', CHECKPOINT_FINAL),
        ]

    # --- VOCABULARY MODE ---
    elif args.mode == 'vocabulary':
        if args.phase in ('0', 'all'):
            run_phase0_plant(device, lr=1e-3, plant_dir=plant_dir, **phase_kw)
            gc.collect()

        if args.phase in ('1', 'all'):
            run_phase1(device, lr=1e-3, ecg_dir=ecg_dir, **phase_kw)
            gc.collect()

        if args.phase in ('2', 'all'):
            run_phase2_vocabulary(device, lr=1e-4, **phase_kw)
            gc.collect()

        if args.phase in ('3', 'all'):
            run_phase3_stimulus(device, lr=1e-4, **phase_kw)
            gc.collect()

        print("\n" + "=" * 60)
        print("TRAINING COMPLETE (VOCABULARY)")
        print("=" * 60)
        checkpoints = [
            ('Phase 0 (Plant)', CHECKPOINT_PHASE0_PLANT),
            ('Phase 1 (ECG)', CHECKPOINT_PHASE1),
            ('Phase 2 (Vocabulary)', CHECKPOINT_PHASE2_VOCAB),
            ('Phase 3 (Stimulus)', CHECKPOINT_PHASE3_STIMULUS),
        ]

    for name, path in checkpoints:
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {name}: {path} ({size_kb:.0f} KB)")
        else:
            print(f"  {name}: not found")

    log_memory("Final")
    print("=" * 60)


if __name__ == '__main__':
    main()
