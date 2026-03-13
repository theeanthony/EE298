# ML Pipeline Reference

## EE297B — Mapping the Language of Mycelium
**Anthony Contreras & Alex Wong | San Jose State University**

---

## Overview

This pipeline discovers and classifies bioelectrical "words" in fungal mycelium signals using Adamatzky's linguistic analogy: spikes are letters, spike trains are words, and repeated patterns form a vocabulary.

**Two complementary approaches:**

1. **Classical ML** — 26 hand-crafted features → Random Forest + SVM (interpretable baseline)
2. **TCN Deep Learning** — 1D Temporal Convolutional Network with 3-phase transfer learning (vocabulary discovery + stimulus-response decoding)

**Key result:** Vocabulary pipeline discovers ~50 word types across 4 fungal species, with a core lexicon of ~12 high-frequency words. Shannon entropy = 3.33 bits (comparable to simple natural languages).

---

## Quick Start

### Local (MacBook)

```bash
cd ml

# Install dependencies
pip install -r requirements.txt

# Download datasets
python download_data.py                    # Adamatzky + Buffi
python download_pretrain_data.py           # ECG + Plant (needs kaggle CLI for ECG)

# Generate synthetic data
python synthetic_data.py

# Classical baseline
python train.py                            # RF + SVM on 26 features
python train.py --from-cache               # Use cached features

# TCN (all 3 phases)
python train_tcn.py --phase all            # Binary mode
python train_tcn.py --mode vocabulary --phase all   # Vocabulary mode

# Vocabulary discovery
python spike_vocabulary.py                 # Discover 50 word types
python build_dictionary.py                 # Map words → stimulus meanings
```

### Google Colab (GPU)

Use `ml/colab_vocabulary_pipeline.ipynb` — runs the full vocabulary pipeline on a T4 GPU in ~20 minutes. Add `--full` flag for Colab-optimized batch sizes.

---

## TCN Architecture

```
Input: [batch, 1, 600]   (single-channel, 60s at 10 Hz)
    |
+-----------------------------+
|  TCN Encoder (66K params)   |
|  +- Block 1: d=1,  1→32    |  causal dilated conv → BN → ReLU → Dropout
|  +- Block 2: d=2,  32→32   |
|  +- Block 3: d=4,  32→32   |
|  +- Block 4: d=8,  32→32   |
|  +- Block 5: d=16, 32→32   |  (default: 5 blocks)
|  (residual connections)     |
+-----------------------------+
    |
+-----------------------------+
|  Global Average Pooling     |  or Multi-Head Attention (+Attention variant)
+-----------------------------+
    |
+-----------------------------+
|  Classification Head        |
|  FC(32,32) → ReLU → Drop   |
|  FC(32, n_classes)          |
+-----------------------------+
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| Window | 600 samples (60s) | Captures multiple spike cycles |
| Hidden channels | 32 | |
| Kernel size | 7 | 0.7s per convolution |
| Num blocks | 5 | Dilation [1,2,4,8,16] |
| Receptive field | 372 samples = 37s | Covers slow oscillation periods |
| Dropout | 0.2 (encoder), 0.3 (head) | |
| Params (5-block) | 66,981 | ~230 KB checkpoint |
| Params (+Attention) | 71,269 | Multi-head attention pooling |
| Head-only params | 1,122 | Phase 3 trains only these |

---

## Transfer Learning (3+1 Phases)

### Phase 0 (optional): Plant AP Pre-training
- **Data:** Plant electrophysiology action potentials (PMC10950275)
- **Task:** Binary classification (AP present/absent)
- **Checkpoint:** `models/tcn_phase0_plant.pt`

### Phase 1: ECG Pre-training
- **Data:** MIT-BIH + PTB Heartbeat (Kaggle, 109K samples, 5 classes)
- **Task:** 5-class heartbeat classification
- **Training:** All params trainable, AdamW lr=1e-3
- **Checkpoint:** `models/tcn_phase1_ecg.pt`

### Phase 2: Domain Adaptation
- **Binary mode:** Plant + Adamatzky → 2-class (active/inactive), freeze blocks 1-2
- **Vocabulary mode:** Adamatzky spike vocabulary → k-class word types, freeze blocks 1-2
- **Training:** AdamW lr=1e-4
- **Checkpoint:** `models/tcn_phase2_adapted.pt` or `tcn_phase2_vocabulary.pt`

### Phase 3: Fine-tuning
- **Binary mode:** Buffi + synthetic → 2-class, freeze entire encoder (head only)
- **Vocabulary mode:** Buffi stimulus-labeled → 5-class (4 stimuli + baseline), head only
- **Training:** AdamW lr=1e-4, early stopping
- **Checkpoint:** `models/tcn_final.pt` or `models/tcn_phase3_stimulus.pt`

---

## Training Enhancements

| Enhancement | Default | CLI flag to disable |
|-------------|---------|---------------------|
| FocalLoss (gamma=2) | ON | `--no-focal-loss` |
| Mixup augmentation (alpha=0.2) | ON | `--no-mixup` |
| AdamW (weight_decay=1e-4) | ON | — |
| LR warmup (2 epochs) | ON | `--warmup-epochs 0` |
| Gradient clipping (max_norm=1.0) | ON | — |
| ReduceLROnPlateau scheduler | ON | — |

To reproduce the original baseline (F1=0.549): `--num-blocks 4 --no-focal-loss --no-mixup --warmup-epochs 0`

---

## Vocabulary Pipeline

```
Adamatzky raw signals (4 species, ~84 MB)
    |
    v  spike_vocabulary.py
1. Detect spikes (baseline + 3σ threshold)
2. Group consecutive spikes into "words" (temporal threshold θ)
3. Extract per-word features (length, amplitude, duration, ISI, etc.)
4. K-means clustering → 50 word types
    |
    v  train_tcn.py --mode vocabulary
5. Phase 2: Train TCN to classify word types from raw signal windows
6. Phase 3: Train TCN to decode stimulus responses (Buffi HDF5 data)
    |
    v  build_dictionary.py
7. Run vocabulary-trained TCN on stimulus-labeled Buffi data
8. Map word types → chemical stimulus meanings → fungal dictionary
```

**Vocabulary stats (from Adamatzky data):**
- Total words discovered: 2,772
- Vocabulary size: 50 word types
- Core lexicon (>1% frequency): 12 word types
- Average word length: 6.32 spikes (Adamatzky reported 5.97; English = 4.8 letters)
- Shannon entropy: 3.33 bits (max possible: 5.64)

---

## Feature Extraction (Classical ML)

26 features per 60-second window, extracted via `feature_extractor.py`:

| Group | Features | Count |
|-------|----------|-------|
| Time domain | RMS, variance, zero-crossing rate, peak-to-peak, skewness, kurtosis | 6 |
| Spike | count, mean amplitude, rate/min, mean ISI, ISI std | 5 |
| Frequency | 4 band powers (0.01–0.05, 0.05–0.2, 0.2–0.5, 0.5–2.0 Hz), total power, dominant freq, spectral centroid | 7 |
| STFT-derived | max power, mean power, power std, spectral entropy | 4 |
| Statistical | median, IQR, Hurst exponent, autocorrelation lag-10 | 4 |

---

## Datasets

| Dataset | Source | Size | Format | Used in |
|---------|--------|------|--------|---------|
| Adamatzky | Zenodo 5790768 | 84.6 MB | .txt (tab-delimited, 1 Hz, 4 species) | Phase 2 + vocabulary |
| Buffi et al. | Mendeley 10.17632/srkxbkh6sp.1 | ~930 MB | HDF5 (4 stimuli: calcimycin, cycloheximide, sodiumazide, voriconazole) | Phase 3 + dictionary |
| ECG Heartbeat | Kaggle shayanfazeli/heartbeat | ~490 MB | CSV (109K samples, 125 Hz, 5 classes) | Phase 1 |
| Plant Electrophys | PMC10950275 | ~50 MB | .wav (10 kHz, action potentials) | Phase 0/2 |
| Synthetic | Generated locally | ~5 MB | CSV (mycelium, nothing, noise, drift) | Classical ML + Phase 3 |

**Download:** `python download_data.py` (Adamatzky + Buffi), `python download_pretrain_data.py` (ECG + Plant)

---

## File Map

| File | Purpose |
|------|---------|
| `train_tcn.py` | 3-phase TCN training (binary + vocabulary modes) |
| `train.py` | Classical ML training (RF + SVM, 5-fold CV) |
| `tcn_model.py` | TCN + TCNWithAttention architecture |
| `spike_vocabulary.py` | Adamatzky spike clustering → word types |
| `build_dictionary.py` | Map word types → stimulus meanings |
| `feature_extractor.py` | 26-feature extraction (wraps SignalProcessor) |
| `data_loader.py` | Multi-format data loading + windowing |
| `augmentation.py` | Signal cleaning + augmentation (noise, shift, scale, wander, mixup) |
| `synthetic_data.py` | Generate labeled synthetic signals |
| `download_data.py` | Download Adamatzky + Buffi datasets |
| `download_pretrain_data.py` | Download ECG + Plant datasets |
| `colab_vocabulary_pipeline.ipynb` | Full pipeline for Google Colab |
| `requirements.txt` | Python dependencies |

**Shared dependency:** `software/processing/signal_processor.py` (DSP module — filtering, STFT, spike detection)

---

## CLI Reference

```bash
# TCN training
python train_tcn.py --phase {0,1,2,3,all}     # Which phase(s) to run
python train_tcn.py --mode {binary,vocabulary}  # Classification mode
python train_tcn.py --full                      # Colab settings (batch=64, epochs=20)
python train_tcn.py --num-blocks N              # TCN depth (default: 5)
python train_tcn.py --attention                 # Use attention pooling
python train_tcn.py --no-focal-loss             # Disable FocalLoss
python train_tcn.py --no-mixup                  # Disable Mixup augmentation
python train_tcn.py --warmup-epochs N           # LR warmup epochs (default: 2)
python train_tcn.py --evaluate                  # Evaluate saved model

# Classical ML
python train.py                                 # Full training
python train.py --synthetic-only                # Synthetic data only
python train.py --from-cache                    # Use cached features
python train.py --full                          # Colab settings
python train.py --no-augment                    # Skip augmentation
python train.py --no-plots                      # Skip plot generation

# Vocabulary
python spike_vocabulary.py                      # Discover vocabulary (k=50)
python spike_vocabulary.py --n-clusters N       # Set k
python spike_vocabulary.py --analyze            # Show stats from saved model
python spike_vocabulary.py --elbow              # Elbow plot for k selection
python spike_vocabulary.py --cluster-method hdbscan  # Auto-find k

# Dictionary
python build_dictionary.py                      # Build word→stimulus mapping

# Data
python download_data.py                         # Adamatzky + Buffi
python download_pretrain_data.py --ecg          # ECG (needs Kaggle CLI)
python download_pretrain_data.py --plant        # Plant (needs gdown)
python synthetic_data.py --num-each 25 --duration 60
```
