# ML Pipeline Plan

## EE297B — Signal Processing for Fungi Propagation
**Last updated:** February 18, 2026

---

## Overview

**Goal:** Detect and classify bioelectrical signals from *Pleurotus ostreatus* mycelium growing on PCB electrodes.

**Core challenge:** Very limited fungal training data (days to weeks of recordings from our setup). Solution: **transfer learning** — pre-train on abundant bioelectrical datasets (ECG, plant signals), then fine-tune on fungal data.

---

## Implementation Progress

### Classical ML Baseline — COMPLETE
| Step | Status | File |
|------|--------|------|
| Synthetic data generator | Done | `software/ml/synthetic_data.py` |
| Dataset downloader (Adamatzky + Buffi) | Done | `software/ml/download_data.py` |
| Multi-format data loader | Done | `software/ml/data_loader.py` |
| Data augmentation (noise, shift, scale, wander) | Done | `software/ml/augmentation.py` |
| 26-feature extractor (time/freq/statistical) | Done | `software/ml/feature_extractor.py` |
| RF + SVM training + evaluation | Done | `software/ml/train.py` |
| Trained models | Done | `software/ml/models/rf_baseline.joblib`, `svm_baseline.joblib` |
| Feature cache | Done | `software/ml/models/feature_cache.npz` |

**Results:** 975 windows, 2.4s extraction, memory flat ~225 MB, F1 stable.

### TCN Deep Learning — IMPLEMENTED, NEEDS DATA + TRAINING
| Step | Status | File |
|------|--------|------|
| TCN model architecture | Done | `software/ml/tcn_model.py` |
| Pre-training data downloader (ECG + Plant) | Done | `software/ml/download_pretrain_data.py` |
| 3-phase training script | Done | `software/ml/train_tcn.py` |
| Phase 1: Pre-train on ECG | **Needs ECG data download** | `models/tcn_phase1_ecg.pt` |
| Phase 2: Domain-adapt on plant+Adamatzky | **Needs plant data download** | `models/tcn_phase2_adapted.pt` |
| Phase 3: Fine-tune on fungal data | **Ready to run** | `models/tcn_final.pt` |
| Evaluation + RF/SVM comparison | **After Phase 3** | plots in `data/ml_results/` |

### Future (After Inoculation)
| Step | Status | File |
|------|--------|------|
| Real-time TCN inference in analyzer | Not started | integrate into `realtime_analyzer.py` |
| Closed-loop actuator control via ML | Not started | — |
| Long-duration headless logger | Not started | — |

---

## How to Run

### Local (MacBook)

```bash
# 0. Install dependencies
pip install -r requirements-ml.txt

# 1. Download pre-training data
cd software/ml
python download_pretrain_data.py             # Both ECG + Plant
python download_pretrain_data.py --ecg       # ECG only (needs kaggle CLI)
python download_pretrain_data.py --plant     # Plant only (needs gdown)

# 2. Download fungal data (Adamatzky + Buffi) if not already done
python download_data.py

# 3. Generate synthetic data if not already done
python synthetic_data.py

# 4. Run TCN training (MacBook-safe defaults: batch=32, epochs=10, workers=0)
python train_tcn.py --phase 1               # Phase 1: ECG pre-train (~15 min CPU)
python train_tcn.py --phase 2               # Phase 2: Domain adapt (~3 min CPU)
python train_tcn.py --phase 3               # Phase 3: Fine-tune (~1 min CPU)
python train_tcn.py --phase all             # All 3 phases back-to-back (~20 min)

# 5. Evaluate and compare with classical baselines
python train_tcn.py --evaluate

# 6. (Optional) Run classical baseline for comparison
python train.py                              # RF + SVM on hand-crafted features
python train.py --from-cache                 # Skip feature extraction, use cached
```

### Google Colab

```python
# === Cell 1: Clone repo + install ===
!git clone https://github.com/YOUR_REPO/EE297B_ResearchProject.git
%cd EE297B_ResearchProject
!pip install -r requirements-ml.txt

# === Cell 2: Download ALL datasets ===
%cd software/ml
!python download_data.py                     # Adamatzky + Buffi
!python download_pretrain_data.py            # ECG + Plant
!python synthetic_data.py                    # Generate synthetic data

# === Cell 3: Train classical baseline (for comparison) ===
!python train.py --full                      # RF + SVM with full power

# === Cell 4: TCN Phase 1 — ECG pre-training ===
!python train_tcn.py --phase 1 --full        # ~5 min on T4 GPU

# === Cell 5: TCN Phase 2 — Domain adaptation ===
!python train_tcn.py --phase 2 --full        # ~1 min on T4 GPU

# === Cell 6: TCN Phase 3 — Fine-tune + evaluate ===
!python train_tcn.py --phase 3 --full        # ~30 sec on T4 GPU

# === Cell 7: (Alternative) Run all 3 phases in one shot ===
# !python train_tcn.py --phase all --full    # ~7 min total on T4

# === Cell 8: Evaluate final model ===
!python train_tcn.py --evaluate

# === Cell 9: Download trained models ===
from google.colab import files
files.download('models/tcn_final.pt')
files.download('models/tcn_phase1_ecg.pt')
files.download('models/tcn_phase2_adapted.pt')
```

**Colab notes:**
- `--full` flag = batch=64, epochs=20, num_workers=2 (uses GPU properly)
- Device auto-detects CUDA on Colab, MPS on Mac, CPU as fallback
- Total training time on free T4: ~7 minutes for all 3 phases
- Download the .pt checkpoint files at the end to use locally

### Kaggle CLI Setup (for ECG data)

```bash
pip install kaggle
# Go to kaggle.com -> Settings -> API -> Create New Token
# Save the downloaded kaggle.json to ~/.kaggle/
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
# Verify
kaggle datasets list -s heartbeat
```

---

## Architecture: 1D Temporal Convolutional Network (TCN)

### Why TCN over RNN/LSTM?
- Parallelizable (faster training)
- Stable gradients (no vanishing gradient problem)
- Flexible receptive field via dilated convolutions
- Proven on ECG/EEG classification benchmarks
- Causal convolutions = suitable for real-time inference on Arduino pipeline

### Model Structure

```
Input: [batch, 1, 600]   (single-channel, 60s at 10 Hz)
    |
    v
+---------------------------+
|  TCN Encoder (52K params) |
|  +- Block 1: d=1, 1->32  |  causal conv -> BN -> ReLU -> Dropout
|  +- Block 2: d=2, 32->32 |  causal conv -> BN -> ReLU -> Dropout
|  +- Block 3: d=4, 32->32 |  causal conv -> BN -> ReLU -> Dropout
|  +- Block 4: d=8, 32->32 |  causal conv -> BN -> ReLU -> Dropout
|  (residual connections)   |
+---------------------------+
    |
    v
+---------------------------+
|  Global Average Pooling   |  (batch, 32, 600) -> (batch, 32)
+---------------------------+
    |
    v
+---------------------------+
|  Classification Head      |
|  FC(32,32) -> ReLU        |
|  Dropout(0.3)             |
|  FC(32, n_classes)        |
+---------------------------+
    |
    v
Output: logits (batch, n_classes)
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Input window | 600 samples (60s at 10 Hz) | Captures multiple spike cycles |
| Hidden channels | 32 | Small model — limited data |
| Kernel size | 7 | Covers 0.7s at 10 Hz per conv |
| Num blocks | 4 | Dilation [1,2,4,8] |
| Receptive field | 180 samples = 18 seconds | 2*(7-1)*(1+2+4+8) |
| Dropout | 0.2 (encoder), 0.3 (head) | Regularization for small datasets |
| Total params | 52,453 | ~60 KB on disk |
| Head-only params | 1,122 | Phase 3 trains only these |

---

## Transfer Learning Strategy (3 Phases)

### Phase 1: Pre-train on ECG Heartbeat Data
**Goal:** Learn general bioelectrical feature extraction (spike shapes, oscillation patterns, noise rejection)

**Dataset:** Kaggle `shayanfazeli/heartbeat` — 109K preprocessed ECG heartbeats (MIT-BIH + PTB)
- 187 samples per heartbeat at 125 Hz, 5 classes (Normal, Supraventricular, Ventricular, Fusion, Unknown)
- Preprocessing: concatenate same-class heartbeats -> resample 125 Hz to 10 Hz -> segment into 600-sample windows -> z-score normalize

**Training:** All parameters trainable, Adam lr=1e-3, 5-class classification
**Checkpoint:** `models/tcn_phase1_ecg.pt`

### Phase 2: Domain Adaptation
**Goal:** Bridge from general bioelectrical -> organism-specific signals

**Datasets:**
- **Plant Electrophysiology** (PMC10950275) — .wav at 10 kHz, decimated to 10 Hz, stimulus event labels
- **Adamatzky Fungi** (Zenodo) — 4 species, 1 Hz resampled to 10 Hz (already downloaded)

**Strategy:**
- Load Phase 1 checkpoint
- Replace head: 5-class -> 2-class (active/inactive)
- Freeze blocks 1-2 (general feature extractors)
- Train blocks 3-4 + head, Adam lr=1e-4
**Checkpoint:** `models/tcn_phase2_adapted.pt`

### Phase 3: Fine-tune on Target Data
**Goal:** Specialize for our specific hardware setup (INA128 + Arduino R4 + PCB electrodes)

**Data:** Buffi et al. (HDF5) + Adamatzky + synthetic data via existing `load_all_data()` + augmentation

**Strategy:**
- Load Phase 2 checkpoint
- Freeze entire encoder — only classification head trainable (1,122 params)
- Adam lr=1e-4, early stopping
**Checkpoint:** `models/tcn_final.pt` (production model)

---

## Classification Tasks (in order of feasibility)

### Task A: Binary Detection (primary goal)
- **Classes:** mycelium activity vs. baseline/noise
- **When:** As soon as we have ~1 day of real recordings
- **Metric:** F1-score, sensitivity (we care more about catching signals than false positives)

### Task B: Stimulus Response Classification (stretch goal)
- **Classes:** baseline / light response / humidity response / spontaneous spike
- **When:** After stimulus experiments (week 9-10 per timeline)
- **Metric:** Accuracy, confusion matrix

### Task C: Growth Stage Detection (aspirational)
- **Classes:** pre-colonization / colonizing / established / fruiting
- **When:** Requires multi-week continuous recording
- **Metric:** Accuracy over time

---

## Feature Extraction Pipeline (Classical Baseline)

26 features extracted from each 60-second window:

### Time Domain (6)
| Feature | Description |
|---------|-------------|
| RMS amplitude | Root mean square of signal |
| Variance | Signal variance |
| Zero-crossing rate | Rate of sign changes |
| Peak-to-peak | Max - min voltage |
| Skewness | Asymmetry of distribution |
| Kurtosis | Tail heaviness |

### Spike Features (5)
| Feature | Description |
|---------|-------------|
| Spike count | Threshold crossings (baseline + 3sigma) |
| Mean spike amplitude | Average height of detected spikes |
| Spike rate per minute | Spikes per minute |
| Mean ISI | Mean inter-spike interval |
| ISI std | ISI variability |

### Frequency Domain (7)
| Feature | Description |
|---------|-------------|
| Band power (ultra-low) | 0.01-0.05 Hz |
| Band power (low) | 0.05-0.2 Hz |
| Band power (mid) | 0.2-0.5 Hz |
| Band power (high) | 0.5-2.0 Hz |
| Total power | Sum of all bands |
| Dominant frequency | Frequency with max power |
| Spectral centroid | Center of mass of spectrum |

### STFT-Derived (4)
| Feature | Description |
|---------|-------------|
| STFT max power | Peak power in spectrogram |
| STFT mean power | Average spectrogram power |
| STFT power std | Spectrogram variability |
| Spectral entropy | Frequency distribution disorder |

### Statistical Shape (4)
| Feature | Description |
|---------|-------------|
| Median | Median voltage |
| IQR | Interquartile range |
| Hurst exponent | Long-range dependence (fractal) |
| Autocorrelation lag-10 | 1-second temporal dependency |

---

## Compute Estimates

| Phase | Data Size | Epochs | Colab T4 | MacBook CPU |
|-------|-----------|--------|----------|-------------|
| 1: ECG pre-train | ~50K windows | 10 | ~5 min | ~15 min |
| 2: Domain adapt | ~5K windows | 10 | ~1 min | ~3 min |
| 3: Fine-tune | ~2K windows | 10 | ~30 sec | ~1 min |
| **Total** | | | **~7 min** | **~20 min** |

Model: 52K params = ~60 KB on disk. Peak memory: <500 MB.

---

## File Locations

| Component | Path | Status |
|-----------|------|--------|
| DSP / feature extraction | `software/processing/signal_processor.py` | Done |
| Real-time visualization | `software/processing/realtime_analyzer.py` | Done |
| Offline analysis | `software/processing/analyze_recording.py` | Done |
| Synthetic data generator | `software/ml/synthetic_data.py` | Done |
| Dataset downloader (fungal) | `software/ml/download_data.py` | Done |
| Dataset downloader (ECG + plant) | `software/ml/download_pretrain_data.py` | Done |
| Multi-format data loader | `software/ml/data_loader.py` | Done |
| Data augmentation | `software/ml/augmentation.py` | Done |
| Feature extractor (26 features) | `software/ml/feature_extractor.py` | Done |
| Classical ML training (RF + SVM) | `software/ml/train.py` | Done |
| TCN model architecture | `software/ml/tcn_model.py` | Done |
| TCN 3-phase training | `software/ml/train_tcn.py` | Done |
| RF baseline model | `software/ml/models/rf_baseline.joblib` | Trained |
| SVM baseline model | `software/ml/models/svm_baseline.joblib` | Trained |
| Feature cache | `software/ml/models/feature_cache.npz` | Cached |
| TCN Phase 1 checkpoint | `software/ml/models/tcn_phase1_ecg.pt` | Not yet trained |
| TCN Phase 2 checkpoint | `software/ml/models/tcn_phase2_adapted.pt` | Not yet trained |
| TCN final model | `software/ml/models/tcn_final.pt` | Not yet trained |
| ML result plots | `data/ml_results/` | Classical done, TCN pending |
| Dependencies | `requirements-ml.txt` | Done |

---

## Fallback Plan

Per the risk assessment in `Project_Gameplan_Report.md`:

> If timeline pressure builds: Drop TCN/ML classifier -> use threshold-based detection instead

The threshold-based detector (baseline + 3sigma) already exists in `signal_processor.py`. The ML pipeline is aspirational improvement, not a hard requirement for the deliverable. Even partial results (e.g., "pre-trained model + preliminary fine-tuning") are publishable.

**Classical ML baseline is already complete** — RF and SVM are trained and evaluated. Even if TCN training doesn't happen before the deadline, we have a working ML story for the paper.
