# Plan: Classical ML Baseline for Fungal Signal Detection

## Context
We need a working ML pipeline **before inoculation** so it's ready when real data arrives (~day 4 post-inoculation). The classical ML baseline (features + Random Forest/SVM) is the practical first approach — works with small datasets, interpretable for the paper, and validates the feature extraction pipeline.

Training data comes from: (1) the Adamatzky Zenodo dataset (4 fungal species, 84.6 MB), (2) synthetic signals generated in Python mirroring the Arduino simulator's MYCELIUM mode, and (3) eventually our own recordings.

## New Files

All new code goes in `software/ml/`:

```
software/ml/
├── __init__.py
├── feature_extractor.py    # Wraps signal_processor.py → 26-feature vector per window
├── data_loader.py          # Loads our CSV + Adamatzky .txt, segments into windows, labels
├── synthetic_data.py       # Python port of Arduino MYCELIUM mode + baseline generators
├── train.py                # Train RF + SVM, cross-validate, save models, plot results
├── download_data.py        # Downloads Adamatzky Zenodo dataset to data/external/
└── models/                 # Saved .joblib model files (gitignored)
```

Also create:
- `data/external/` — downloaded datasets (gitignored)
- `data/synthetic/` — generated synthetic CSVs (gitignored)
- `requirements-ml.txt` — ML-specific dependencies

## Implementation Details

### 1. `feature_extractor.py` — 26 features per 60-second window

Wraps `signal_processor.py` (no DSP duplication). For each window:

**Time domain (6):** RMS, variance, zero-crossing rate, peak-to-peak, skewness, kurtosis

**Spike features (5):** spike count, mean spike amplitude, spike rate/min, mean ISI, ISI std — via `SignalProcessor.detect_spikes()` + `characterize_spikes()`

**Frequency domain (7):** 4 band powers (ultra_low/low/mid/high), total power, dominant frequency, spectral centroid — via `SignalProcessor.get_frequency_bands()` + `get_dominant_frequency()`

**STFT-derived (4):** max STFT power, mean STFT power, STFT power std, spectral entropy — via `SignalProcessor.compute_spectrogram()`

**Statistical shape (4):** median, IQR, Hurst exponent estimate, autocorrelation at lag-10

Returns a numpy array + feature name list.

### 2. `data_loader.py` — Multi-format loader + windowing

- **Our CSV:** `timestamp_ms, adc_raw, voltage_mV` → extract voltage_mV column
- **Adamatzky .txt:** Tab/space-delimited, 8 electrode pair columns, 1 Hz. Resample to 10 Hz via interpolation. Each channel treated as separate recording.
- **Windowing:** Slide 60-second windows (600 samples at 10 Hz) with 50% overlap
- **Labeling:**
  - Adamatzky data → label=1 (active fungal signal)
  - Synthetic baseline/noise → label=0
  - Our future data → manual labeling or threshold-based

### 3. `synthetic_data.py` — Generate labeled training data in Python

Port the key Arduino simulator modes to Python (no Arduino needed):
- **MYCELIUM mode** (label=1): 3 overlapping oscillations + wandering baseline + action potential trains + biological noise (mirrors `generateMycelium()` from firmware)
- **NOTHING mode** (label=0): flat baseline + tiny noise
- **NOISE mode** (label=0): pure random
- **DRIFT mode** (label=0): slow electrode drift

Generates CSVs in our format to `data/synthetic/`. Configurable duration and number of recordings.

### 4. `train.py` — End-to-end training script

1. Load data (synthetic + Adamatzky via data_loader)
2. Extract features (feature_extractor)
3. Train/test split (80/20, stratified)
4. Train Random Forest + SVM (RBF kernel) with 5-fold cross-validation
5. Print classification report (precision, recall, F1)
6. Plot: confusion matrix, feature importance (RF), ROC curve
7. Save best model to `software/ml/models/`

### 5. `download_data.py` — Adamatzky dataset fetcher

Downloads from Zenodo (https://zenodo.org/records/5790768), extracts .txt.zip files to `data/external/adamatzky/`. Simple wget/urllib with progress.

## Dependencies (`requirements-ml.txt`)

```
scikit-learn>=1.3
joblib
pandas
```

(numpy, scipy, matplotlib already required by signal_processor.py)

## Verification

1. Run `synthetic_data.py` → generates CSVs in `data/synthetic/`
2. Run `train.py --synthetic-only` → trains on synthetic data, prints metrics
3. Run `download_data.py` → fetches Adamatzky dataset
4. Run `train.py` → trains on synthetic + Adamatzky, compare metrics
5. Sanity check: model should easily separate MYCELIUM vs NOTHING synthetic signals (>95% accuracy), and perform reasonably on Adamatzky (>80%)

## Key Reuse

- `signal_processor.py:SignalProcessor` — all filtering, STFT, spike detection
- `signal_processor.py:load_csv_data()` — CSV parsing
- `analyze_recording.py` — pattern for stats computation (skew/kurtosis imports)
