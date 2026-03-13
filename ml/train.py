#!/usr/bin/env python3
"""
train.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

End-to-end training script for classical ML baseline.
Trains Random Forest + SVM on extracted features, reports metrics, saves models.

Usage:
    python train.py --synthetic-only          # Train on synthetic data only
    python train.py                           # Train on synthetic + Adamatzky
    python train.py --no-plots                # Skip plot generation
    python train.py --full                    # Full-power mode (for Colab)
    python train.py --from-cache              # Load cached features (skip load/augment/extract)
"""

import numpy as np
import os
import sys
import gc
import argparse
import time
import joblib
import psutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# Local imports
from feature_extractor import extract_features_batch, FEATURE_NAMES
from data_loader import load_all_data
from augmentation import augment_dataset

# Paths
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
SYNTHETIC_DIR = os.path.join(PROJECT_ROOT, 'data', 'synthetic')
ADAMATZKY_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'adamatzky')
BUFFI_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'buffi')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'ml_results')
CACHE_PATH = os.path.join(MODELS_DIR, 'feature_cache.npz')

# Track start time for elapsed reporting
_t0 = time.time()


def log_memory(stage: str):
    """Print RSS memory usage and elapsed time for a pipeline stage."""
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    elapsed = time.time() - _t0
    print(f"  [{elapsed:6.1f}s] {stage}: {rss_mb:.0f} MB RSS")


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       feature_names, save_plots=True, full_power=False):
    """
    Train RF and SVM classifiers, evaluate, and save results.

    Args:
        full_power: If True, use aggressive settings (for Colab). If False,
                    use MacBook-safe settings.

    Returns:
        best_model: The better-performing model (Pipeline with scaler)
        results: Dict of metrics for both models
    """
    results = {}

    # Model hyperparams — tune down for MacBook, full power for Colab
    if full_power:
        rf_n_estimators = 200
        rf_n_jobs = -1
        svm_cache_size = 500  # More cache for Colab
    else:
        rf_n_estimators = 100
        rf_n_jobs = 2
        svm_cache_size = 200  # sklearn default, MacBook-safe

    # --- Random Forest ---
    print("\n" + "=" * 60)
    print("RANDOM FOREST")
    print("=" * 60)

    rf_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=rf_n_estimators,
            max_depth=15,
            class_weight='balanced',
            random_state=42,
            n_jobs=rf_n_jobs
        ))
    ])

    # 5-fold cross-validation on training set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_cv_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv, scoring='f1')
    print(f"5-fold CV F1: {rf_cv_scores.mean():.4f} +/- {rf_cv_scores.std():.4f}")

    # Train on full training set
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=['Inactive', 'Active']))

    results['rf'] = {
        'cv_f1_mean': rf_cv_scores.mean(),
        'cv_f1_std': rf_cv_scores.std(),
        'predictions': rf_pred,
        'probabilities': rf_proba,
        'pipeline': rf_pipeline,
    }
    log_memory("After RF training")

    # --- SVM (RBF kernel) ---
    print("\n" + "=" * 60)
    print("SVM (RBF KERNEL)")
    print("=" * 60)

    svm_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42,
            cache_size=svm_cache_size
        ))
    ])

    svm_cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=cv, scoring='f1')
    print(f"5-fold CV F1: {svm_cv_scores.mean():.4f} +/- {svm_cv_scores.std():.4f}")

    svm_pipeline.fit(X_train, y_train)
    svm_pred = svm_pipeline.predict(X_test)
    svm_proba = svm_pipeline.predict_proba(X_test)[:, 1]

    print("\nTest Set Classification Report:")
    print(classification_report(y_test, svm_pred, target_names=['Inactive', 'Active']))

    results['svm'] = {
        'cv_f1_mean': svm_cv_scores.mean(),
        'cv_f1_std': svm_cv_scores.std(),
        'predictions': svm_pred,
        'probabilities': svm_proba,
        'pipeline': svm_pipeline,
    }
    log_memory("After SVM training")

    # --- Determine best model ---
    if results['rf']['cv_f1_mean'] >= results['svm']['cv_f1_mean']:
        best_name = 'rf'
        print("\n>>> Best model: Random Forest")
    else:
        best_name = 'svm'
        print("\n>>> Best model: SVM")

    best_model = results[best_name]['pipeline']

    # --- Save models ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    rf_path = os.path.join(MODELS_DIR, 'rf_baseline.joblib')
    svm_path = os.path.join(MODELS_DIR, 'svm_baseline.joblib')
    best_path = os.path.join(MODELS_DIR, 'best_model.joblib')

    joblib.dump(rf_pipeline, rf_path)
    joblib.dump(svm_pipeline, svm_path)
    joblib.dump(best_model, best_path)
    print(f"\nModels saved to {MODELS_DIR}/")

    # --- Generate plots ---
    if save_plots:
        plot_results(y_test, results, feature_names, rf_pipeline)

    return best_model, results


def plot_results(y_test, results, feature_names, rf_pipeline):
    """Generate confusion matrix, ROC curve, and feature importance plots."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Classical ML Baseline — Fungal Signal Detection', fontsize=14)

    # 1. RF Confusion Matrix
    cm_rf = confusion_matrix(y_test, results['rf']['predictions'])
    ConfusionMatrixDisplay(cm_rf, display_labels=['Inactive', 'Active']).plot(
        ax=axes[0, 0], cmap='Blues'
    )
    axes[0, 0].set_title(f"Random Forest (CV F1: {results['rf']['cv_f1_mean']:.3f})")

    # 2. SVM Confusion Matrix
    cm_svm = confusion_matrix(y_test, results['svm']['predictions'])
    ConfusionMatrixDisplay(cm_svm, display_labels=['Inactive', 'Active']).plot(
        ax=axes[0, 1], cmap='Oranges'
    )
    axes[0, 1].set_title(f"SVM RBF (CV F1: {results['svm']['cv_f1_mean']:.3f})")

    # 3. ROC Curves
    for name, color in [('rf', 'blue'), ('svm', 'orange')]:
        fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
        roc_auc = auc(fpr, tpr)
        label = f"{'RF' if name == 'rf' else 'SVM'} (AUC = {roc_auc:.3f})"
        axes[1, 0].plot(fpr, tpr, color=color, linewidth=2, label=label)

    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curves')
    axes[1, 0].legend(loc='lower right')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Feature Importance (RF)
    rf_clf = rf_pipeline.named_steps['clf']
    importances = rf_clf.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15 features

    axes[1, 1].barh(range(len(indices)),
                     importances[indices],
                     color='steelblue', edgecolor='black')
    axes[1, 1].set_yticks(range(len(indices)))
    axes[1, 1].set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 15 Feature Importances (RF)')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'ml_baseline_results.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Results plot saved to {plot_path}")


def main():
    global _t0
    _t0 = time.time()

    parser = argparse.ArgumentParser(
        description='Train classical ML baseline for fungal signal detection'
    )
    parser.add_argument('--synthetic-only', action='store_true',
                        help='Train on synthetic data only (skip Adamatzky)')
    parser.add_argument('--synthetic-dir', type=str, default=SYNTHETIC_DIR,
                        help='Path to synthetic data directory')
    parser.add_argument('--adamatzky-dir', type=str, default=ADAMATZKY_DIR,
                        help='Path to Adamatzky dataset directory')
    parser.add_argument('--buffi-dir', type=str, default=BUFFI_DIR,
                        help='Path to Buffi HDF5 dataset directory')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--no-augment', action='store_true',
                        help='Skip data augmentation')
    parser.add_argument('--full', action='store_true',
                        help='Full-power mode: more trees, all cores (for Colab)')
    parser.add_argument('--from-cache', action='store_true',
                        help='Load cached feature matrix from .npz (skip load/augment/extract)')

    args = parser.parse_args()

    print("=" * 60)
    print("CLASSICAL ML BASELINE — FUNGAL SIGNAL DETECTION")
    print("EE297B Research Project — SJSU")
    print("=" * 60)
    if not args.full:
        print("  (MacBook mode: reduced parallelism — use --full for Colab)")
    log_memory("Start")

    if args.from_cache:
        # --- Load cached features ---
        if not os.path.exists(CACHE_PATH):
            print(f"\nERROR: No feature cache found at {CACHE_PATH}")
            print("  Run without --from-cache first to generate it.")
            sys.exit(1)
        print(f"\n--- LOADING CACHED FEATURES from {CACHE_PATH} ---")
        cached = np.load(CACHE_PATH)
        X_features = cached['X_features']
        y = cached['y']
        print(f"Loaded {X_features.shape[0]} samples x {X_features.shape[1]} features")
        log_memory("After cache load")
    else:
        # --- 1. Load data ---
        print("\n--- LOADING DATA ---")
        adamatzky = None if args.synthetic_only else args.adamatzky_dir
        buffi = None if args.synthetic_only else args.buffi_dir
        adamatzky_cap = 360000 if args.full else 36000  # 100h Colab vs 10h MacBook
        X_windows, y = load_all_data(
            synthetic_dir=args.synthetic_dir,
            adamatzky_dir=adamatzky,
            buffi_dir=buffi,
            adamatzky_max_rows=adamatzky_cap,
        )

        if X_windows.shape[0] == 0:
            print("\nERROR: No data loaded. Run synthetic_data.py first:")
            print("  python software/ml/synthetic_data.py")
            sys.exit(1)

        log_memory("After data load")

        # --- 2. Data augmentation ---
        if not args.no_augment:
            print("\n--- DATA AUGMENTATION ---")
            print(f"Before augmentation: {X_windows.shape[0]} windows "
                  f"(label 0: {np.sum(y == 0)}, label 1: {np.sum(y == 1)})")
            X_windows, y = augment_dataset(X_windows, y,
                                            n_augmented_per_sample=2,
                                            balance_classes=True)
            print(f"After augmentation:  {X_windows.shape[0]} windows "
                  f"(label 0: {np.sum(y == 0)}, label 1: {np.sum(y == 1)})")
        else:
            print("\n--- SKIPPING AUGMENTATION ---")
        log_memory("After augmentation")
        gc.collect()

        # --- 3. Extract features ---
        print("\n--- EXTRACTING FEATURES ---")
        t_start = time.time()
        X_features = extract_features_batch(X_windows)
        elapsed = time.time() - t_start
        print(f"Extracted {X_features.shape[1]} features from {X_features.shape[0]} windows in {elapsed:.1f}s")
        log_memory("After feature extraction")

        # Free raw windows — no longer needed
        del X_windows
        gc.collect()
        log_memory("After gc (windows freed)")

        # NaN/Inf safety net (imputer in pipeline handles remaining NaNs properly)
        n_nan = np.sum(~np.isfinite(X_features))
        if n_nan > 0:
            print(f"  Note: {n_nan} NaN/Inf values found — will be imputed with median in pipeline")

        # --- Checkpoint features to disk ---
        os.makedirs(MODELS_DIR, exist_ok=True)
        np.savez_compressed(CACHE_PATH, X_features=X_features, y=y)
        print(f"  Feature cache saved to {CACHE_PATH}")

    # --- 4. Train/test split ---
    print("\n--- TRAIN/TEST SPLIT ---")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} samples (label 0: {np.sum(y_train == 0)}, label 1: {np.sum(y_train == 1)})")
    print(f"Test:  {X_test.shape[0]} samples (label 0: {np.sum(y_test == 0)}, label 1: {np.sum(y_test == 1)})")

    # --- 5. Train and evaluate ---
    best_model, results = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        FEATURE_NAMES,
        save_plots=not args.no_plots,
        full_power=args.full
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  RF  CV F1: {results['rf']['cv_f1_mean']:.4f} +/- {results['rf']['cv_f1_std']:.4f}")
    print(f"  SVM CV F1: {results['svm']['cv_f1_mean']:.4f} +/- {results['svm']['cv_f1_std']:.4f}")
    print(f"  Models saved to: {MODELS_DIR}/")
    if not args.no_plots:
        print(f"  Plots saved to:  {PLOTS_DIR}/")
    log_memory("Final")
    print("=" * 60)


if __name__ == '__main__':
    main()
