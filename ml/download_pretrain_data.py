#!/usr/bin/env python3
"""
download_pretrain_data.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Downloads external datasets for TCN pre-training and domain adaptation:
  1. ECG Heartbeat (Phase 1) — MIT-BIH + PTB from Kaggle
  2. Plant Electrophysiology (Phase 2) — from Google Drive (PMC10950275)

Usage:
    python download_pretrain_data.py             # Download both
    python download_pretrain_data.py --ecg       # ECG only
    python download_pretrain_data.py --plant     # Plant only
"""

import os
import sys
import argparse
import subprocess
import zipfile
import urllib.request

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
DEFAULT_ECG_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'ecg_heartbeat')
DEFAULT_PLANT_DIR = os.path.join(PROJECT_ROOT, 'data', 'external', 'plant_electrophys')

# ============================================================
# ECG HEARTBEAT DATASET (Kaggle)
# Source: kaggle.com/datasets/shayanfazeli/heartbeat
# Preprocessed MIT-BIH Arrhythmia + PTB Diagnostic ECG
# Format: CSV — each row is 187 signal values + 1 label (0-4)
# Total: ~109K rows (87K train + 22K test)
# ============================================================
KAGGLE_DATASET = 'shayanfazeli/heartbeat'
ECG_EXPECTED_FILES = ['mitbih_train.csv', 'mitbih_test.csv']

# ============================================================
# PLANT ELECTROPHYSIOLOGY DATASET (Google Drive)
# Source: PMC10950275 supplementary data
# Google Drive folder: https://t.ly/imDXR
# Format: .wav files at 10 kHz + .txt event markers
# ============================================================
# Direct Google Drive folder ID (from the shortened URL)
PLANT_GDRIVE_FOLDER_ID = '1ViRN1OUpVUJfxnVQ6LIoHqgbOqJEryOz'
PLANT_GDRIVE_URL = f'https://drive.google.com/drive/folders/{PLANT_GDRIVE_FOLDER_ID}'


def download_file(url: str, dest_path: str):
    """Download a file with progress display (reused from download_data.py pattern)."""
    print(f"  Downloading {os.path.basename(dest_path)}...")

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')]
    urllib.request.install_opener(opener)

    def progress_hook(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f'\r    {pct}% ({mb_down:.1f}/{mb_total:.1f} MB)')
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest_path, reporthook=progress_hook)
    print()


def extract_zip(zip_path: str, extract_dir: str):
    """Extract a zip file."""
    print(f"  Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)


def _check_kaggle_cli():
    """Check if kaggle CLI is available."""
    try:
        result = subprocess.run(
            ['kaggle', '--version'],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_ecg(output_dir: str = DEFAULT_ECG_DIR):
    """
    Download the ECG Heartbeat Classification dataset from Kaggle.

    Requires: pip install kaggle + API token in ~/.kaggle/kaggle.json
    See: https://www.kaggle.com/docs/api#authentication
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ECG HEARTBEAT — MIT-BIH + PTB (Phase 1 Pre-training)")
    print(f"Kaggle dataset: {KAGGLE_DATASET}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print("=" * 60)

    # Check if already downloaded
    existing = [f for f in ECG_EXPECTED_FILES if os.path.exists(os.path.join(output_dir, f))]
    if len(existing) == len(ECG_EXPECTED_FILES):
        print("\n[SKIP] ECG data already downloaded:")
        for f in existing:
            size_mb = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
            print(f"  {f} ({size_mb:.1f} MB)")
        return 0

    # Try kaggle CLI
    if _check_kaggle_cli():
        print("\nUsing kaggle CLI...")
        try:
            result = subprocess.run(
                ['kaggle', 'datasets', 'download', '-d', KAGGLE_DATASET, '-p', output_dir, '--unzip'],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                # Verify files exist
                downloaded = [f for f in ECG_EXPECTED_FILES if os.path.exists(os.path.join(output_dir, f))]
                print(f"  Downloaded {len(downloaded)} files:")
                for f in downloaded:
                    size_mb = os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
                    print(f"    {f} ({size_mb:.1f} MB)")
                return len(downloaded)
            else:
                print(f"  kaggle CLI error: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            print("  kaggle CLI timed out")
    else:
        print("\nkaggle CLI not found.")

    # Fallback: manual instructions
    print("\n" + "-" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("-" * 60)
    print("The kaggle CLI is not configured. To set it up:")
    print("  1. pip install kaggle")
    print("  2. Go to kaggle.com -> Settings -> API -> Create New Token")
    print("  3. Save kaggle.json to ~/.kaggle/kaggle.json")
    print("  4. chmod 600 ~/.kaggle/kaggle.json")
    print("  5. Re-run this script")
    print()
    print("Or download manually:")
    print(f"  https://www.kaggle.com/datasets/{KAGGLE_DATASET}")
    print(f"  Extract mitbih_train.csv and mitbih_test.csv to:")
    print(f"    {os.path.abspath(output_dir)}/")
    print("-" * 60)
    return 0


def download_plant(output_dir: str = DEFAULT_PLANT_DIR):
    """
    Download the plant electrophysiology dataset from Google Drive.

    Requires: pip install gdown
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("PLANT ELECTROPHYSIOLOGY (Phase 2 Domain Adaptation)")
    print(f"Source: PMC10950275 supplementary data")
    print(f"Output: {os.path.abspath(output_dir)}")
    print("=" * 60)

    # Check if already downloaded (look for .wav files)
    existing_wav = [f for f in os.listdir(output_dir) if f.endswith('.wav')] if os.path.exists(output_dir) else []
    if existing_wav:
        print(f"\n[SKIP] Plant data already downloaded ({len(existing_wav)} .wav files)")
        return 0

    # Try gdown
    try:
        import gdown  # noqa: F401
        has_gdown = True
    except ImportError:
        has_gdown = False

    if has_gdown:
        print("\nUsing gdown to download Google Drive folder...")
        try:
            import gdown
            gdown.download_folder(
                url=PLANT_GDRIVE_URL,
                output=output_dir,
                quiet=False,
            )
            # Check results
            files = os.listdir(output_dir)
            wav_files = [f for f in files if f.endswith('.wav')]
            txt_files = [f for f in files if f.endswith('.txt')]
            print(f"\n  Downloaded {len(wav_files)} .wav files, {len(txt_files)} .txt files")
            return len(wav_files)
        except Exception as e:
            print(f"  gdown error: {e}")
            print("  Falling back to manual instructions...")

    # Fallback: manual instructions
    print("\n" + "-" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("-" * 60)
    if not has_gdown:
        print("gdown is not installed. Install it with:")
        print("  pip install gdown")
        print()
    print("Or download manually from Google Drive:")
    print(f"  {PLANT_GDRIVE_URL}")
    print(f"  Save .wav and .txt files to:")
    print(f"    {os.path.abspath(output_dir)}/")
    print("-" * 60)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Download pre-training datasets for TCN transfer learning'
    )
    parser.add_argument('--ecg', action='store_true',
                        help='Download ECG heartbeat dataset only')
    parser.add_argument('--plant', action='store_true',
                        help='Download plant electrophysiology dataset only')
    parser.add_argument('--ecg-dir', type=str, default=DEFAULT_ECG_DIR,
                        help='Output directory for ECG dataset')
    parser.add_argument('--plant-dir', type=str, default=DEFAULT_PLANT_DIR,
                        help='Output directory for plant dataset')

    args = parser.parse_args()

    # If neither flag set, download both
    do_ecg = args.ecg or (not args.ecg and not args.plant)
    do_plant = args.plant or (not args.ecg and not args.plant)

    total = 0
    if do_ecg:
        total += download_ecg(args.ecg_dir)
    if do_plant:
        total += download_plant(args.plant_dir)

    print(f"\n{'=' * 60}")
    print(f"All done! Total files downloaded: {total}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
