#!/usr/bin/env python3
"""
download_data.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Downloads external fungal bioelectrical signal datasets:
  1. Adamatzky et al. - Zenodo (4 species, tab-delimited .txt in .zip)
  2. Buffi et al. 2025 - Mendeley Data (Fusarium oxysporum, HDF5)

Usage:
    python download_data.py                  # Download both datasets
    python download_data.py --adamatzky      # Adamatzky only
    python download_data.py --buffi          # Buffi only
"""

import os
import sys
import argparse
import urllib.request
import zipfile

# ============================================================
# ADAMATZKY DATASET (Zenodo)
# Record: https://zenodo.org/records/5790768
# Species: Schizophyllum commune, Cordyceps militaris,
#          Omphalotus nidiformis, Flammulina velutipes
# ============================================================
ZENODO_RECORD = '5790768'
ZENODO_API_BASE = f'https://zenodo.org/api/records/{ZENODO_RECORD}/files'

ADAMATZKY_FILES = [
    {
        'filename': 'Schizophyllum commune.txt.zip',
        'url': f'{ZENODO_API_BASE}/Schizophyllum%20commune.txt.zip/content',
        'species': 'schizophyllum_commune',
    },
    {
        'filename': 'Cordyceps militari.txt.zip',
        'url': f'{ZENODO_API_BASE}/Cordyceps%20militari.txt.zip/content',
        'species': 'cordyceps_militaris',
    },
    {
        'filename': 'Ghost Fungi Omphalotus nidiformis.txt.zip',
        'url': f'{ZENODO_API_BASE}/Ghost%20Fungi%20Omphalotus%20nidiformis.txt.zip/content',
        'species': 'omphalotus_nidiformis',
    },
    {
        'filename': 'Enoki fungi Flammulina velutipes.txt.zip',
        'url': f'{ZENODO_API_BASE}/Enoki%20fungi%20Flammulina%20velutipes.txt.zip/content',
        'species': 'flammulina_velutipes',
    },
]

# ============================================================
# BUFFI DATASET (Mendeley Data)
# DOI: 10.17632/srkxbkh6sp.1
# Species: Fusarium oxysporum (Neu 195)
# Format: HDF5 (from PicoLog 6), 8 differential channels, ~17 Hz
# Stimuli: calcimycin, cycloheximide, sodium azide, voriconazole
# ============================================================
BUFFI_FILES = [
    {
        'filename': 'calcimycin.hdf5',
        'url': 'https://data.mendeley.com/public-files/datasets/srkxbkh6sp/files/0dc860e5-cfa4-4a5f-89d8-65bd5436d18b/file_downloaded',
        'stimulus': 'calcimycin',
    },
    {
        'filename': 'cycloheximide.hdf5',
        'url': 'https://data.mendeley.com/public-files/datasets/srkxbkh6sp/files/b3850f94-183b-4301-b401-197cf80020cc/file_downloaded',
        'stimulus': 'cycloheximide',
    },
    {
        'filename': 'sodiumazide.hdf5',
        'url': 'https://data.mendeley.com/public-files/datasets/srkxbkh6sp/files/c102fd52-ed4c-4d7a-8a41-cb530b33ebb7/file_downloaded',
        'stimulus': 'sodium_azide',
    },
    {
        'filename': 'voriconazole.hdf5',
        'url': 'https://data.mendeley.com/public-files/datasets/srkxbkh6sp/files/e22699b3-c96f-4ba5-9996-8fda91c8f818/file_downloaded',
        'stimulus': 'voriconazole',
    },
]

DEFAULT_ADAMATZKY_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'external', 'adamatzky'
)
DEFAULT_BUFFI_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'external', 'buffi'
)


def download_file(url: str, dest_path: str):
    """Download a file with progress display."""
    print(f"  Downloading {os.path.basename(dest_path)}...")

    # Mendeley requires browser-like User-Agent, otherwise 403
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


def download_adamatzky(output_dir: str = DEFAULT_ADAMATZKY_DIR):
    """Download the Adamatzky fungal electrical activity dataset from Zenodo."""
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ADAMATZKY — ELECTRICAL ACTIVITY OF FUNGI")
    print(f"Zenodo record: {ZENODO_RECORD}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print("=" * 60)

    downloaded = 0
    skipped = 0

    for entry in ADAMATZKY_FILES:
        species_dir = os.path.join(output_dir, entry['species'])
        zip_path = os.path.join(output_dir, entry['filename'])

        # Check if already extracted
        if os.path.exists(species_dir) and os.listdir(species_dir):
            print(f"\n[SKIP] {entry['species']} already exists")
            skipped += 1
            continue

        os.makedirs(species_dir, exist_ok=True)
        print(f"\n[{entry['species']}]")

        try:
            download_file(entry['url'], zip_path)
            extract_zip(zip_path, species_dir)
            os.remove(zip_path)

            txt_files = [f for f in os.listdir(species_dir) if f.endswith('.txt')]
            print(f"  Extracted {len(txt_files)} .txt file(s)")
            downloaded += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if os.path.exists(species_dir) and not os.listdir(species_dir):
                os.rmdir(species_dir)

    print(f"\nAdamatzky: {downloaded} downloaded, {skipped} skipped")
    return downloaded


def download_buffi(output_dir: str = DEFAULT_BUFFI_DIR):
    """Download the Buffi et al. 2025 dataset from Mendeley Data."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("BUFFI ET AL. 2025 — FUNGAL MYCELIA ELECTRICAL SIGNALS")
    print("Mendeley Data DOI: 10.17632/srkxbkh6sp.1")
    print(f"Output: {os.path.abspath(output_dir)}")
    print("=" * 60)

    downloaded = 0
    skipped = 0

    for entry in BUFFI_FILES:
        dest_path = os.path.join(output_dir, entry['filename'])

        if os.path.exists(dest_path):
            print(f"\n[SKIP] {entry['filename']} already exists")
            skipped += 1
            continue

        print(f"\n[{entry['stimulus']}]")

        try:
            download_file(entry['url'], dest_path)
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"  Saved: {entry['filename']} ({size_mb:.1f} MB)")
            downloaded += 1

        except Exception as e:
            print(f"  ERROR: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)

    print(f"\nBuffi: {downloaded} downloaded, {skipped} skipped")

    # Print HDF5 usage hint
    if downloaded > 0:
        print("\nNote: Buffi data is HDF5 format. To inspect:")
        print("  pip install h5py")
        print("  python -c \"import h5py; f=h5py.File('calcimycin.hdf5','r'); print(list(f.keys()))\"")

    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description='Download external fungal bioelectrical signal datasets'
    )
    parser.add_argument('--adamatzky', action='store_true',
                        help='Download Adamatzky dataset only')
    parser.add_argument('--buffi', action='store_true',
                        help='Download Buffi dataset only')
    parser.add_argument('--adamatzky-dir', type=str, default=DEFAULT_ADAMATZKY_DIR,
                        help='Output directory for Adamatzky dataset')
    parser.add_argument('--buffi-dir', type=str, default=DEFAULT_BUFFI_DIR,
                        help='Output directory for Buffi dataset')

    args = parser.parse_args()

    # If neither flag set, download both
    do_adamatzky = args.adamatzky or (not args.adamatzky and not args.buffi)
    do_buffi = args.buffi or (not args.adamatzky and not args.buffi)

    total = 0
    if do_adamatzky:
        total += download_adamatzky(args.adamatzky_dir)
    if do_buffi:
        total += download_buffi(args.buffi_dir)

    print(f"\n{'=' * 60}")
    print(f"All done! Total files downloaded: {total}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
