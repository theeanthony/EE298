#!/usr/bin/env python3
"""
synthetic_data.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Python port of the Arduino MYCELIUM signal simulator + baseline generators.
Generates labeled CSV files for ML training.

Usage:
    python synthetic_data.py                    # Generate default dataset
    python synthetic_data.py --num-each 20      # 20 recordings per class
    python synthetic_data.py --duration 120     # 120-second recordings
"""

import numpy as np
import csv
import os
import argparse

# Default output directory
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'synthetic'
)

SAMPLE_RATE = 10.0  # Hz, matches Arduino


def generate_mycelium(duration_sec: float = 60.0, sample_rate: float = SAMPLE_RATE) -> np.ndarray:
    """
    Generate realistic mycelium bioelectrical signal.

    Python port of generateMycelium() from mycelium_signal_simulator.ino.
    Produces voltage in mV (post-amplification scale).

    Components:
      1. Three overlapping slow oscillations (0.01, 0.05, 0.2 Hz)
      2. Wandering baseline
      3. Action potential trains (fast rise, plateau, slow decay)
      4. Correlated biological noise
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # 1. Slow oscillations (frequencies from literature: 0.01-1 Hz)
    wave1 = 0.6 * np.sin(2 * np.pi * 0.01 * t + np.random.uniform(0, 2 * np.pi))
    wave2 = 0.4 * np.sin(2 * np.pi * 0.05 * t + np.random.uniform(0, 2 * np.pi))
    wave3 = 0.3 * np.sin(2 * np.pi * 0.2 * t + np.random.uniform(0, 2 * np.pi))

    # 2. Wandering baseline (slow random walk)
    baseline = np.zeros(n_samples)
    baseline[0] = np.random.uniform(-0.2, 0.2)
    for i in range(1, n_samples):
        if np.random.random() < 0.03:  # 3% chance of step (matches Arduino)
            baseline[i] = baseline[i - 1] + np.random.uniform(-0.04, 0.04)
        else:
            baseline[i] = baseline[i - 1]
    baseline = np.clip(baseline, -0.5, 0.5)

    # 3. Action potentials (irregular spiking trains)
    action_potentials = np.zeros(n_samples)
    i = 0
    while i < n_samples:
        # Wait for next AP (20-60 sec interval, scaled from Arduino)
        wait = int(np.random.uniform(20, 60) * sample_rate)
        i += wait
        if i >= n_samples:
            break

        # AP shape: fast rise (~4 samples), plateau (~30 samples), slow decay (~16 samples)
        ap_duration = 50  # ~5 sec at 10 Hz (matches Arduino's 50-sample AP)
        rise = 10
        plateau = 30
        decay = 10

        for j in range(ap_duration):
            idx = i + j
            if idx >= n_samples:
                break
            if j < rise:
                action_potentials[idx] = 1.5 * (j / rise)  # Fast rise to ~1.5 mV
            elif j < rise + plateau:
                action_potentials[idx] = 1.5 + np.random.uniform(-0.1, 0.1)  # Plateau with jitter
            else:
                frac = (j - rise - plateau) / decay
                action_potentials[idx] = 1.5 * (1 - frac)  # Slow decay

        i += ap_duration

    # 4. Biological noise (slightly correlated, low amplitude)
    bio_noise = np.zeros(n_samples)
    bio_noise[0] = np.random.uniform(-0.05, 0.05)
    for i in range(1, n_samples):
        bio_noise[i] = 0.9 * bio_noise[i - 1] + np.random.uniform(-0.06, 0.06)
    bio_noise = np.clip(bio_noise, -0.3, 0.3)

    # Combine all components
    signal = baseline + wave1 + wave2 + wave3 + action_potentials + bio_noise

    # Center around a realistic mean (~1.0 mV, typical post-INA128 offset)
    signal += 1.0

    return signal


def generate_nothing(duration_sec: float = 60.0, sample_rate: float = SAMPLE_RATE) -> np.ndarray:
    """
    Generate flat baseline signal (no biological activity).

    Mirrors Arduino's generateNothing(): flat around a small value + tiny noise.
    """
    n_samples = int(duration_sec * sample_rate)
    # Flat at ~0.5 mV with very small noise (mimics electrode noise floor)
    signal = 0.5 + np.random.normal(0, 0.02, n_samples)
    return signal


def generate_noise(duration_sec: float = 60.0, sample_rate: float = SAMPLE_RATE) -> np.ndarray:
    """
    Generate pure random noise (electrical interference).

    Mirrors Arduino's generateNoise(): full random range.
    """
    n_samples = int(duration_sec * sample_rate)
    # Random noise across voltage range (scaled to mV)
    signal = np.random.uniform(0, 3.0, n_samples)
    return signal


def generate_drift(duration_sec: float = 60.0, sample_rate: float = SAMPLE_RATE) -> np.ndarray:
    """
    Generate slowly drifting signal (electrode drift / temperature).

    Mirrors Arduino's generateDrift(): slow upward trend + small noise.
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Linear drift from ~0.3 to ~2.0 mV over duration
    drift = 0.3 + (1.7 / duration_sec) * t
    # Add small noise
    noise = np.random.normal(0, 0.1, n_samples)
    signal = drift + noise
    return signal


def save_recording(signal: np.ndarray, filepath: str, sample_rate: float = SAMPLE_RATE):
    """
    Save a synthetic signal as CSV in our standard format.

    Format: timestamp_ms, adc_raw, voltage_mV
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_ms', 'adc_raw', 'voltage_mV'])

        for i, v in enumerate(signal):
            t_ms = int(i / sample_rate * 1000)
            # Simulate ADC raw value (14-bit, 5V ref): adc = v_mV / 5000 * 16383
            adc_raw = int(np.clip(v / 5000.0 * 16383, 0, 16383))
            writer.writerow([t_ms, adc_raw, f'{v:.4f}'])


def generate_dataset(output_dir: str = DEFAULT_OUTPUT_DIR,
                     num_each: int = 10,
                     duration_sec: float = 60.0):
    """
    Generate a complete synthetic dataset for training.

    Creates:
      - num_each MYCELIUM recordings (label=1)
      - num_each NOTHING recordings (label=0)
      - num_each NOISE recordings (label=0)
      - num_each DRIFT recordings (label=0)

    Also writes a manifest CSV listing all files with labels.
    """
    os.makedirs(output_dir, exist_ok=True)

    manifest = []  # (filepath, label, mode)

    generators = {
        'mycelium': (generate_mycelium, 1),
        'nothing': (generate_nothing, 0),
        'noise': (generate_noise, 0),
        'drift': (generate_drift, 0),
    }

    for mode_name, (gen_func, label) in generators.items():
        print(f"Generating {num_each} {mode_name} recordings (label={label})...")
        for i in range(num_each):
            signal = gen_func(duration_sec)
            filename = f'{mode_name}_{i:03d}.csv'
            filepath = os.path.join(output_dir, filename)
            save_recording(signal, filepath)
            manifest.append((filename, label, mode_name))

    # Write manifest
    manifest_path = os.path.join(output_dir, 'manifest.csv')
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label', 'mode'])
        for row in manifest:
            writer.writerow(row)

    total = len(manifest)
    active = sum(1 for _, l, _ in manifest if l == 1)
    print(f"\nGenerated {total} recordings ({active} active, {total - active} inactive)")
    print(f"Manifest: {manifest_path}")
    print(f"Output dir: {output_dir}")

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic fungal signal data for ML training'
    )
    parser.add_argument('--output-dir', '-o', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for CSVs')
    parser.add_argument('--num-each', '-n', type=int, default=10,
                        help='Number of recordings per class (default: 10)')
    parser.add_argument('--duration', '-d', type=float, default=60.0,
                        help='Duration of each recording in seconds (default: 60)')

    args = parser.parse_args()
    generate_dataset(args.output_dir, args.num_each, args.duration)


if __name__ == '__main__':
    main()
