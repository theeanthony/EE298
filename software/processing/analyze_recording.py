#!/usr/bin/env python3
"""
analyze_recording.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Offline analysis of recorded fungal bioelectrical signals.
Supports 14-column 4-channel format from long_duration_logger.py.

CSV columns (14):
  seq, wall_clock_utc, timestamp_ms,
  adc_p2, v_p2_mV, adc_p3, v_p3_mV,
  adc_p6_ctrl, v_p6_mV, adc_p7_ctrl, v_p7_mV,
  mister, fan, led

Usage:
    # Single file:
    python analyze_recording.py recording_20260312.csv

    # Multi-day (concatenates in order):
    python analyze_recording.py data/raw/recording_2026030*.csv

    # Save plots to folder:
    python analyze_recording.py recording_*.csv --output-dir ./results

    # Restrict to inoculation window only:
    python analyze_recording.py recording_*.csv --start-date 2026-03-05

Requirements:
    pip install numpy scipy matplotlib pandas
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless-safe; swap to 'TkAgg' if you want interactive window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import butter, filtfilt, welch, find_peaks
import argparse
import os
import sys
import csv
from datetime import datetime, timezone, timedelta


# ── Channel definitions ───────────────────────────────────────────────────────
CHANNELS = [
    {'key': 'v_p2',   'label': 'Pair 2 (inoculated)', 'color': '#2196F3'},
    {'key': 'v_p3',   'label': 'Pair 3 (inoculated)', 'color': '#4CAF50'},
    {'key': 'v_p6',   'label': 'Pair 6 (control)',    'color': '#FF9800'},
    {'key': 'v_p7',   'label': 'Pair 7 (control)',    'color': '#9C27B0'},
]


# ── CSV loader ────────────────────────────────────────────────────────────────
def load_files(paths, start_date=None, end_date=None):
    """Load one or more 14-column CSV files and return a combined dict of arrays."""
    records = []

    for path in sorted(paths):
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if parts[0] == 'seq':   # header row
                    continue
                if len(parts) < 11:
                    continue
                try:
                    wall_utc = datetime.fromisoformat(parts[1])
                    if start_date and wall_utc < start_date:
                        continue
                    if end_date and wall_utc > end_date:
                        continue
                    records.append({
                        'seq':   int(parts[0]),
                        'wall':  wall_utc,
                        't_ms':  int(parts[2]),
                        'v_p2':  float(parts[4]),
                        'v_p3':  float(parts[6]),
                        'v_p6':  float(parts[8]),
                        'v_p7':  float(parts[10]),
                        'mister': int(parts[11]),
                        'fan':   int(parts[12]),
                        'led':   int(parts[13]) if len(parts) > 13 else 0,
                    })
                except (ValueError, IndexError):
                    continue

    if not records:
        print("ERROR: no valid data rows found. Check file format.")
        sys.exit(1)

    print(f"Loaded {len(records):,} samples from {len(paths)} file(s)")
    wall  = np.array([r['wall']  for r in records])
    t_ms  = np.array([r['t_ms'] for r in records], dtype=float)
    data  = {ch['key']: np.array([r[ch['key']] for r in records]) for ch in CHANNELS}
    data['wall']   = wall
    data['t_ms']   = t_ms
    data['mister'] = np.array([r['mister'] for r in records])
    data['fan']    = np.array([r['fan']    for r in records])
    data['led']    = np.array([r['led']    for r in records])
    return data


# ── Signal processing ─────────────────────────────────────────────────────────
def bandpass(signal, fs=10.0, low=0.01, high=2.0, order=2):
    nyq = fs / 2
    b, a = butter(order, [max(0.001, low/nyq), min(0.999, high/nyq)], btype='band')
    return filtfilt(b, a, signal)


def downsample(arr, factor):
    """Simple block-average downsample for plotting large arrays."""
    n = (len(arr) // factor) * factor
    return arr[:n].reshape(-1, factor).mean(axis=1)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_overview(data, output_dir, inoculation_utc=None):
    """4-channel voltage over full recording period."""
    wall = data['wall']
    # Downsample for plotting (keep ~50k points max)
    n = len(wall)
    ds = max(1, n // 50000)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('Fungal Signal Recording — All Channels', fontsize=13)

    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        v_ds   = downsample(data[ch['key']], ds)
        w_ds   = wall[::ds][:len(v_ds)]
        ax.plot(w_ds, v_ds, linewidth=0.4, color=ch['color'], alpha=0.8)
        ax.set_ylabel('mV', fontsize=9)
        ax.set_title(ch['label'], fontsize=10)
        ax.set_ylim(0, 5000)
        ax.axhline(2378, color='gray', linestyle=':', linewidth=0.6, alpha=0.5,
                   label='baseline 2378 mV')
        ax.grid(True, alpha=0.2)
        if inoculation_utc:
            ax.axvline(inoculation_utc, color='red', linestyle='--', linewidth=1,
                       alpha=0.7, label='Day 0 inoculation')
        ax.legend(fontsize=7, loc='upper right')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    path = os.path.join(output_dir, '01_overview_all_channels.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_inoculated_comparison(data, output_dir, inoculation_utc=None):
    """Pair 2 vs Pair 3 side-by-side — the key comparison."""
    wall = data['wall']
    n = len(wall)
    ds = max(1, n // 50000)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle('Inoculated Pairs — Signal Comparison', fontsize=13)

    for ax, key, label, color in [
        (ax1, 'v_p2', 'Pair 2 (one-side LC inoculation)', '#2196F3'),
        (ax2, 'v_p3', 'Pair 3 (both-side LC inoculation)', '#4CAF50'),
    ]:
        v_ds = downsample(data[key], ds)
        w_ds = wall[::ds][:len(v_ds)]
        ax.plot(w_ds, v_ds, linewidth=0.4, color=color)
        ax.set_ylabel('Voltage (mV)', fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 5000)
        ax.axhline(2378, color='gray', linestyle=':', linewidth=0.8, alpha=0.6,
                   label='baseline')
        ax.grid(True, alpha=0.2)
        if inoculation_utc:
            ax.axvline(inoculation_utc, color='red', linestyle='--', linewidth=1,
                       alpha=0.7, label='Day 0')
        ax.legend(fontsize=8, loc='upper right')

    axes_list = [ax1, ax2]
    axes_list[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(axes_list[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    path = os.path.join(output_dir, '02_inoculated_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_filtered_signals(data, output_dir, fs=10.0):
    """Bandpass-filtered pair 2 and pair 3 — removes DC offset to show oscillations."""
    wall = data['wall']
    n = len(wall)

    # Only filter if long enough
    if n < 100:
        print("Not enough samples for filtered plot, skipping.")
        return

    ds = max(1, n // 100000)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle('Bandpass Filtered Signals (0.01–2 Hz) — DC Removed', fontsize=13)

    for i, (key, label, color) in enumerate([
        ('v_p2', 'Pair 2 (inoculated)', '#2196F3'),
        ('v_p3', 'Pair 3 (inoculated)', '#4CAF50'),
    ]):
        try:
            v_filt = bandpass(data[key], fs=fs)
        except Exception as e:
            print(f"  Filter failed for {key}: {e}")
            v_filt = data[key] - np.mean(data[key])

        v_ds   = downsample(v_filt, ds)
        w_ds   = wall[::ds][:len(v_ds)]
        std    = np.std(v_filt)

        axes[i].plot(w_ds, v_ds, linewidth=0.4, color=color)
        axes[i].axhline(0, color='gray', linestyle=':', linewidth=0.6)
        axes[i].axhline( 3*std, color='red', linestyle='--', linewidth=0.7,
                         alpha=0.6, label=f'+3σ = {3*std:.2f} mV')
        axes[i].axhline(-3*std, color='red', linestyle='--', linewidth=0.7, alpha=0.6)
        axes[i].set_ylabel('Filtered mV', fontsize=9)
        axes[i].set_title(label, fontsize=10)
        axes[i].grid(True, alpha=0.2)
        axes[i].legend(fontsize=8, loc='upper right')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    path = os.path.join(output_dir, '03_filtered_signals.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_psd(data, output_dir, fs=10.0):
    """Power spectral density for pair 2 vs pair 3."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Power Spectral Density (0–2 Hz)', fontsize=13)

    for ax, key, label, color in [
        (axes[0], 'v_p2', 'Pair 2 (inoculated)', '#2196F3'),
        (axes[1], 'v_p3', 'Pair 3 (inoculated)', '#4CAF50'),
    ]:
        try:
            v = bandpass(data[key], fs=fs)
        except Exception:
            v = data[key] - np.mean(data[key])
        nperseg = min(1024, len(v) // 4)
        f, psd = welch(v, fs=fs, nperseg=nperseg)
        mask = f <= 2.0
        ax.semilogy(f[mask], psd[mask], color=color, linewidth=1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (mV²/Hz)')
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        if len(psd[mask]) > 0:
            peak_f = f[mask][np.argmax(psd[mask])]
            ax.axvline(peak_f, color='red', linestyle='--', linewidth=0.8,
                       alpha=0.7, label=f'peak {peak_f:.4f} Hz')
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, '04_power_spectral_density.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_zoom_window(data, output_dir, start_utc=None, hours=12.0, fs=10.0):
    """Zoomed 4-panel view: raw + filtered for pair 2 and pair 3.

    Useful for inspecting slow-wave structure, micro-oscillations, and
    whether pair 2 activity has temporal coherence (biological) vs
    random amplitude drift (electrochemical artifact).

    If start_utc is None, defaults to the last `hours` of the recording.
    """
    wall = data['wall']

    if start_utc is None:
        start_utc = wall[-1] - timedelta(hours=hours)
    end_utc = start_utc + timedelta(hours=hours)

    mask = (wall >= start_utc) & (wall <= end_utc)
    n = int(np.sum(mask))

    if n < 10:
        print(f"  Zoom window has only {n} samples — skipping.")
        return

    print(f"  Zoom: {start_utc.strftime('%Y-%m-%d %H:%M UTC')} → "
          f"{end_utc.strftime('%Y-%m-%d %H:%M UTC')} ({n:,} samples)")

    w_zoom = wall[mask]
    pairs = [
        ('v_p2', 'Pair 2 (inoculated)', '#2196F3'),
        ('v_p3', 'Pair 3 (inoculated)', '#4CAF50'),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(
        f'Zoomed Window — {start_utc.strftime("%Y-%m-%d %H:%M")} UTC + {hours:.0f}h'
        f'  ({n:,} samples)',
        fontsize=13,
    )

    ax_idx = 0
    for key, label, color in pairs:
        raw = data[key]

        # Raw voltage panel
        axes[ax_idx].plot(w_zoom, raw[mask], linewidth=0.6, color=color)
        axes[ax_idx].set_ylabel('mV', fontsize=9)
        axes[ax_idx].set_title(f'{label} — Raw', fontsize=10)
        axes[ax_idx].grid(True, alpha=0.2)
        ax_idx += 1

        # Bandpass-filtered panel — computed on full array to avoid edge effects
        std = 0.0
        v_filt_zoom = raw[mask] - np.mean(raw[mask])   # fallback: DC-subtract
        try:
            v_filt = bandpass(raw, fs=fs)
            v_filt_zoom = v_filt[mask]
            std = np.std(v_filt_zoom)
        except Exception as e:
            print(f'  Filter fallback for {key}: {e}')

        axes[ax_idx].plot(w_zoom, v_filt_zoom, linewidth=0.6, color=color, alpha=0.85)
        axes[ax_idx].axhline(0, color='gray', linestyle=':', linewidth=0.5)
        if std > 0:
            axes[ax_idx].axhline(
                3 * std, color='red', linestyle='--', linewidth=0.8,
                alpha=0.6, label=f'+3σ = {3 * std:.1f} mV',
            )
            axes[ax_idx].axhline(-3 * std, color='red', linestyle='--',
                                 linewidth=0.8, alpha=0.6)
            axes[ax_idx].legend(fontsize=8, loc='upper right')
        axes[ax_idx].set_ylabel('Filtered mV', fontsize=9)
        axes[ax_idx].set_title(
            f'{label} — Bandpass 0.01–2 Hz  (3σ = {3 * std:.1f} mV)', fontsize=10
        )
        axes[ax_idx].grid(True, alpha=0.2)
        ax_idx += 1

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()

    zoom_tag = start_utc.strftime('%Y%m%d_%H%M')
    path = os.path.join(output_dir, f'05_zoom_{zoom_tag}.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary(data, fs=10.0):
    """Print key statistics to terminal."""
    n = len(data['wall'])
    duration_h = (data['wall'][-1] - data['wall'][0]).total_seconds() / 3600
    actual_rate = n / ((data['wall'][-1] - data['wall'][0]).total_seconds())

    print("\n" + "="*55)
    print("RECORDING SUMMARY")
    print("="*55)
    print(f"  Start:        {data['wall'][0].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  End:          {data['wall'][-1].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Duration:     {duration_h:.1f} hours ({duration_h/24:.1f} days)")
    print(f"  Samples:      {n:,}")
    print(f"  Actual rate:  {actual_rate:.2f} Hz")
    print()
    print(f"  {'Channel':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*57}")
    for ch in CHANNELS:
        v = data[ch['key']]
        print(f"  {ch['label']:<25} {np.mean(v):>8.1f} {np.std(v):>8.1f} "
              f"{np.min(v):>8.1f} {np.max(v):>8.1f}")
    print("="*55)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Analyze 4-channel fungal bioelectrical recordings'
    )
    parser.add_argument('inputs', nargs='+', help='CSV file(s) to analyze (supports glob)')
    parser.add_argument('--output-dir', '-o', default='./analysis_results',
                        help='Directory to save plots (default: ./analysis_results)')
    parser.add_argument('--sample-rate', '-s', type=float, default=10.0)
    parser.add_argument('--start-date', type=str, default=None,
                        help='Ignore data before this UTC date, e.g. 2026-03-05')
    parser.add_argument('--inoculation', type=str, default='2026-03-05T20:41:00+00:00',
                        help='ISO UTC datetime of Day 0 inoculation')
    parser.add_argument('--zoom-start', type=str, default=None,
                        help='Start of zoom window (ISO UTC datetime). '
                             'Default: last --zoom-hours of recording. '
                             'Example: 2026-03-10T00:00:00+00:00')
    parser.add_argument('--zoom-hours', type=float, default=12.0,
                        help='Width of zoom window in hours (default: 12.0)')
    args = parser.parse_args()

    start_dt = None
    if args.start_date:
        start_dt = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)

    inoc_dt = datetime.fromisoformat(args.inoculation)

    # Load
    data = load_files(args.inputs, start_date=start_dt)

    # Summary
    print_summary(data, fs=args.sample_rate)

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving plots to: {args.output_dir}/")

    # Plots
    plot_overview(data, args.output_dir, inoculation_utc=inoc_dt)
    plot_inoculated_comparison(data, args.output_dir, inoculation_utc=inoc_dt)
    plot_filtered_signals(data, args.output_dir, fs=args.sample_rate)
    plot_psd(data, args.output_dir, fs=args.sample_rate)

    # Zoom window (plot 05) — last 12h by default, or explicit --zoom-start
    zoom_start = None
    if args.zoom_start:
        zoom_start = datetime.fromisoformat(args.zoom_start)
        if zoom_start.tzinfo is None:
            zoom_start = zoom_start.replace(tzinfo=timezone.utc)
    plot_zoom_window(data, args.output_dir, start_utc=zoom_start,
                     hours=args.zoom_hours, fs=args.sample_rate)

    print("\nDone. Open the PNG files in your output directory.")


if __name__ == '__main__':
    main()
