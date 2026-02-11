#!/usr/bin/env python3
"""
analyze_recording.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Offline analysis of recorded fungal bioelectrical signals.
Generates comprehensive plots and statistics.

Usage:
    python analyze_recording.py recording.csv
    python analyze_recording.py recording.csv --output-dir ./results
    python analyze_recording.py recording.csv --sample-rate 10

Requirements:
    pip install numpy scipy matplotlib pandas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, stft, find_peaks, welch
from scipy.stats import skew, kurtosis
import argparse
import os
import csv
from datetime import datetime


def load_recording(filepath):
    """Load recording from CSV file."""
    times = []
    pwm = []
    adc_raw = []
    voltages = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)

        for row in reader:
            # Skip comments and headers
            if not row or row[0].startswith('#') or row[0].startswith('timestamp'):
                continue

            try:
                if len(row) >= 4:
                    times.append(float(row[0]))
                    pwm.append(int(row[1]))
                    adc_raw.append(int(row[2]))
                    voltages.append(float(row[3]))
                elif len(row) >= 3:
                    times.append(float(row[0]))
                    adc_raw.append(int(row[1]))
                    voltages.append(float(row[2]))
            except (ValueError, IndexError):
                continue

    return {
        'time_ms': np.array(times),
        'time_s': np.array(times) / 1000.0,
        'pwm': np.array(pwm) if pwm else None,
        'adc_raw': np.array(adc_raw),
        'voltage_mV': np.array(voltages)
    }


def compute_filters(fs):
    """Compute filter coefficients."""
    nyq = fs / 2

    # Bandpass (0.01-2 Hz)
    bp_low = max(0.001, 0.01 / nyq)
    bp_high = min(0.99, 2.0 / nyq)
    b_bp, a_bp = butter(2, [bp_low, bp_high], btype='band')

    # High-pass (0.01 Hz)
    hp_norm = max(0.001, 0.01 / nyq)
    b_hp, a_hp = butter(2, hp_norm, btype='high')

    # Low-pass (2 Hz)
    lp_norm = min(0.99, 2.0 / nyq)
    b_lp, a_lp = butter(2, lp_norm, btype='low')

    return {
        'bandpass': (b_bp, a_bp),
        'highpass': (b_hp, a_hp),
        'lowpass': (b_lp, a_lp)
    }


def analyze_signal(data, fs):
    """Perform comprehensive signal analysis."""
    voltage = data['voltage_mV']
    time_s = data['time_s']

    # Get filters
    filters = compute_filters(fs)

    # Apply filtering
    voltage_centered = voltage - np.median(voltage)
    if len(voltage) >= 15:
        voltage_filtered = filtfilt(*filters['bandpass'], voltage_centered)
    else:
        voltage_filtered = voltage_centered

    # Basic statistics
    stats = {
        'n_samples': len(voltage),
        'duration_s': time_s[-1] - time_s[0] if len(time_s) > 1 else 0,
        'sample_rate_actual': len(voltage) / (time_s[-1] - time_s[0]) if len(time_s) > 1 else fs,

        'raw_mean': np.mean(voltage),
        'raw_std': np.std(voltage),
        'raw_min': np.min(voltage),
        'raw_max': np.max(voltage),
        'raw_range': np.max(voltage) - np.min(voltage),
        'raw_skewness': skew(voltage),
        'raw_kurtosis': kurtosis(voltage),

        'filtered_mean': np.mean(voltage_filtered),
        'filtered_std': np.std(voltage_filtered),
        'filtered_min': np.min(voltage_filtered),
        'filtered_max': np.max(voltage_filtered),
    }

    # Spike detection
    baseline = np.median(voltage_filtered)
    threshold = baseline + 3 * np.std(voltage_filtered)
    peaks, properties = find_peaks(voltage_filtered, height=threshold, distance=5)

    stats['n_spikes'] = len(peaks)
    stats['spike_rate_per_min'] = len(peaks) / (stats['duration_s'] / 60) if stats['duration_s'] > 0 else 0
    stats['spike_threshold'] = threshold

    if len(peaks) > 0:
        stats['spike_amplitudes'] = voltage_filtered[peaks]
        stats['spike_mean_amplitude'] = np.mean(voltage_filtered[peaks])
        stats['spike_std_amplitude'] = np.std(voltage_filtered[peaks])

    # STFT analysis
    if len(voltage) >= 64:
        nperseg = min(64, len(voltage) // 4)
        noverlap = nperseg * 3 // 4

        f_stft, t_stft, Zxx = stft(
            voltage_filtered,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=128
        )
        stats['stft'] = {
            'frequencies': f_stft,
            'times': t_stft,
            'magnitude': np.abs(Zxx)
        }

    # Power spectral density
    if len(voltage) >= 64:
        f_psd, psd = welch(voltage_filtered, fs=fs, nperseg=min(64, len(voltage)//4))
        stats['psd'] = {
            'frequencies': f_psd,
            'power': psd
        }

        # Dominant frequency
        peak_idx = np.argmax(psd)
        stats['dominant_frequency'] = f_psd[peak_idx]
        stats['dominant_power'] = psd[peak_idx]

        # Band powers
        bands = {
            'ultra_low': (0.01, 0.05),
            'low': (0.05, 0.2),
            'mid': (0.2, 0.5),
            'high': (0.5, 2.0)
        }
        stats['band_powers'] = {}
        for band_name, (f_low, f_high) in bands.items():
            mask = (f_psd >= f_low) & (f_psd < f_high)
            stats['band_powers'][band_name] = np.sum(psd[mask]) if np.any(mask) else 0

    return stats, voltage_filtered, peaks


def create_analysis_plots(data, stats, voltage_filtered, peaks, fs, output_dir=None):
    """Create comprehensive analysis plots."""

    voltage = data['voltage_mV']
    time_s = data['time_s'] - data['time_s'][0]  # Normalize to start at 0

    # ==================== Figure 1: Signal Overview ====================
    fig1, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig1.suptitle('Signal Analysis Overview', fontsize=14)

    # Raw signal
    axes[0].plot(time_s, voltage, 'b-', linewidth=0.5, alpha=0.8)
    axes[0].axhline(y=stats['raw_mean'], color='red', linestyle='--', alpha=0.5, label=f"Mean: {stats['raw_mean']:.3f}")
    axes[0].fill_between(time_s,
                          stats['raw_mean'] - stats['raw_std'],
                          stats['raw_mean'] + stats['raw_std'],
                          alpha=0.2, color='red', label=f"±1 Std: {stats['raw_std']:.3f}")
    axes[0].set_ylabel('Voltage (mV)')
    axes[0].set_title('Raw Signal')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Filtered signal with spikes
    axes[1].plot(time_s, voltage_filtered, 'g-', linewidth=0.5, alpha=0.8)
    if len(peaks) > 0:
        axes[1].scatter(time_s[peaks], voltage_filtered[peaks],
                       c='red', s=50, zorder=5, label=f'Spikes (n={len(peaks)})')
    axes[1].axhline(y=stats['spike_threshold'], color='red', linestyle='--',
                    alpha=0.5, label=f"Threshold: {stats['spike_threshold']:.3f}")
    axes[1].set_ylabel('Voltage (mV)')
    axes[1].set_title('Filtered Signal (0.01-2 Hz Bandpass) + Spike Detection')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Histogram
    axes[2].hist(voltage_filtered, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
    axes[2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(x=stats['spike_threshold'], color='red', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Voltage (mV)')
    axes[2].set_ylabel('Probability Density')
    axes[2].set_title('Amplitude Distribution')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # ==================== Figure 2: Frequency Analysis ====================
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Frequency Domain Analysis', fontsize=14)

    # STFT Spectrogram
    if 'stft' in stats:
        stft_data = stats['stft']
        im = axes[0, 0].pcolormesh(stft_data['times'], stft_data['frequencies'],
                                    stft_data['magnitude'], shading='gouraud', cmap='viridis')
        axes[0, 0].set_ylim(0, 2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Frequency (Hz)')
        axes[0, 0].set_title('STFT Spectrogram')
        plt.colorbar(im, ax=axes[0, 0], label='Magnitude')
    else:
        axes[0, 0].text(0.5, 0.5, 'Insufficient data for STFT',
                        ha='center', va='center', transform=axes[0, 0].transAxes)

    # Power Spectral Density
    if 'psd' in stats:
        psd_data = stats['psd']
        axes[0, 1].semilogy(psd_data['frequencies'], psd_data['power'], 'b-', linewidth=1)
        axes[0, 1].axvline(x=stats['dominant_frequency'], color='red', linestyle='--',
                           alpha=0.5, label=f"Peak: {stats['dominant_frequency']:.3f} Hz")
        axes[0, 1].set_xlim(0, 2)
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power Spectral Density')
        axes[0, 1].set_title('Power Spectrum')
        axes[0, 1].legend(loc='upper right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # Frequency bands
        band_names = list(stats['band_powers'].keys())
        band_values = [stats['band_powers'][b] for b in band_names]
        colors = ['red', 'orange', 'green', 'blue']
        bars = axes[1, 0].bar(band_names, band_values, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Frequency Band')
        axes[1, 0].set_ylabel('Total Power')
        axes[1, 0].set_title('Power by Frequency Band')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, band_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                           f'{val:.2e}', ha='center', va='bottom', fontsize=8)

    # Statistics text
    axes[1, 1].axis('off')
    stats_text = f"""
RECORDING STATISTICS
════════════════════════════════════════

Duration:        {stats['duration_s']:.1f} seconds
Samples:         {stats['n_samples']}
Sample Rate:     {stats['sample_rate_actual']:.2f} Hz

RAW SIGNAL
────────────────────────────────────────
Mean:            {stats['raw_mean']:.4f} mV
Std Dev:         {stats['raw_std']:.4f} mV
Min:             {stats['raw_min']:.4f} mV
Max:             {stats['raw_max']:.4f} mV
Range:           {stats['raw_range']:.4f} mV
Skewness:        {stats['raw_skewness']:.4f}
Kurtosis:        {stats['raw_kurtosis']:.4f}

SPIKE DETECTION
────────────────────────────────────────
Threshold:       {stats['spike_threshold']:.4f} mV
Spikes Found:    {stats['n_spikes']}
Spike Rate:      {stats['spike_rate_per_min']:.2f} /min
"""
    if 'spike_mean_amplitude' in stats:
        stats_text += f"""Mean Amplitude:  {stats['spike_mean_amplitude']:.4f} mV
"""

    if 'dominant_frequency' in stats:
        stats_text += f"""
FREQUENCY ANALYSIS
────────────────────────────────────────
Dominant Freq:   {stats['dominant_frequency']:.4f} Hz
Peak Power:      {stats['dominant_power']:.4e}
"""

    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', family='monospace')

    plt.tight_layout()

    # Save if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig1.savefig(os.path.join(output_dir, 'signal_overview.png'), dpi=150)
        fig2.savefig(os.path.join(output_dir, 'frequency_analysis.png'), dpi=150)
        print(f"Plots saved to {output_dir}/")

    return fig1, fig2


def main():
    parser = argparse.ArgumentParser(
        description='Analyze recorded fungal bioelectrical signals'
    )
    parser.add_argument('input', type=str, help='Input CSV file')
    parser.add_argument('--output-dir', '-o', type=str, default=None,
                        help='Output directory for plots and results')
    parser.add_argument('--sample-rate', '-s', type=float, default=10.0,
                        help='Sample rate in Hz (default: 10)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (just save)')

    args = parser.parse_args()

    # Load data
    print(f"Loading {args.input}...")
    data = load_recording(args.input)
    print(f"Loaded {len(data['voltage_mV'])} samples")

    # Analyze
    print("Analyzing signal...")
    stats, voltage_filtered, peaks = analyze_signal(data, args.sample_rate)

    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Duration: {stats['duration_s']:.1f} seconds")
    print(f"Samples: {stats['n_samples']}")
    print(f"Actual sample rate: {stats['sample_rate_actual']:.2f} Hz")
    print(f"\nRaw Signal:")
    print(f"  Mean: {stats['raw_mean']:.4f} mV")
    print(f"  Std:  {stats['raw_std']:.4f} mV")
    print(f"  Range: {stats['raw_min']:.4f} - {stats['raw_max']:.4f} mV")
    print(f"\nSpike Detection:")
    print(f"  Threshold: {stats['spike_threshold']:.4f} mV")
    print(f"  Spikes found: {stats['n_spikes']}")
    print(f"  Rate: {stats['spike_rate_per_min']:.2f} per minute")

    if 'dominant_frequency' in stats:
        print(f"\nFrequency Analysis:")
        print(f"  Dominant frequency: {stats['dominant_frequency']:.4f} Hz")
        print(f"  Band powers:")
        for band, power in stats['band_powers'].items():
            print(f"    {band}: {power:.4e}")

    # Create plots
    print("\nGenerating plots...")
    fig1, fig2 = create_analysis_plots(
        data, stats, voltage_filtered, peaks,
        args.sample_rate, args.output_dir
    )

    # Save statistics to file
    if args.output_dir:
        stats_file = os.path.join(args.output_dir, 'statistics.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Analysis of: {args.input}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")

            for key, value in stats.items():
                if not isinstance(value, (dict, np.ndarray)):
                    f.write(f"{key}: {value}\n")

        print(f"Statistics saved to {stats_file}")

    if not args.no_show:
        plt.show()

    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
