#!/usr/bin/env python3
"""
signal_processor.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Signal processing pipeline for fungal bioelectrical signals.
Implements filters and STFT analysis based on Buffi et al. 2025 methodology.

Features:
    - 60 Hz notch filter (removes mains interference)
    - High-pass filter (removes DC drift, electrode offset)
    - Low-pass filter (anti-aliasing)
    - Bandpass filter (isolates fungal signal band)
    - STFT spectrogram analysis (Buffi et al. methodology)
    - Spike detection and characterization

Usage:
    # As a module
    from signal_processor import SignalProcessor
    processor = SignalProcessor(sample_rate=10)
    filtered = processor.process(raw_signal)

    # As a standalone script
    python signal_processor.py --input recording.csv --output processed.csv
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch, stft, find_peaks
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import warnings


@dataclass
class FilterConfig:
    """Configuration for signal filtering."""
    # Sampling
    sample_rate: float = 10.0  # Hz (matches Arduino code)

    # High-pass filter (removes DC drift)
    highpass_cutoff: float = 0.01  # Hz (very low to preserve slow oscillations)
    highpass_order: int = 2

    # Low-pass filter (anti-aliasing, noise reduction)
    lowpass_cutoff: float = 2.0  # Hz (fungal signals are < 1 Hz)
    lowpass_order: int = 2

    # Notch filter (60 Hz mains removal)
    notch_freq: float = 60.0  # Hz
    notch_quality: float = 30.0  # Q factor (higher = narrower notch)

    # Bandpass (alternative to separate HP/LP)
    bandpass_low: float = 0.01  # Hz
    bandpass_high: float = 2.0  # Hz
    bandpass_order: int = 2


@dataclass
class STFTConfig:
    """Configuration for STFT analysis (based on Buffi et al. 2025)."""
    # Window parameters
    window_type: str = 'hann'  # Hanning window (common for STFT)
    nperseg: int = 64  # Samples per segment
    noverlap: int = 48  # Overlap between segments (75%)
    nfft: int = 128  # FFT size (zero-padded for smoother spectrum)

    # Frequency range of interest
    freq_min: float = 0.0  # Hz
    freq_max: float = 2.0  # Hz

    # Power threshold for event detection
    power_threshold_db: float = -20.0  # dB below max


@dataclass
class SpikeConfig:
    """Configuration for spike detection."""
    threshold_std: float = 3.0  # Standard deviations above baseline
    min_distance_samples: int = 5  # Minimum samples between spikes
    min_prominence: float = 0.1  # mV minimum prominence


class SignalProcessor:
    """
    Complete signal processing pipeline for fungal bioelectrical signals.

    Based on methodology from:
    - Buffi et al. 2025 (iScience) - STFT analysis of fungal signals
    - Adamatzky et al. - Spike characterization in mycelium
    """

    def __init__(self,
                 sample_rate: float = 10.0,
                 filter_config: Optional[FilterConfig] = None,
                 stft_config: Optional[STFTConfig] = None,
                 spike_config: Optional[SpikeConfig] = None):
        """
        Initialize the signal processor.

        Args:
            sample_rate: Sampling rate in Hz
            filter_config: Filter configuration (uses defaults if None)
            stft_config: STFT configuration (uses defaults if None)
            spike_config: Spike detection configuration (uses defaults if None)
        """
        self.sample_rate = sample_rate
        self.filter_config = filter_config or FilterConfig(sample_rate=sample_rate)
        self.stft_config = stft_config or STFTConfig()
        self.spike_config = spike_config or SpikeConfig()

        # Pre-compute filter coefficients
        self._init_filters()

    def _init_filters(self):
        """Initialize filter coefficients."""
        fs = self.sample_rate
        nyq = fs / 2
        cfg = self.filter_config

        # High-pass filter coefficients
        if cfg.highpass_cutoff < nyq:
            hp_normalized = cfg.highpass_cutoff / nyq
            hp_normalized = max(0.001, min(hp_normalized, 0.99))
            self.b_hp, self.a_hp = butter(
                cfg.highpass_order,
                hp_normalized,
                btype='high'
            )
        else:
            self.b_hp, self.a_hp = [1], [1]

        # Low-pass filter coefficients
        if cfg.lowpass_cutoff < nyq:
            lp_normalized = cfg.lowpass_cutoff / nyq
            lp_normalized = max(0.001, min(lp_normalized, 0.99))
            self.b_lp, self.a_lp = butter(
                cfg.lowpass_order,
                lp_normalized,
                btype='low'
            )
        else:
            self.b_lp, self.a_lp = [1], [1]

        # Bandpass filter coefficients
        bp_low = max(0.001, cfg.bandpass_low / nyq)
        bp_high = min(0.99, cfg.bandpass_high / nyq)
        if bp_low < bp_high:
            self.b_bp, self.a_bp = butter(
                cfg.bandpass_order,
                [bp_low, bp_high],
                btype='band'
            )
        else:
            self.b_bp, self.a_bp = [1], [1]

        # Notch filter coefficients (60 Hz)
        # Note: At 10 Hz sample rate, we can't filter 60 Hz directly
        # This is included for when using higher sample rates
        if cfg.notch_freq < nyq:
            self.b_notch, self.a_notch = iirnotch(
                cfg.notch_freq,
                cfg.notch_quality,
                fs
            )
        else:
            # Can't filter frequencies above Nyquist
            self.b_notch, self.a_notch = [1], [1]

    # ==================== FILTERING ====================

    def apply_highpass(self, data: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to remove DC drift and electrode offset.

        Args:
            data: Input signal array

        Returns:
            High-pass filtered signal
        """
        if len(data) < 15:
            return data
        return filtfilt(self.b_hp, self.a_hp, data)

    def apply_lowpass(self, data: np.ndarray) -> np.ndarray:
        """
        Apply low-pass filter for anti-aliasing and noise reduction.

        Args:
            data: Input signal array

        Returns:
            Low-pass filtered signal
        """
        if len(data) < 15:
            return data
        return filtfilt(self.b_lp, self.a_lp, data)

    def apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter to isolate fungal signal frequency band.

        Args:
            data: Input signal array

        Returns:
            Bandpass filtered signal
        """
        if len(data) < 15:
            return data
        return filtfilt(self.b_bp, self.a_bp, data)

    def apply_notch(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove 60 Hz mains interference.

        Note: Only effective if sample_rate > 120 Hz (Nyquist > 60 Hz).
        At 10 Hz sample rate, 60 Hz noise is already aliased.

        Args:
            data: Input signal array

        Returns:
            Notch filtered signal
        """
        if len(data) < 15:
            return data
        if self.filter_config.notch_freq >= self.sample_rate / 2:
            warnings.warn(
                f"Notch filter at {self.filter_config.notch_freq} Hz not possible "
                f"with {self.sample_rate} Hz sample rate. Skipping."
            )
            return data
        return filtfilt(self.b_notch, self.a_notch, data)

    def apply_median_filter(self, data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply median filter to remove impulse noise/spikes.

        Args:
            data: Input signal array
            kernel_size: Size of median filter kernel (odd number)

        Returns:
            Median filtered signal
        """
        return median_filter(data, size=kernel_size)

    def remove_baseline(self, data: np.ndarray, method: str = 'median') -> np.ndarray:
        """
        Remove baseline drift from signal.

        Args:
            data: Input signal array
            method: 'median', 'mean', or 'polynomial'

        Returns:
            Baseline-corrected signal
        """
        if method == 'median':
            baseline = np.median(data)
        elif method == 'mean':
            baseline = np.mean(data)
        elif method == 'polynomial':
            # Fit low-order polynomial to remove slow drift
            x = np.arange(len(data))
            coeffs = np.polyfit(x, data, deg=2)
            baseline = np.polyval(coeffs, x)
        else:
            baseline = 0

        return data - baseline

    def process(self, data: np.ndarray,
                remove_dc: bool = True,
                apply_bandpass: bool = True) -> np.ndarray:
        """
        Apply complete processing pipeline.

        Pipeline order:
        1. Remove DC offset (optional)
        2. Apply bandpass filter

        Args:
            data: Input signal array
            remove_dc: Whether to remove DC offset
            apply_bandpass: Whether to apply bandpass filter

        Returns:
            Processed signal
        """
        result = data.copy()

        if remove_dc:
            result = self.remove_baseline(result, method='median')

        if apply_bandpass:
            if len(result) >= 15:
                result = self.apply_bandpass(result)

        return result

    # ==================== STFT ANALYSIS ====================

    def compute_stft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform (STFT).

        Based on Buffi et al. 2025 methodology for analyzing
        fungal bioelectrical signal frequency content over time.

        Args:
            data: Input signal array

        Returns:
            f: Frequency array (Hz)
            t: Time array (seconds)
            Zxx: Complex STFT matrix
        """
        cfg = self.stft_config

        f, t, Zxx = stft(
            data,
            fs=self.sample_rate,
            window=cfg.window_type,
            nperseg=cfg.nperseg,
            noverlap=cfg.noverlap,
            nfft=cfg.nfft
        )

        return f, t, Zxx

    def compute_spectrogram(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute power spectrogram from STFT.

        Args:
            data: Input signal array

        Returns:
            f: Frequency array (Hz)
            t: Time array (seconds)
            power: Power spectrogram (magnitude squared)
        """
        f, t, Zxx = self.compute_stft(data)
        power = np.abs(Zxx) ** 2
        return f, t, power

    def compute_spectrogram_db(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute power spectrogram in decibels.

        Args:
            data: Input signal array

        Returns:
            f: Frequency array (Hz)
            t: Time array (seconds)
            power_db: Power spectrogram in dB
        """
        f, t, power = self.compute_spectrogram(data)

        # Convert to dB, avoiding log(0)
        power_db = 10 * np.log10(power + 1e-10)

        return f, t, power_db

    def get_dominant_frequency(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Find the dominant frequency in the signal.

        Args:
            data: Input signal array

        Returns:
            freq: Dominant frequency (Hz)
            power: Power at dominant frequency
        """
        f, t, power = self.compute_spectrogram(data)

        # Average power over time
        avg_power = np.mean(power, axis=1)

        # Find peak frequency
        peak_idx = np.argmax(avg_power)

        return f[peak_idx], avg_power[peak_idx]

    def get_frequency_bands(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute power in different frequency bands.

        Bands based on fungal signal characteristics:
        - Ultra-low: 0.01-0.05 Hz (slow oscillations)
        - Low: 0.05-0.2 Hz (typical fungal rhythm)
        - Mid: 0.2-0.5 Hz (faster activity)
        - High: 0.5-2.0 Hz (rapid events)

        Args:
            data: Input signal array

        Returns:
            Dictionary with power in each band
        """
        f, t, power = self.compute_spectrogram(data)
        avg_power = np.mean(power, axis=1)

        bands = {
            'ultra_low': (0.01, 0.05),
            'low': (0.05, 0.2),
            'mid': (0.2, 0.5),
            'high': (0.5, 2.0)
        }

        result = {}
        for band_name, (f_low, f_high) in bands.items():
            mask = (f >= f_low) & (f < f_high)
            if np.any(mask):
                result[band_name] = np.sum(avg_power[mask])
            else:
                result[band_name] = 0.0

        # Total power
        result['total'] = np.sum(avg_power)

        return result

    # ==================== SPIKE DETECTION ====================

    def detect_spikes(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect spikes/action potentials in the signal.

        Uses threshold-based detection with the following criteria:
        - Amplitude > mean + threshold_std * std
        - Minimum distance between spikes
        - Minimum prominence

        Args:
            data: Input signal array

        Returns:
            spike_indices: Array of spike locations (sample indices)
            spike_info: Dictionary with spike statistics
        """
        cfg = self.spike_config

        if len(data) < 10:
            return np.array([]), {'count': 0}

        # Calculate threshold
        baseline = np.median(data)
        std = np.std(data)
        threshold = baseline + cfg.threshold_std * std

        # Find peaks
        peaks, properties = find_peaks(
            data,
            height=threshold,
            distance=cfg.min_distance_samples,
            prominence=cfg.min_prominence
        )

        # Compile spike information
        spike_info = {
            'count': len(peaks),
            'threshold': threshold,
            'baseline': baseline,
            'std': std,
            'amplitudes': data[peaks] if len(peaks) > 0 else np.array([]),
            'prominences': properties.get('prominences', np.array([])),
            'rate_per_minute': len(peaks) / (len(data) / self.sample_rate / 60) if len(data) > 0 else 0
        }

        return peaks, spike_info

    def characterize_spikes(self, data: np.ndarray,
                           spike_indices: np.ndarray,
                           window_samples: int = 10) -> List[Dict]:
        """
        Characterize individual spikes.

        For each spike, extract:
        - Peak amplitude
        - Rise time
        - Fall time
        - Duration
        - Area under curve

        Args:
            data: Input signal array
            spike_indices: Indices of detected spikes
            window_samples: Samples to analyze around each spike

        Returns:
            List of dictionaries with spike characteristics
        """
        spikes = []

        for idx in spike_indices:
            # Window bounds
            start = max(0, idx - window_samples)
            end = min(len(data), idx + window_samples + 1)

            window = data[start:end]
            local_idx = idx - start

            # Peak amplitude
            amplitude = data[idx]

            # Find half-max points for duration
            half_max = (amplitude + np.min(window)) / 2
            above_half = window > half_max

            # Rise time (samples from start to peak)
            rise_samples = local_idx

            # Fall time (samples from peak to end)
            fall_samples = len(window) - local_idx - 1

            # Duration at half-max
            duration_samples = np.sum(above_half)

            # Area under curve (simple integration)
            area = np.trapz(window - np.min(window))

            spikes.append({
                'index': idx,
                'time': idx / self.sample_rate,
                'amplitude': amplitude,
                'rise_time': rise_samples / self.sample_rate,
                'fall_time': fall_samples / self.sample_rate,
                'duration': duration_samples / self.sample_rate,
                'area': area
            })

        return spikes

    # ==================== VISUALIZATION ====================

    def plot_processing_pipeline(self, data: np.ndarray,
                                 title: str = "Signal Processing Pipeline"):
        """
        Plot the signal at each stage of processing.

        Shows:
        1. Raw signal
        2. After highpass (DC removed)
        3. After bandpass (frequency limited)
        4. With detected spikes

        Args:
            data: Input signal array
            title: Plot title
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(title, fontsize=14)

        t = np.arange(len(data)) / self.sample_rate

        # Raw signal
        axes[0].plot(t, data, 'b-', linewidth=0.5)
        axes[0].set_ylabel('Voltage (mV)')
        axes[0].set_title('Raw Signal')
        axes[0].grid(True, alpha=0.3)

        # After DC removal
        data_no_dc = self.remove_baseline(data)
        axes[1].plot(t, data_no_dc, 'g-', linewidth=0.5)
        axes[1].set_ylabel('Voltage (mV)')
        axes[1].set_title('After DC Removal (Baseline Correction)')
        axes[1].grid(True, alpha=0.3)

        # After bandpass
        if len(data) >= 15:
            data_filtered = self.apply_bandpass(data_no_dc)
        else:
            data_filtered = data_no_dc
        axes[2].plot(t, data_filtered, 'purple', linewidth=0.5)
        axes[2].set_ylabel('Voltage (mV)')
        axes[2].set_title(f'After Bandpass ({self.filter_config.bandpass_low}-{self.filter_config.bandpass_high} Hz)')
        axes[2].grid(True, alpha=0.3)

        # Spike detection
        spikes, spike_info = self.detect_spikes(data_filtered)
        axes[3].plot(t, data_filtered, 'b-', linewidth=0.5)
        if len(spikes) > 0:
            axes[3].scatter(t[spikes], data_filtered[spikes],
                           c='red', s=50, zorder=5, label=f'Spikes (n={len(spikes)})')
        axes[3].axhline(y=spike_info['threshold'], color='r', linestyle='--',
                        alpha=0.5, label=f'Threshold ({spike_info["threshold"]:.2f} mV)')
        axes[3].axhline(y=spike_info['baseline'], color='gray', linestyle='--',
                        alpha=0.5, label=f'Baseline ({spike_info["baseline"]:.2f} mV)')
        axes[3].set_ylabel('Voltage (mV)')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_title('Spike Detection')
        axes[3].legend(loc='upper right', fontsize=8)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_spectrogram(self, data: np.ndarray,
                         title: str = "STFT Spectrogram Analysis"):
        """
        Plot STFT spectrogram with signal overlay.

        Args:
            data: Input signal array
            title: Plot title
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                                  gridspec_kw={'height_ratios': [1, 2]})
        fig.suptitle(title, fontsize=14)

        t_signal = np.arange(len(data)) / self.sample_rate

        # Plot filtered signal
        processed = self.process(data)
        axes[0].plot(t_signal, processed, 'b-', linewidth=0.5)
        axes[0].set_ylabel('Voltage (mV)')
        axes[0].set_title('Filtered Signal')
        axes[0].grid(True, alpha=0.3)

        # Plot spectrogram
        f, t, power_db = self.compute_spectrogram_db(data)

        # Limit frequency range
        freq_mask = f <= self.stft_config.freq_max

        im = axes[1].pcolormesh(t, f[freq_mask], power_db[freq_mask, :],
                                shading='gouraud', cmap='viridis')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('STFT Power Spectrogram')

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1])
        cbar.set_label('Power (dB)')

        plt.tight_layout()
        return fig

    def plot_frequency_analysis(self, data: np.ndarray,
                                title: str = "Frequency Analysis"):
        """
        Plot frequency domain analysis.

        Shows:
        - Power spectral density
        - Frequency band breakdown

        Args:
            data: Input signal array
            title: Plot title
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=14)

        # Compute PSD
        f, t, power = self.compute_spectrogram(data)
        avg_psd = np.mean(power, axis=1)

        # Plot PSD
        axes[0].semilogy(f, avg_psd, 'b-', linewidth=1)
        axes[0].set_xlabel('Frequency (Hz)')
        axes[0].set_ylabel('Power Spectral Density')
        axes[0].set_title('Average Power Spectrum')
        axes[0].set_xlim([0, self.stft_config.freq_max])
        axes[0].grid(True, alpha=0.3)

        # Add frequency band regions
        band_colors = {'ultra_low': 'red', 'low': 'orange', 'mid': 'green', 'high': 'blue'}
        bands = {
            'ultra_low': (0.01, 0.05),
            'low': (0.05, 0.2),
            'mid': (0.2, 0.5),
            'high': (0.5, 2.0)
        }
        for band_name, (f_low, f_high) in bands.items():
            axes[0].axvspan(f_low, f_high, alpha=0.1, color=band_colors[band_name],
                           label=f'{band_name} ({f_low}-{f_high} Hz)')
        axes[0].legend(fontsize=8)

        # Plot frequency band breakdown
        band_powers = self.get_frequency_bands(data)
        band_names = ['ultra_low', 'low', 'mid', 'high']
        powers = [band_powers[b] for b in band_names]
        colors = [band_colors[b] for b in band_names]

        bars = axes[1].bar(band_names, powers, color=colors, alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Frequency Band')
        axes[1].set_ylabel('Total Power')
        axes[1].set_title('Power by Frequency Band')
        axes[1].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, power in zip(bars, powers):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{power:.2e}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return fig


# ==================== STANDALONE EXECUTION ====================

def load_csv_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load data from CSV file."""
    import csv

    times = []
    voltages = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header

        for row in reader:
            try:
                if len(row) >= 3:
                    times.append(float(row[0]))
                    voltages.append(float(row[2]))  # voltage_mV column
            except (ValueError, IndexError):
                continue

    return np.array(times), np.array(voltages)


def main():
    """Main function for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Signal processing for fungal bioelectrical signals'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file for processed data')
    parser.add_argument('--sample-rate', '-s', type=float, default=10.0,
                        help='Sampling rate in Hz (default: 10)')
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots')
    parser.add_argument('--save-plots', type=str, default=None,
                        help='Save plots to directory')

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    times, voltages = load_csv_data(args.input)
    print(f"Loaded {len(voltages)} samples")

    # Create processor
    processor = SignalProcessor(sample_rate=args.sample_rate)

    # Process signal
    print("Processing signal...")
    processed = processor.process(voltages)

    # Detect spikes
    spikes, spike_info = processor.detect_spikes(processed)
    print(f"Detected {spike_info['count']} spikes")
    print(f"Spike rate: {spike_info['rate_per_minute']:.2f} per minute")

    # Frequency analysis
    bands = processor.get_frequency_bands(voltages)
    print("\nFrequency band power:")
    for band, power in bands.items():
        print(f"  {band}: {power:.4e}")

    # Get dominant frequency
    dom_freq, dom_power = processor.get_dominant_frequency(voltages)
    print(f"\nDominant frequency: {dom_freq:.3f} Hz")

    # Save processed data
    if args.output:
        import csv
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['time_ms', 'raw_mV', 'processed_mV', 'is_spike'])

            spike_set = set(spikes)
            for i, (t, raw, proc) in enumerate(zip(times, voltages, processed)):
                is_spike = 1 if i in spike_set else 0
                writer.writerow([t, raw, proc, is_spike])

        print(f"\nProcessed data saved to {args.output}")

    # Generate plots
    if args.show_plots or args.save_plots:
        print("\nGenerating plots...")

        fig1 = processor.plot_processing_pipeline(voltages)
        fig2 = processor.plot_spectrogram(voltages)
        fig3 = processor.plot_frequency_analysis(voltages)

        if args.save_plots:
            import os
            os.makedirs(args.save_plots, exist_ok=True)

            fig1.savefig(os.path.join(args.save_plots, 'pipeline.png'), dpi=150)
            fig2.savefig(os.path.join(args.save_plots, 'spectrogram.png'), dpi=150)
            fig3.savefig(os.path.join(args.save_plots, 'frequency_analysis.png'), dpi=150)
            print(f"Plots saved to {args.save_plots}/")

        if args.show_plots:
            plt.show()

    print("\nDone!")


if __name__ == '__main__':
    main()
