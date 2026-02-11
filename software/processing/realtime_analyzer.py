#!/usr/bin/env python3
"""
realtime_analyzer.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Real-time signal analysis with STFT spectrogram, filtering, and spike detection.
Works with the voltage divider simulator or real mycelium signals.

Usage:
    python realtime_analyzer.py                    # Auto-detect Arduino
    python realtime_analyzer.py --port COM3        # Windows
    python realtime_analyzer.py --port /dev/ttyACM0  # Linux/Mac
    python realtime_analyzer.py --demo             # Demo mode (no Arduino needed)

Requirements:
    pip install pyserial numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from scipy.signal import butter, filtfilt, iirnotch, stft
import serial
import serial.tools.list_ports
import argparse
import time
import sys
from datetime import datetime


class RealtimeAnalyzer:
    """Real-time signal analyzer with filtering and STFT."""

    def __init__(self, sample_rate=10, buffer_seconds=60):
        self.sample_rate = sample_rate
        self.buffer_size = sample_rate * buffer_seconds

        # Data buffers
        self.times = deque(maxlen=self.buffer_size)
        self.raw_data = deque(maxlen=self.buffer_size)
        self.filtered_data = deque(maxlen=self.buffer_size)

        # Statistics
        self.spike_count = 0
        self.start_time = None

        # Initialize filters
        self._init_filters()

    def _init_filters(self):
        """Initialize filter coefficients."""
        fs = self.sample_rate
        nyq = fs / 2

        # High-pass filter (0.01 Hz) - removes DC drift
        hp_cutoff = 0.01
        if hp_cutoff < nyq:
            hp_norm = max(0.001, hp_cutoff / nyq)
            self.b_hp, self.a_hp = butter(2, hp_norm, btype='high')
        else:
            self.b_hp, self.a_hp = [1], [1]

        # Low-pass filter (2 Hz) - removes high-freq noise
        lp_cutoff = 2.0
        if lp_cutoff < nyq:
            lp_norm = min(0.99, lp_cutoff / nyq)
            self.b_lp, self.a_lp = butter(2, lp_norm, btype='low')
        else:
            self.b_lp, self.a_lp = [1], [1]

        # Bandpass filter (0.01-2 Hz) - isolates fungal band
        bp_low = max(0.001, 0.01 / nyq)
        bp_high = min(0.99, 2.0 / nyq)
        self.b_bp, self.a_bp = butter(2, [bp_low, bp_high], btype='band')

        # Notch filter (60 Hz) - only if sample rate allows
        if 60.0 < nyq:
            self.b_notch, self.a_notch = iirnotch(60.0, 30.0, fs)
            self.can_notch = True
        else:
            self.b_notch, self.a_notch = [1], [1]
            self.can_notch = False

    def add_sample(self, t, voltage):
        """Add a new sample to the buffer."""
        self.times.append(t)
        self.raw_data.append(voltage)

        # Apply filtering if enough data
        if len(self.raw_data) >= 15:
            data = np.array(self.raw_data)

            # Remove DC offset
            data = data - np.median(data)

            # Apply bandpass
            try:
                filtered = filtfilt(self.b_bp, self.a_bp, data)
                self.filtered_data.append(filtered[-1])
            except:
                self.filtered_data.append(data[-1])
        else:
            self.filtered_data.append(voltage)

    def get_data(self):
        """Get current buffer data as numpy arrays."""
        return np.array(self.times), np.array(self.raw_data), np.array(self.filtered_data)

    def compute_stft(self, data):
        """Compute STFT spectrogram."""
        if len(data) < 64:
            return None, None, None

        f, t, Zxx = stft(
            data,
            fs=self.sample_rate,
            window='hann',
            nperseg=min(64, len(data) // 2),
            noverlap=min(48, len(data) // 3),
            nfft=128
        )
        return f, t, np.abs(Zxx)

    def detect_spikes(self, data, threshold_std=3.0):
        """Detect spikes above threshold."""
        if len(data) < 10:
            return np.array([]), 0, 0

        baseline = np.median(data)
        std = np.std(data)
        threshold = baseline + threshold_std * std

        spikes = np.where(data > threshold)[0]
        return spikes, baseline, threshold


class SerialReader:
    """Read data from Arduino serial port."""

    def __init__(self, port=None, baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None

    def find_arduino(self):
        """Auto-detect Arduino port."""
        ports = serial.tools.list_ports.comports()
        keywords = ['arduino', 'usbmodem', 'usbserial', 'acm', 'ch340', 'cp210']

        for port in ports:
            port_lower = (port.device + ' ' + str(port.description)).lower()
            for keyword in keywords:
                if keyword in port_lower:
                    print(f"Found Arduino: {port.device}")
                    return port.device

        if ports:
            print(f"Using first available port: {ports[0].device}")
            return ports[0].device

        return None

    def connect(self):
        """Connect to serial port."""
        if not self.port:
            self.port = self.find_arduino()

        if not self.port:
            print("Error: No serial port found!")
            return False

        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(2)  # Wait for Arduino reset
            self.ser.reset_input_buffer()
            print(f"Connected to {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Error: {e}")
            return False

    def read_sample(self):
        """Read one sample from serial."""
        if not self.ser or not self.ser.is_open:
            return None

        try:
            line = self.ser.readline().decode('utf-8').strip()

            if not line or line.startswith('#'):
                return None

            parts = line.split(',')
            if len(parts) >= 4:
                t_ms = int(parts[0])
                voltage = float(parts[3])  # voltage_mV column
                return t_ms / 1000.0, voltage

        except (ValueError, UnicodeDecodeError):
            pass

        return None

    def close(self):
        """Close serial connection."""
        if self.ser and self.ser.is_open:
            self.ser.close()


class DemoSignalGenerator:
    """Generate demo signals for testing without Arduino."""

    def __init__(self, sample_rate=10):
        self.sample_rate = sample_rate
        self.t = 0
        self.last_spike = 0

    def get_sample(self):
        """Generate a simulated fungal signal sample."""
        # Base slow oscillation (0.05 Hz)
        slow = 0.5 * np.sin(2 * np.pi * 0.05 * self.t)

        # Random noise
        noise = 0.1 * np.random.randn()

        # Occasional spikes
        spike = 0
        if self.t - self.last_spike > np.random.uniform(3, 10):
            spike = np.random.uniform(0.5, 1.5)
            self.last_spike = self.t

        voltage = 1.0 + slow + noise + spike  # Center around 1 mV

        self.t += 1.0 / self.sample_rate
        return self.t, voltage


def run_realtime_analyzer(port=None, demo_mode=False):
    """Main function to run real-time analyzer."""

    # Initialize components
    analyzer = RealtimeAnalyzer(sample_rate=10, buffer_seconds=60)

    if demo_mode:
        print("Running in DEMO mode (simulated signals)")
        reader = None
        demo_gen = DemoSignalGenerator(sample_rate=10)
    else:
        reader = SerialReader(port=port)
        if not reader.connect():
            print("Failed to connect. Use --demo for demo mode.")
            return
        demo_gen = None

    # Set up matplotlib figure
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Fungal Bioelectrical Signal Analyzer', fontsize=14, color='white')

    # Create subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5], hspace=0.3, wspace=0.3)

    ax_raw = fig.add_subplot(gs[0, :])
    ax_filtered = fig.add_subplot(gs[1, :])
    ax_stft = fig.add_subplot(gs[2, 0])
    ax_stats = fig.add_subplot(gs[2, 1])

    # Initialize plots
    line_raw, = ax_raw.plot([], [], 'cyan', linewidth=0.5, label='Raw')
    ax_raw.set_ylabel('Voltage (mV)')
    ax_raw.set_title('Raw Signal')
    ax_raw.grid(True, alpha=0.3)
    ax_raw.set_xlim(0, 60)
    ax_raw.set_ylim(-1, 3)

    line_filtered, = ax_filtered.plot([], [], 'lime', linewidth=0.5, label='Filtered')
    line_baseline = ax_filtered.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    line_threshold = ax_filtered.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    scatter_spikes = ax_filtered.scatter([], [], c='red', s=50, zorder=5)
    ax_filtered.set_ylabel('Voltage (mV)')
    ax_filtered.set_title('Filtered (0.01-2 Hz Bandpass) + Spike Detection')
    ax_filtered.grid(True, alpha=0.3)
    ax_filtered.set_xlim(0, 60)
    ax_filtered.set_ylim(-1, 2)

    ax_stft.set_xlabel('Time (s)')
    ax_stft.set_ylabel('Frequency (Hz)')
    ax_stft.set_title('STFT Spectrogram')

    # Stats display
    ax_stats.axis('off')
    stats_text = ax_stats.text(0.1, 0.9, '', transform=ax_stats.transAxes,
                                fontsize=12, verticalalignment='top',
                                family='monospace', color='white')

    start_time = time.time()
    sample_count = [0]
    spike_total = [0]

    def update(frame):
        """Animation update function."""
        nonlocal start_time

        # Read samples (multiple per frame for smoother updates)
        for _ in range(3):
            if demo_mode:
                t, v = demo_gen.get_sample()
                analyzer.add_sample(t, v)
                sample_count[0] += 1
            else:
                result = reader.read_sample()
                if result:
                    t, v = result
                    analyzer.add_sample(t, v)
                    sample_count[0] += 1

        # Get data
        times, raw, filtered = analyzer.get_data()

        if len(times) < 10:
            return line_raw, line_filtered, scatter_spikes, stats_text

        # Normalize time axis
        t_norm = times - times[0]

        # Update raw plot
        line_raw.set_data(t_norm, raw)
        ax_raw.set_xlim(0, max(60, t_norm[-1]))
        ax_raw.set_ylim(np.min(raw) - 0.5, np.max(raw) + 0.5)

        # Update filtered plot
        line_filtered.set_data(t_norm, filtered)

        # Spike detection
        spikes, baseline, threshold = analyzer.detect_spikes(filtered)

        # Update threshold lines
        line_baseline.set_ydata([baseline, baseline])
        line_threshold.set_ydata([threshold, threshold])

        # Update spike scatter
        if len(spikes) > 0:
            spike_times = t_norm[spikes]
            spike_vals = filtered[spikes]
            scatter_spikes.set_offsets(np.column_stack([spike_times, spike_vals]))
            spike_total[0] += len(spikes)
        else:
            scatter_spikes.set_offsets(np.empty((0, 2)))

        ax_filtered.set_xlim(0, max(60, t_norm[-1]))
        ax_filtered.set_ylim(np.min(filtered) - 0.5, np.max(filtered) + 0.5)

        # Update STFT spectrogram
        if len(filtered) >= 64:
            f, t_stft, Sxx = analyzer.compute_stft(filtered)
            if f is not None:
                ax_stft.clear()
                ax_stft.pcolormesh(t_stft, f, Sxx, shading='gouraud', cmap='magma')
                ax_stft.set_ylim(0, 2)
                ax_stft.set_xlabel('Time (s)')
                ax_stft.set_ylabel('Frequency (Hz)')
                ax_stft.set_title('STFT Spectrogram')

        # Update stats
        elapsed = time.time() - start_time
        rate = sample_count[0] / elapsed if elapsed > 0 else 0

        stats_str = f"""
STATISTICS
══════════════════════════
Samples:     {sample_count[0]:>8}
Duration:    {elapsed:>8.1f} s
Sample Rate: {rate:>8.1f} Hz

SIGNAL
══════════════════════════
Current:     {raw[-1]:>8.3f} mV
Mean:        {np.mean(raw):>8.3f} mV
Std Dev:     {np.std(raw):>8.3f} mV
Min:         {np.min(raw):>8.3f} mV
Max:         {np.max(raw):>8.3f} mV

DETECTION
══════════════════════════
Baseline:    {baseline:>8.3f} mV
Threshold:   {threshold:>8.3f} mV
Spikes:      {len(spikes):>8}

FILTERS
══════════════════════════
Highpass:         0.01 Hz
Lowpass:          2.00 Hz
Notch (60Hz):     {'N/A (fs<120)' if not analyzer.can_notch else 'Active'}
        """
        stats_text.set_text(stats_str)

        return line_raw, line_filtered, scatter_spikes, stats_text

    # Create animation
    ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

    print("\n" + "=" * 50)
    print("Real-time Analyzer Started")
    print("Press Ctrl+C or close window to stop")
    print("=" * 50 + "\n")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        if reader:
            reader.close()
        print("\nAnalyzer stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time fungal signal analyzer with STFT'
    )
    parser.add_argument('--port', '-p', type=str, default=None,
                        help='Serial port (auto-detect if not specified)')
    parser.add_argument('--demo', '-d', action='store_true',
                        help='Demo mode with simulated signals')
    parser.add_argument('--list-ports', '-l', action='store_true',
                        help='List available serial ports')

    args = parser.parse_args()

    if args.list_ports:
        ports = serial.tools.list_ports.comports()
        print("Available ports:")
        for port in ports:
            print(f"  {port.device}: {port.description}")
        return

    run_realtime_analyzer(port=args.port, demo_mode=args.demo)


if __name__ == '__main__':
    main()
