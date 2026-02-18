#!/usr/bin/env python3
"""
realtime_analyzer.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Real-time signal analysis with STFT spectrogram, filtering, and spike detection.
Works with the voltage divider simulator or real mycelium signals.

NEW: Detects mode changes and shows different segments in different colors!

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
from matplotlib.collections import LineCollection
from collections import deque
from scipy.signal import butter, filtfilt, iirnotch, stft
import serial
import serial.tools.list_ports
import argparse
import time
import sys
import threading
import queue
from datetime import datetime


# Colors for different modes/segments
SEGMENT_COLORS = [
    '#00FFFF',  # Cyan
    '#FF6B6B',  # Red/Coral
    '#4ECDC4',  # Teal
    '#FFE66D',  # Yellow
    '#95E1D3',  # Mint
    '#F38181',  # Salmon
    '#AA96DA',  # Purple
    '#FCBAD3',  # Pink
    '#A8D8EA',  # Light Blue
    '#FF9F43',  # Orange
]

MODE_NAMES = {
    0: 'Sine',
    1: 'Random Walk',
    2: 'Spikes',
    3: 'Composite',
    4: 'NOTHING',
    5: 'NOISE',
    6: 'DRIFT',
    7: 'SATURATION',
    8: 'INTERMITTENT',
    9: 'STIMULUS',
    10: 'MYCELIUM',  # Realistic fungal signal simulation
}


class RealtimeAnalyzer:
    """Real-time signal analyzer with filtering and STFT."""

    def __init__(self, sample_rate=10, buffer_seconds=120):
        self.sample_rate = sample_rate
        self.buffer_size = sample_rate * buffer_seconds

        # Data buffers
        self.times = deque(maxlen=self.buffer_size)
        self.raw_data = deque(maxlen=self.buffer_size)
        self.filtered_data = deque(maxlen=self.buffer_size)

        # Segment tracking (for color changes on mode switch)
        self.segment_ids = deque(maxlen=self.buffer_size)  # Which segment each sample belongs to
        self.current_segment = 0
        self.segment_starts = [0]  # Sample indices where segments start
        self.segment_modes = ['Composite']  # Default to Composite (mode 3) on startup

        # Statistics
        self.spike_count = 0
        self.start_time = None
        self.continuous_time = 0  # Continuous time counter

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

    def add_sample(self, arduino_time, voltage):
        """Add a new sample to the buffer.

        Note: Mode changes are now handled by keyboard input, not Arduino time resets.
        The keyboard handler creates new segments directly.
        """
        # Use continuous time instead of Arduino time (avoids jumps when Arduino resets)
        self.continuous_time += 1.0 / self.sample_rate

        self.times.append(self.continuous_time)
        self.raw_data.append(voltage)
        self.segment_ids.append(self.current_segment)

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

    def update_mode_name(self, mode_num, from_keyboard=False):
        """Update the name of the current/latest segment.

        Only update if from_keyboard=True (user initiated) to avoid
        Arduino messages overwriting user's mode selection.
        """
        if not from_keyboard:
            return  # Ignore Arduino mode messages

        mode_name = MODE_NAMES.get(mode_num, f'Mode {mode_num}')

        # Always update the latest segment with the new mode name
        if self.segment_modes:
            self.segment_modes[-1] = mode_name
            print(f"[SEGMENT] Updated to: {mode_name}")

    def get_data(self):
        """Get current buffer data as numpy arrays."""
        return (np.array(self.times),
                np.array(self.raw_data),
                np.array(self.filtered_data),
                np.array(self.segment_ids))

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


class KeyboardInput:
    """Keyboard input handler using matplotlib figure key press events.

    This is more reliable than terminal input because it uses matplotlib's
    built-in event system which works on all platforms.
    """

    def __init__(self):
        self.mode_queue = queue.Queue()
        self.fig = None

    def connect_to_figure(self, fig):
        """Connect key press handler to a matplotlib figure."""
        self.fig = fig
        fig.canvas.mpl_connect('key_press_event', self._on_key_press)

        print("\n" + "=" * 50)
        print("KEYBOARD INPUT ENABLED")
        print("=" * 50)
        print(">>> CLICK ON THE GRAPH WINDOW, then press 0-9 or m <<<")
        print("  0 = Sine        5 = NOISE")
        print("  1 = Random Walk 6 = DRIFT")
        print("  2 = Spikes      7 = SATURATION")
        print("  3 = Composite   8 = INTERMITTENT")
        print("  4 = NOTHING     9 = STIMULUS")
        print("  m = MYCELIUM (realistic fungal signal!)")
        print("  q = Quit")
        print("=" * 50 + "\n")

    def _on_key_press(self, event):
        """Handle key press events from matplotlib."""
        # Debug: show ALL key presses
        print(f"[KEY] Pressed: '{event.key}'", flush=True)

        if event.key is None:
            return

        key = event.key

        if key in '0123456789':
            mode_num = int(key)
            mode_name = MODE_NAMES.get(mode_num, f'Mode {mode_num}')
            print(f"\n>>> MODE: {mode_num} ({mode_name}) <<<", flush=True)
            self.mode_queue.put(mode_num)
        elif key.lower() == 'm':
            mode_num = 10  # MYCELIUM mode
            mode_name = MODE_NAMES.get(mode_num, 'MYCELIUM')
            print(f"\n>>> MODE: {mode_num} ({mode_name}) <<<", flush=True)
            print("    Realistic fungal bioelectrical signal simulation!", flush=True)
            print("    Watch for: slow oscillations, action potentials, nutrient responses\n", flush=True)
            self.mode_queue.put(mode_num)
        elif key.lower() == 'q':
            print("Quit requested...", flush=True)
            plt.close('all')

    def start(self):
        """Compatibility method - does nothing since we use figure events."""
        pass

    def get_pending_mode(self):
        """Get next pending mode change from queue (non-blocking)."""
        try:
            return self.mode_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop the listener."""
        pass


class SerialReader:
    """Read data from Arduino serial port."""

    def __init__(self, port=None, baud=115200):
        self.port = port
        self.baud = baud
        self.ser = None
        self.current_mode = None

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
        """Read one sample from serial. Returns (time, voltage, mode_changed, mode_num)"""
        if not self.ser or not self.ser.is_open:
            return None

        try:
            line = self.ser.readline().decode('utf-8').strip()

            if not line:
                return None

            # Check for mode change messages
            if line.startswith('# Mode:'):
                # Parse mode number from "# Mode: X (Name)"
                # Note: We don't print or act on these - keyboard controls mode now
                try:
                    mode_num = int(line.split(':')[1].split('(')[0].strip())
                    # Only print if mode actually changed
                    if self.current_mode != mode_num:
                        self.current_mode = mode_num
                        mode_name = MODE_NAMES.get(mode_num, f'Mode {mode_num}')
                        print(f"[ARDUINO] Confirmed mode: {mode_name}")
                        return ('MODE_CHANGE', mode_num)
                except:
                    pass
                return None

            # Skip other comments
            if line.startswith('#'):
                if 'STIMULUS' in line or 'INTERMITTENT' in line:
                    print(f"[ARDUINO] {line}")
                return None

            parts = line.split(',')
            if len(parts) >= 4:
                t_ms = int(parts[0])
                voltage = float(parts[3])  # voltage_mV column
                return t_ms / 1000.0, voltage, self.current_mode

        except (ValueError, UnicodeDecodeError):
            pass

        return None

    def send_mode(self, mode_num):
        """Send mode change command to Arduino."""
        if self.ser and self.ser.is_open:
            try:
                cmd = b'm' if mode_num == 10 else str(mode_num).encode()
                self.ser.write(cmd)
                print(f"[SENT TO ARDUINO] Mode: {mode_num}")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to send: {e}")
                return False
        return False

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
        return self.t, voltage, 3  # Mode 3 = Composite


def run_realtime_analyzer(port=None, demo_mode=False):
    """Main function to run real-time analyzer."""

    # Initialize components
    analyzer = RealtimeAnalyzer(sample_rate=10, buffer_seconds=120)

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
    fig.suptitle('Fungal Bioelectrical Signal Analyzer - Press 0-9 to change mode', fontsize=14, color='white')

    # Set up keyboard input handler connected to the figure
    keyboard = KeyboardInput()
    keyboard.connect_to_figure(fig)

    # Create subplots
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.5], hspace=0.3, wspace=0.3)

    ax_raw = fig.add_subplot(gs[0, :])
    ax_filtered = fig.add_subplot(gs[1, :])
    ax_stft = fig.add_subplot(gs[2, 0])
    ax_stats = fig.add_subplot(gs[2, 1])

    # Initialize empty line collections for multi-colored segments
    ax_raw.set_ylabel('Voltage (mV)')
    ax_raw.set_title('Raw Signal (colors = different modes)')
    ax_raw.grid(True, alpha=0.3)
    ax_raw.set_xlim(0, 60)
    ax_raw.set_ylim(-1, 3)

    line_baseline = ax_filtered.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    line_threshold = ax_filtered.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    scatter_spikes = ax_filtered.scatter([], [], c='white', s=50, zorder=5, marker='*')
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
    stats_text = ax_stats.text(0.05, 0.95, '', transform=ax_stats.transAxes,
                                fontsize=11, verticalalignment='top',
                                family='monospace', color='white')

    start_time = time.time()
    sample_count = [0]
    spike_total = [0]
    current_mode = [3]  # Default to composite

    def update(frame):
        """Animation update function."""
        nonlocal start_time

        # Check for keyboard mode change request
        pending = keyboard.get_pending_mode()
        if pending is not None:
            if reader:
                # Send to Arduino
                reader.send_mode(pending)
            # Update local display immediately
            current_mode[0] = pending
            analyzer.current_segment += 1
            analyzer.segment_starts.append(len(analyzer.times))
            mode_name = MODE_NAMES.get(pending, f'Mode {pending}')
            analyzer.segment_modes.append(mode_name)
            print(f"[MODE] New segment: {mode_name}", flush=True)

        # Read samples (multiple per frame for smoother updates)
        for _ in range(3):
            if demo_mode:
                t, v, mode = demo_gen.get_sample()
                analyzer.add_sample(t, v)
                sample_count[0] += 1
            else:
                result = reader.read_sample()
                if result:
                    # Check if it's a mode change message from Arduino
                    if isinstance(result, tuple) and len(result) == 2 and result[0] == 'MODE_CHANGE':
                        # Just log it, don't update segment (keyboard controls that now)
                        pass
                    elif isinstance(result, tuple) and len(result) == 3:
                        t, v, mode = result
                        analyzer.add_sample(t, v)
                        sample_count[0] += 1
                        # Don't update mode from Arduino - keyboard controls it now

        # Get data
        times, raw, filtered, segment_ids = analyzer.get_data()

        if len(times) < 10:
            return

        # Clear and redraw raw signal with colored segments
        ax_raw.clear()
        ax_raw.set_ylabel('Voltage (mV)')
        ax_raw.set_title('Raw Signal (colors = different modes)')
        ax_raw.grid(True, alpha=0.3)

        # Plot each segment with different color
        unique_segments = np.unique(segment_ids)
        for seg_id in unique_segments:
            mask = segment_ids == seg_id
            seg_times = times[mask]
            seg_raw = raw[mask]
            color = SEGMENT_COLORS[seg_id % len(SEGMENT_COLORS)]

            # Get segment name
            if seg_id < len(analyzer.segment_modes):
                label = analyzer.segment_modes[seg_id]
            else:
                label = f'Segment {seg_id}'

            ax_raw.plot(seg_times, seg_raw, color=color, linewidth=0.8, label=label)

        # Add legend if multiple segments
        if len(unique_segments) > 1:
            ax_raw.legend(loc='upper right', fontsize=8, ncol=min(3, len(unique_segments)))

        ax_raw.set_xlim(times[0], max(times[0] + 60, times[-1]))
        ax_raw.set_ylim(np.min(raw) - 0.3, np.max(raw) + 0.3)

        # Clear and redraw filtered signal with colored segments
        ax_filtered.clear()
        ax_filtered.set_ylabel('Voltage (mV)')
        ax_filtered.set_title('Filtered (0.01-2 Hz Bandpass) + Spike Detection')
        ax_filtered.grid(True, alpha=0.3)

        for seg_id in unique_segments:
            mask = segment_ids == seg_id
            seg_times = times[mask]
            seg_filtered = filtered[mask]
            color = SEGMENT_COLORS[seg_id % len(SEGMENT_COLORS)]
            ax_filtered.plot(seg_times, seg_filtered, color=color, linewidth=0.8)

        # Spike detection
        spikes, baseline, threshold = analyzer.detect_spikes(filtered)

        # Draw threshold lines
        ax_filtered.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax_filtered.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')

        # Draw spikes
        if len(spikes) > 0:
            spike_times = times[spikes]
            spike_vals = filtered[spikes]
            ax_filtered.scatter(spike_times, spike_vals, c='white', s=80, zorder=5,
                              marker='*', edgecolors='red', linewidths=0.5)
            spike_total[0] = len(spikes)

        ax_filtered.set_xlim(times[0], max(times[0] + 60, times[-1]))
        ax_filtered.set_ylim(np.min(filtered) - 0.3, np.max(filtered) + 0.3)
        ax_filtered.legend(loc='upper right', fontsize=8)

        # Update STFT spectrogram
        if len(filtered) >= 64:
            f, t_stft, Sxx = analyzer.compute_stft(filtered)
            if f is not None:
                ax_stft.clear()
                ax_stft.pcolormesh(t_stft + times[0], f, Sxx, shading='gouraud', cmap='magma')
                ax_stft.set_ylim(0, 2)
                ax_stft.set_xlabel('Time (s)')
                ax_stft.set_ylabel('Frequency (Hz)')
                ax_stft.set_title('STFT Spectrogram')

        # Update stats
        elapsed = time.time() - start_time
        rate = sample_count[0] / elapsed if elapsed > 0 else 0

        # Build segment history string
        seg_history = ""
        for i, mode_name in enumerate(analyzer.segment_modes[-5:]):  # Last 5 segments
            color_idx = (analyzer.current_segment - len(analyzer.segment_modes[-5:]) + i + 1) % len(SEGMENT_COLORS)
            seg_history += f"  {i+1}. {mode_name}\n"

        mode_name = MODE_NAMES.get(current_mode[0], f'Mode {current_mode[0]}')

        stats_str = f"""STATISTICS
══════════════════════════════
Samples:     {sample_count[0]:>8}
Duration:    {elapsed:>8.1f} s
Sample Rate: {rate:>8.1f} Hz

CURRENT MODE
══════════════════════════════
{mode_name}

SIGNAL
══════════════════════════════
Current:     {raw[-1]:>8.3f} mV
Mean:        {np.mean(raw):>8.3f} mV
Std Dev:     {np.std(raw):>8.3f} mV
Range:  {np.min(raw):>6.2f} - {np.max(raw):.2f} mV

DETECTION
══════════════════════════════
Baseline:    {baseline:>8.3f} mV
Threshold:   {threshold:>8.3f} mV
Spikes:      {spike_total[0]:>8}

SEGMENT HISTORY
══════════════════════════════
{seg_history}
COMMANDS (press on graph)
══════════════════════════════
0-9 = Change mode
  m = MYCELIUM (realistic!)
  q = Quit
"""
        stats_text.set_text(stats_str)

    # Create animation
    ani = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)

    print("\n" + "=" * 50)
    print("Real-time Analyzer Started")
    print("=" * 50)
    print("Mode changes will show as DIFFERENT COLORS")
    print(">>> CLICK THE GRAPH WINDOW, then press 0-9 <<<")
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
