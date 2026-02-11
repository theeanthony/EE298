#!/usr/bin/env python3
"""
read_signal.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Real-time acquisition and visualization of fungal bioelectrical signals.

Usage:
    python read_signal.py                    # Auto-detect port
    python read_signal.py --port COM3        # Specify port (Windows)
    python read_signal.py --port /dev/ttyACM0  # Specify port (Linux)
    python read_signal.py --no-plot          # Logging only, no visualization

Requirements:
    pip install pyserial numpy scipy matplotlib
"""

import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import stft, butter, filtfilt
from datetime import datetime
import csv
import os
import sys
import argparse
import time


# ============== CONFIGURATION ==============
class Config:
    # Serial settings
    BAUD_RATE = 115200
    SERIAL_TIMEOUT = 1.0

    # Sampling
    SAMPLE_RATE = 10  # Hz (must match Arduino)
    BUFFER_SECONDS = 60  # Rolling window size
    BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

    # Filter settings (bandpass for fungal signals)
    FILTER_LOW = 0.05   # Hz (lower bound)
    FILTER_HIGH = 2.0   # Hz (upper bound)
    FILTER_ORDER = 2

    # Spike detection
    SPIKE_THRESHOLD_MV = 0.3  # mV above baseline

    # Visualization
    PLOT_UPDATE_INTERVAL = 10  # Update plot every N samples
    SPECTROGRAM_NPERSEG = 32
    SPECTROGRAM_NOVERLAP = 24

    # Data logging
    LOG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')


def find_arduino_port():
    """Auto-detect Arduino serial port."""
    ports = serial.tools.list_ports.comports()

    # Common Arduino identifiers
    arduino_keywords = ['arduino', 'usbmodem', 'usbserial', 'acm', 'ch340', 'cp210']

    for port in ports:
        port_lower = (port.device + ' ' + str(port.description)).lower()
        for keyword in arduino_keywords:
            if keyword in port_lower:
                print(f"Found Arduino on: {port.device} ({port.description})")
                return port.device

    # List all ports if no Arduino found
    print("Could not auto-detect Arduino. Available ports:")
    for port in ports:
        print(f"  {port.device}: {port.description}")

    if ports:
        print(f"\nTrying first available port: {ports[0].device}")
        return ports[0].device

    return None


def create_bandpass_filter(lowcut, highcut, fs, order=2):
    """Create Butterworth bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)  # Avoid zero
    high = min(highcut / nyq, 0.999)  # Avoid Nyquist

    if low >= high:
        print(f"Warning: Invalid filter bounds ({lowcut}-{highcut} Hz). Using defaults.")
        low, high = 0.01, 0.4

    b, a = butter(order, [low, high], btype='band')
    return b, a


def detect_spikes(signal, threshold):
    """
    Simple threshold-based spike detection.

    Returns indices of samples exceeding baseline + threshold.
    """
    if len(signal) < 10:
        return np.array([]), 0

    baseline = np.median(signal)
    above_threshold = signal > (baseline + threshold)
    spike_indices = np.where(above_threshold)[0]

    return spike_indices, baseline


class SignalAcquisition:
    def __init__(self, port, enable_plot=True):
        self.port = port
        self.enable_plot = enable_plot
        self.ser = None
        self.log_file = None
        self.csv_writer = None

        # Data buffers
        self.times = deque(maxlen=Config.BUFFER_SIZE)
        self.voltages = deque(maxlen=Config.BUFFER_SIZE)
        self.sample_count = 0

        # Filter coefficients
        self.b, self.a = create_bandpass_filter(
            Config.FILTER_LOW,
            Config.FILTER_HIGH,
            Config.SAMPLE_RATE,
            Config.FILTER_ORDER
        )

        # Statistics
        self.start_time = None
        self.total_spikes = 0

    def setup_logging(self):
        """Initialize CSV logging."""
        os.makedirs(Config.LOG_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(Config.LOG_DIR, f'recording_{timestamp}.csv')

        self.log_file = open(log_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow(['timestamp_ms', 'adc_raw', 'voltage_mV'])

        print(f"Logging to: {log_filename}")
        return log_filename

    def setup_serial(self):
        """Initialize serial connection."""
        print(f"Connecting to {self.port} at {Config.BAUD_RATE} baud...")

        try:
            self.ser = serial.Serial(
                self.port,
                Config.BAUD_RATE,
                timeout=Config.SERIAL_TIMEOUT
            )
            time.sleep(2)  # Wait for Arduino reset

            # Clear any startup messages
            self.ser.reset_input_buffer()

            print("Connected successfully!")
            return True

        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            return False

    def setup_plot(self):
        """Initialize matplotlib figure."""
        if not self.enable_plot:
            return

        plt.ion()
        self.fig, self.axes = plt.subplots(3, 1, figsize=(12, 8))
        self.fig.suptitle('Fungal Bioelectrical Signal Acquisition', fontsize=14)

        self.ax_raw = self.axes[0]
        self.ax_filtered = self.axes[1]
        self.ax_stft = self.axes[2]

        plt.tight_layout()

    def parse_line(self, line):
        """Parse a line of serial data."""
        # Skip comments and empty lines
        if not line or line.startswith('#'):
            if line.startswith('#'):
                print(line)  # Print Arduino status messages
            return None

        try:
            parts = line.split(',')
            if len(parts) >= 3:
                t_ms = int(parts[0])
                adc_raw = int(parts[1])
                voltage_mv = float(parts[2])
                return t_ms, adc_raw, voltage_mv
        except (ValueError, IndexError):
            pass

        return None

    def update_plot(self):
        """Update visualization."""
        if not self.enable_plot:
            return

        if len(self.voltages) < 50:
            return

        t_arr = np.array(self.times)
        v_arr = np.array(self.voltages)

        # Normalize time to start at 0
        t_arr = t_arr - t_arr[0]

        # ===== Plot 1: Raw Signal =====
        self.ax_raw.clear()
        self.ax_raw.plot(t_arr, v_arr, 'b-', linewidth=0.5, alpha=0.8)
        self.ax_raw.set_ylabel('Voltage (mV)')
        self.ax_raw.set_title('Raw ADC Signal')
        self.ax_raw.grid(True, alpha=0.3)

        # ===== Plot 2: Filtered + Spike Detection =====
        if len(v_arr) > 15:
            try:
                # Apply bandpass filter
                v_filtered = filtfilt(self.b, self.a, v_arr)

                # Detect spikes
                spikes, baseline = detect_spikes(v_filtered, Config.SPIKE_THRESHOLD_MV)

                self.ax_filtered.clear()
                self.ax_filtered.plot(t_arr, v_filtered, 'g-', linewidth=0.5, alpha=0.8)
                self.ax_filtered.axhline(y=baseline, color='gray', linestyle='--',
                                          alpha=0.5, label='Baseline')
                self.ax_filtered.axhline(y=baseline + Config.SPIKE_THRESHOLD_MV,
                                          color='r', linestyle='--', alpha=0.5,
                                          label='Threshold')

                if len(spikes) > 0:
                    self.ax_filtered.scatter(t_arr[spikes], v_filtered[spikes],
                                              c='red', s=30, zorder=5, label='Spikes')
                    self.total_spikes += len(spikes)

                self.ax_filtered.set_ylabel('Voltage (mV)')
                self.ax_filtered.set_title(
                    f'Filtered ({Config.FILTER_LOW}-{Config.FILTER_HIGH} Hz) | '
                    f'Spikes in window: {len(spikes)}'
                )
                self.ax_filtered.grid(True, alpha=0.3)
                self.ax_filtered.legend(loc='upper right', fontsize=8)

            except Exception as e:
                pass

        # ===== Plot 3: STFT Spectrogram =====
        if len(v_arr) >= Config.SPECTROGRAM_NPERSEG * 2:
            try:
                f, t_stft, Zxx = stft(
                    v_arr,
                    fs=Config.SAMPLE_RATE,
                    nperseg=Config.SPECTROGRAM_NPERSEG,
                    noverlap=Config.SPECTROGRAM_NOVERLAP
                )

                self.ax_stft.clear()
                self.ax_stft.pcolormesh(t_stft, f, np.abs(Zxx),
                                         shading='gouraud', cmap='viridis')
                self.ax_stft.set_ylabel('Frequency (Hz)')
                self.ax_stft.set_xlabel('Time (s)')
                self.ax_stft.set_title('STFT Spectrogram')
                self.ax_stft.set_ylim([0, Config.FILTER_HIGH + 0.5])

            except Exception as e:
                pass

        plt.tight_layout()
        plt.pause(0.01)

    def send_command(self, cmd):
        """Send a command to Arduino."""
        if self.ser and self.ser.is_open:
            self.ser.write(cmd.encode())
            print(f"Sent command: {cmd}")

    def run(self):
        """Main acquisition loop."""
        self.setup_logging()

        if not self.setup_serial():
            return

        self.setup_plot()
        self.start_time = time.time()

        print("\n" + "=" * 50)
        print("Acquisition started. Press Ctrl+C to stop.")
        print("Commands: M/m=mister, F/f=fan, L/l=LED, S/s=simulator, ?=status")
        print("=" * 50 + "\n")

        try:
            while True:
                # Read from serial
                try:
                    line = self.ser.readline().decode('utf-8').strip()
                except UnicodeDecodeError:
                    continue

                data = self.parse_line(line)
                if data is None:
                    continue

                t_ms, adc_raw, voltage_mv = data

                # Store in buffers
                self.times.append(t_ms / 1000.0)
                self.voltages.append(voltage_mv)
                self.sample_count += 1

                # Log to file
                self.csv_writer.writerow([t_ms, adc_raw, voltage_mv])

                # Update plot periodically
                if self.sample_count % Config.PLOT_UPDATE_INTERVAL == 0:
                    self.update_plot()

                    # Print periodic status
                    if self.sample_count % 100 == 0:
                        elapsed = time.time() - self.start_time
                        rate = self.sample_count / elapsed if elapsed > 0 else 0
                        print(f"Samples: {self.sample_count} | "
                              f"Rate: {rate:.1f} Hz | "
                              f"Latest: {voltage_mv:.2f} mV")

        except KeyboardInterrupt:
            print("\n\nStopping acquisition...")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        if self.ser and self.ser.is_open:
            self.ser.close()

        if self.log_file:
            self.log_file.close()

        elapsed = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 50)
        print("Acquisition Summary:")
        print(f"  Total samples: {self.sample_count}")
        print(f"  Duration: {elapsed:.1f} seconds")
        print(f"  Average rate: {self.sample_count / elapsed:.2f} Hz" if elapsed > 0 else "")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description='Fungal Bioelectrical Signal Acquisition'
    )
    parser.add_argument('--port', '-p', type=str, default=None,
                        help='Serial port (e.g., COM3, /dev/ttyACM0)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable real-time plotting')
    parser.add_argument('--list-ports', '-l', action='store_true',
                        help='List available serial ports and exit')

    args = parser.parse_args()

    if args.list_ports:
        ports = serial.tools.list_ports.comports()
        print("Available serial ports:")
        for port in ports:
            print(f"  {port.device}: {port.description}")
        return

    # Find port
    port = args.port if args.port else find_arduino_port()
    if not port:
        print("Error: No serial port found. Use --port to specify.")
        sys.exit(1)

    # Run acquisition
    acq = SignalAcquisition(port, enable_plot=not args.no_plot)
    acq.run()


if __name__ == '__main__':
    main()
