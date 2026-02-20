#!/usr/bin/env python3
"""
long_duration_logger.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong | San Jose State University

Headless 24/7 logger for fungal bioelectrical signals.
No matplotlib — CSV only, safe for multi-day unattended runs.

Features:
  - Daily file rotation (new CSV at midnight)
  - Auto-reconnect on USB disconnect
  - Heartbeat status file updated every 60s
  - ~175 MB/week at 10 Hz

Usage:
    python long_duration_logger.py                       # Auto-detect Arduino port
    python long_duration_logger.py --port /dev/cu.usbmodem1401

Launch via start_logger.sh for caffeinate (no sleep) + auto-restart on crash.

Output files (data/raw/):
    recording_YYYYMMDD.csv   — daily data
    logger_status.txt        — heartbeat (updated every 60s, check remotely via SSH)
    logger.log               — event log
"""

import serial
import serial.tools.list_ports
import csv
import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime, date


# ============================================================
# CONFIG
# ============================================================
BAUD_RATE        = 115200
SERIAL_TIMEOUT   = 2.0   # seconds — long enough to detect disconnects
RECONNECT_DELAY  = 5     # seconds between reconnect attempts
STATUS_INTERVAL  = 60    # seconds between heartbeat writes
FLUSH_INTERVAL   = 100   # samples between explicit CSV flushes (every 10s at 10 Hz)

LOG_DIR     = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')
LOG_FILE    = os.path.join(LOG_DIR, 'logger.log')
STATUS_FILE = os.path.join(LOG_DIR, 'logger_status.txt')


# ============================================================
# LOGGING
# ============================================================
def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-7s  %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


# ============================================================
# PORT DETECTION
# ============================================================
def find_arduino_port():
    keywords = ['arduino', 'usbmodem', 'usbserial', 'acm', 'ch340', 'cp210']
    for port in serial.tools.list_ports.comports():
        desc = (port.device + ' ' + str(port.description)).lower()
        if any(k in desc for k in keywords):
            return port.device
    ports = serial.tools.list_ports.comports()
    if ports:
        return ports[0].device
    return None


# ============================================================
# CSV MANAGEMENT
# ============================================================
def csv_path(d):
    return os.path.join(LOG_DIR, f'recording_{d.strftime("%Y%m%d")}.csv')


def open_csv(d):
    path = csv_path(d)
    is_new = not os.path.exists(path)
    f = open(path, 'a', newline='')
    writer = csv.writer(f)
    if is_new:
        writer.writerow(['timestamp_ms', 'adc_raw', 'voltage_mV'])
    logging.info(f"CSV {'created' if is_new else 'reopened'}: {os.path.basename(path)}")
    return f, writer, d


# ============================================================
# STATUS HEARTBEAT
# ============================================================
def write_status(start_time, sample_count, current_date, connected):
    elapsed = time.time() - start_time
    rate = sample_count / elapsed if elapsed > 0 else 0.0
    path = csv_path(current_date)
    size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0
    lines = [
        f"updated:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"connected:    {connected}",
        f"samples:      {sample_count:,}",
        f"elapsed_h:    {elapsed / 3600:.2f}",
        f"avg_rate_hz:  {rate:.2f}",
        f"current_file: {os.path.basename(path)}",
        f"file_size_mb: {size_mb:.1f}",
    ]
    try:
        with open(STATUS_FILE, 'w') as f:
            f.write('\n'.join(lines) + '\n')
    except OSError:
        pass


# ============================================================
# MAIN LOOP
# ============================================================
def run(port_name):
    setup_logging()
    logging.info('=' * 50)
    logging.info('Fungal signal long-duration logger starting')
    logging.info(f'Port:       {port_name}')
    logging.info(f'Output dir: {LOG_DIR}')
    logging.info('Ctrl+C to stop cleanly. Crashes auto-restart via start_logger.sh.')
    logging.info('=' * 50)

    os.makedirs(LOG_DIR, exist_ok=True)

    # Clean shutdown on SIGINT / SIGTERM → exit(0) so bash loop does NOT restart
    shutdown = {'flag': False}

    def handle_signal(sig, frame):
        logging.info('Shutdown signal received — stopping cleanly')
        shutdown['flag'] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    start_time = time.time()
    sample_count = 0
    last_status_time = 0

    today = date.today()
    csv_file, csv_writer, csv_date = open_csv(today)

    ser = None
    connected = False

    while not shutdown['flag']:

        # ---- (Re)connect ----
        if not connected:
            logging.info(f'Connecting to {port_name}...')
            try:
                ser = serial.Serial(port_name, BAUD_RATE, timeout=SERIAL_TIMEOUT)
                time.sleep(2)           # wait for Arduino reset
                ser.reset_input_buffer()
                connected = True
                logging.info('Connected')
            except serial.SerialException as e:
                logging.warning(f'Connect failed ({e}) — retry in {RECONNECT_DELAY}s')
                time.sleep(RECONNECT_DELAY)
                continue

        # ---- Read one line ----
        try:
            raw = ser.readline()
        except serial.SerialException as e:
            logging.warning(f'Read error ({e}) — reconnecting')
            try:
                ser.close()
            except Exception:
                pass
            connected = False
            continue

        if not raw:
            # Timeout with no data — Arduino may have disconnected
            logging.warning('No data received (timeout) — reconnecting')
            try:
                ser.close()
            except Exception:
                pass
            connected = False
            continue

        try:
            line = raw.decode('utf-8', errors='replace').strip()
        except Exception:
            continue

        # Skip comments and blank lines
        if not line or line.startswith('#'):
            if line.startswith('#'):
                logging.info(f'Arduino: {line}')
            continue

        # ---- Parse: timestamp_ms,adc_raw,voltage_mV ----
        parts = line.split(',')
        if len(parts) < 3:
            continue
        try:
            t_ms      = int(parts[0])
            adc_raw   = int(parts[1])
            voltage_mv = float(parts[2])
        except ValueError:
            continue

        # ---- Daily rotation ----
        today = date.today()
        if today != csv_date:
            csv_file.flush()
            csv_file.close()
            logging.info('Midnight rollover — opening new CSV')
            csv_file, csv_writer, csv_date = open_csv(today)

        # ---- Write ----
        csv_writer.writerow([t_ms, adc_raw, f'{voltage_mv:.4f}'])
        sample_count += 1

        if sample_count % FLUSH_INTERVAL == 0:
            csv_file.flush()

        # ---- Heartbeat ----
        now = time.time()
        if now - last_status_time >= STATUS_INTERVAL:
            write_status(start_time, sample_count, csv_date, connected)
            last_status_time = now
            logging.info(
                f'Status: {sample_count:,} samples | '
                f'{(now - start_time) / 3600:.1f}h elapsed | '
                f'{sample_count / (now - start_time):.2f} Hz avg'
            )

    # ---- Clean shutdown ----
    logging.info(f'Stopping — {sample_count:,} total samples recorded')
    csv_file.flush()
    csv_file.close()
    if ser and ser.is_open:
        ser.close()
    write_status(start_time, sample_count, csv_date, False)
    logging.info('Logger stopped cleanly')
    sys.exit(0)   # exit 0 → start_logger.sh will NOT restart


# ============================================================
# ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='24/7 headless fungal signal logger')
    parser.add_argument('--port', '-p', type=str, default=None,
                        help='Serial port (e.g. /dev/cu.usbmodem1401). '
                             'Auto-detected if not specified.')
    args = parser.parse_args()

    port = args.port or find_arduino_port()
    if not port:
        print('ERROR: No serial port found. Plug in the Arduino or use --port.')
        sys.exit(1)

    run(port)


if __name__ == '__main__':
    main()
