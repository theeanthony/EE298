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
  - Metadata header in every new CSV (experiment params, UTC start time)
  - Wall-clock UTC timestamp column (stable across reboots)
  - Sequence counter for detecting dropped samples
  - GAP comment row written on reconnect (timestamps the outage)
  - Heartbeat status file updated every 60s
  - Parses 6-column Arduino output: timestamp_ms,adc_raw,voltage_mV,mister,fan,led

Usage:
    python long_duration_logger.py                       # Auto-detect Arduino port
    python long_duration_logger.py --port /dev/cu.usbmodem1401
    python long_duration_logger.py --pair-id 3 --adc-bits 14 --gain 51

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
from datetime import datetime, date, timezone


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

CSV_COLUMNS = ['seq', 'wall_clock_utc', 'timestamp_ms', 'adc_raw',
               'voltage_mV', 'mister', 'fan', 'led']


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


def write_metadata_header(f, pair_id, adc_bits, gain):
    """Write # comment lines at the top of a new CSV file."""
    lines = [
        f"# experiment: mycelium_colonization_001",
        f"# start_utc: {datetime.now(timezone.utc).isoformat(timespec='milliseconds')}",
        f"# sample_rate_hz: 10",
        f"# adc_bits: {adc_bits}",
        f"# gain: {gain}",
        f"# pair_id: {pair_id}",
        f"# active_pairs: 2,3,4,5",
        f"# control_pairs: 6,7",
    ]
    for line in lines:
        f.write(line + '\n')


def open_csv(d, pair_id, adc_bits, gain):
    path = csv_path(d)
    is_new = not os.path.exists(path)
    f = open(path, 'a', newline='')
    if is_new:
        write_metadata_header(f, pair_id, adc_bits, gain)
        f.flush()
    writer = csv.writer(f)
    if is_new:
        writer.writerow(CSV_COLUMNS)
    logging.info(f"CSV {'created' if is_new else 'reopened'}: {os.path.basename(path)}")
    return f, writer, d


# ============================================================
# STATUS HEARTBEAT
# ============================================================
def write_status(start_time, sample_count, current_date, connected, pair_id):
    elapsed = time.time() - start_time
    rate = sample_count / elapsed if elapsed > 0 else 0.0
    path = csv_path(current_date)
    size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0
    lines = [
        f"updated:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"connected:    {connected}",
        f"pair_id:      {pair_id}",
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
# LINE PARSER
# ============================================================
def parse_line(line):
    """
    Parse a data row from the Arduino.

    Accepts both old 3-column format (timestamp_ms,adc_raw,voltage_mV) and
    new 6-column format (timestamp_ms,adc_raw,voltage_mV,mister,fan,led).

    Returns (t_ms, adc_raw, voltage_mv, mister, fan, led) or None on failure.
    """
    parts = line.split(',')
    if len(parts) < 3:
        return None
    try:
        t_ms       = int(parts[0])
        adc_raw    = int(parts[1])
        voltage_mv = float(parts[2])
        mister     = int(parts[3]) if len(parts) >= 4 else 0
        fan        = int(parts[4]) if len(parts) >= 5 else 0
        led        = int(parts[5]) if len(parts) >= 6 else 0
        return t_ms, adc_raw, voltage_mv, mister, fan, led
    except (ValueError, IndexError):
        return None


# ============================================================
# MAIN LOOP
# ============================================================
def run(port_name, pair_id, adc_bits, gain):
    setup_logging()
    logging.info('=' * 50)
    logging.info('Fungal signal long-duration logger starting')
    logging.info(f'Port:       {port_name}')
    logging.info(f'Pair ID:    {pair_id}')
    logging.info(f'ADC bits:   {adc_bits}')
    logging.info(f'Gain:       {gain}x')
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
    seq = 0
    last_status_time = 0
    disconnect_time = None   # wall time of last disconnect (for gap duration)

    today = date.today()
    csv_file, csv_writer, csv_date = open_csv(today, pair_id, adc_bits, gain)

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

                # Write GAP row if this is a reconnect (not first connect)
                if disconnect_time is not None:
                    reconnect_utc = datetime.now(timezone.utc).isoformat(timespec='milliseconds')
                    gap_duration = time.time() - disconnect_time
                    gap_line = (
                        f"# GAP: disconnected ~{gap_duration:.0f}s, "
                        f"seq_before={seq}, "
                        f"reconnected_at={reconnect_utc}"
                    )
                    csv_file.write(gap_line + '\n')
                    csv_file.flush()
                    logging.info(gap_line)

                disconnect_time = None
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
            disconnect_time = time.time()
            continue

        if not raw:
            # Timeout with no data — Arduino may have disconnected
            logging.warning('No data received (timeout) — reconnecting')
            try:
                ser.close()
            except Exception:
                pass
            connected = False
            disconnect_time = time.time()
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

        # ---- Parse ----
        parsed = parse_line(line)
        if parsed is None:
            continue
        t_ms, adc_raw, voltage_mv, mister, fan, led = parsed

        # ---- Daily rotation ----
        today = date.today()
        if today != csv_date:
            csv_file.flush()
            csv_file.close()
            logging.info('Midnight rollover — opening new CSV')
            csv_file, csv_writer, csv_date = open_csv(today, pair_id, adc_bits, gain)

        # ---- Wall-clock UTC timestamp ----
        wall_utc = datetime.now(timezone.utc).isoformat(timespec='milliseconds')

        # ---- Write ----
        seq += 1
        csv_writer.writerow([seq, wall_utc, t_ms, adc_raw,
                             f'{voltage_mv:.4f}', mister, fan, led])
        sample_count += 1

        if sample_count % FLUSH_INTERVAL == 0:
            csv_file.flush()

        # ---- Heartbeat ----
        now = time.time()
        if now - last_status_time >= STATUS_INTERVAL:
            write_status(start_time, sample_count, csv_date, connected, pair_id)
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
    write_status(start_time, sample_count, csv_date, False, pair_id)
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
    parser.add_argument('--pair-id', type=int, default=2,
                        help='Electrode pair being recorded (default: 2). '
                             'Written to CSV metadata header.')
    parser.add_argument('--adc-bits', type=int, default=14,
                        help='ADC resolution in bits (default: 14 for Uno R4). '
                             'Written to CSV metadata header.')
    parser.add_argument('--gain', type=int, default=51,
                        help='INA128 amplifier gain (default: 51 for RG=1kΩ). '
                             'Written to CSV metadata header.')
    args = parser.parse_args()

    port = args.port or find_arduino_port()
    if not port:
        print('ERROR: No serial port found. Plug in the Arduino or use --port.')
        sys.exit(1)

    run(port, args.pair_id, args.adc_bits, args.gain)


if __name__ == '__main__':
    main()
