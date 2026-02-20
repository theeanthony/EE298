#!/bin/bash
# start_logger.sh
# EE297B Research Project — SJSU
# Anthony Contreras & Alex Wong
#
# Launches the 24/7 fungal signal logger with:
#   - caffeinate -s   prevents Mac from sleeping while on AC power
#   - auto-restart    restarts Python if it crashes (non-zero exit)
#   - clean stop      Ctrl+C exits without restarting
#
# Usage:
#   bash start_logger.sh                            # auto-detect Arduino port
#   bash start_logger.sh --port /dev/cu.usbmodem1401

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGGER="$SCRIPT_DIR/long_duration_logger.py"

echo "=========================================="
echo "  EE297B Fungal Signal Logger"
echo "  $(date)"
echo "  Ctrl+C to stop cleanly"
echo "  Status: $(dirname "$LOGGER")/../../data/raw/logger_status.txt"
echo "=========================================="

# Prevent Mac sleep while on AC power (-s = only while plugged in)
caffeinate -s -w $$ &
CAFFEINATE_PID=$!
echo "caffeinate running (PID $CAFFEINATE_PID)"

# On Ctrl+C: kill caffeinate, exit without restarting
trap 'echo ""; echo "$(date): Ctrl+C — stopping logger"; kill $CAFFEINATE_PID 2>/dev/null; exit 0' INT TERM

while true; do
    python3 "$LOGGER" "$@"
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "$(date): Logger stopped cleanly (exit 0)"
        kill $CAFFEINATE_PID 2>/dev/null
        exit 0
    fi

    echo "$(date): Logger exited with code $EXIT_CODE — restarting in 10s..."
    sleep 10
done
