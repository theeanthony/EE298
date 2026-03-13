# EE297B Research Project - Signal Processing for Fungi Propagation

## Team
- Anthony Contreras & Alex Wong, SJSU MSEE

## Project Goal
Measure bioelectrical signals from live oyster mushroom mycelium. Aspirational: full pipeline (detect/classify signals, closed-loop actuator control, STFT characterization). Will take what's feasible.

## Current Phase
**LIVE INOCULATION ACTIVE** — Second attempt Day 0: 2026-03-05.
(First attempt 2026-02-24 failed — LC missed pads, dried before colonization.)
Improved method: agar agar scaffold on each electrode pad, then LC on positive electrode per pair.
4× INA128 amps connected: A0=Pair2, A1=Pair3, A2=Pair6(ctrl), A3=Pair7(ctrl).
Baseline ~2378 mV all channels after agar equilibration.
Production logger running on Computer B (24/7), 14-column CSV format.
Signal changes expected at day 4+ (~2026-03-09) per Buffi et al.

## Hardware Signal Chain
```
PCB JP1 electrodes (differential pairs)
        |
  4× INA128 (one per pair) on breadboard
  Pin 3 (IN+) ← positive electrode
  Pin 2 (IN−) ← negative electrode
  Pin 6 (VOUT) → Arduino analog pin
        |
  Arduino A0=Pair2, A1=Pair3, A2=Pair6(ctrl), A3=Pair7(ctrl)
  14-bit ADC, 10 Hz sampling
```

### INA128 Instrumentation Amplifier (×4)
- **Gain resistor (RG):** 1kΩ between pins 1 and 8 → Gain = 51x
- **Pin 6 (VOUT):** Connected to Arduino analog pin (confirmed)
- Gain formula: G = 1 + (50kΩ / RG)
- Decoupling caps: 0.1 µF pin 7 (V+) → GND, 0.1 µF pin 4 (V−) → GND
- REF pin (pin 5): ~2.23V mid-rail via 10kΩ+10kΩ divider from +5V rail
  (enables bipolar output swing; bandpass filter removes DC offset in post-processing)
- Computer B runs on laptop battery (wall charger unplugged) — reduces switching noise

### Electrode PCB
- JP1: **14 pins** (not 12), single column, 2.54mm pitch
- JP1 routes **PAIR 2–7** (6 differential pairs)
- PAIR 1 and PAIR 8 route only to DB-25 (unpopulated, unusable)
- DB-25 footprint left empty — Arduino-only setup

### Actuators (in fungal_signal_acquisition.ino)
- D6: LED (PWM brightness control)
- D7: Mister relay
- D8: Fan relay

## Firmware
- `firmware/mycelium_signal_simulator/` — 11 signal modes for pipeline testing
- `firmware/fungal_signal_acquisition/` — Real acquisition + actuator control

## Python Software

### Hardware / Acquisition
- `software/processing/realtime_analyzer.py` — Live visualization, STFT, spike detection
- `software/processing/signal_processor.py` — Core DSP module (Buffi et al. STFT methodology)
- `software/processing/analyze_recording.py` — Offline CSV analysis
- `software/acquisition/read_signal.py` — Acquisition + CSV logging

### ML Pipeline (self-contained under `ml/`)
- See `ml/ML_REFERENCE.md` for full documentation
- `ml/train_tcn.py` — 3-phase TCN training (binary + vocabulary modes)
- `ml/train.py` — Classical ML training (RF + SVM)
- `ml/spike_vocabulary.py` — Adamatzky spike clustering → word types
- `ml/build_dictionary.py` — Map word types → stimulus meanings
- `ml/colab_vocabulary_pipeline.ipynb` — Full pipeline for Google Colab

## Signal Parameters (from literature)
- Frequency: 0.01–1 Hz
- Amplitude: 0.5–2.1 mV
- Sample rate: 10 Hz
- Bandpass filter: 0.01–2.0 Hz
- Spike threshold: baseline + 3σ

## Long-Duration Logger (implemented, running)
- `software/acquisition/long_duration_logger.py` — 14-column CSV, 4-channel
- Run: `nohup caffeinate -s python3 .../long_duration_logger.py --port /dev/cu.usbmodemF412FA6FA0802 --pair-id 2 --adc-bits 14 --gain 51 > logger_stdout.log 2>&1 &`
- Daily file rotation, auto-reconnect, heartbeat every 60s to `logger_status.txt`
- Monitor: `tail -f ~/EE298/data/raw/logger_stdout.log`
- Status: `ssh anthonycontreras@192.168.12.103 "cat ~/EE298/data/raw/logger_status.txt"`
- Sync data: `rsync -av --progress "anthonycontreras@192.168.12.103:~/EE298/data/raw/recording_*.csv" ~/Downloads/mycelium_data/`

## Offline Analysis
- `software/processing/analyze_recording.py` — 4-channel analysis, rewritten 2026-03-12
- Generates: overview plot, inoculated comparison, filtered signals, PSD
- Run: `python3 software/processing/analyze_recording.py ~/Downloads/mycelium_data/recording_2026030*.csv --output-dir ~/Downloads/mycelium_data/analysis --start-date 2026-03-05`

## User Setup
- Mac (Darwin), basic soldering iron, AstroAI multimeter
- Arduino Uno R4, mini breadboard
