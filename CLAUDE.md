# EE297B Research Project - Signal Processing for Fungi Propagation

## Team
- Anthony Contreras & Alex Wong, SJSU MSEE

## Project Goal
Measure bioelectrical signals from live oyster mushroom mycelium. Aspirational: full pipeline (detect/classify signals, closed-loop actuator control, STFT characterization). Will take what's feasible.

## Current Phase
Simulation/testing only — no mycelium inoculated yet.

## Hardware Signal Chain
```
[Voltage Divider] or [Electrodes]
        |
  INA128 Pin 3 (V_IN+)    ← signal input
  INA128 Pin 6 (VOUT)     ← signal output (CONFIRM: user may have A0 on wrong pin)
        |
  Arduino A0 (14-bit ADC, 10 Hz sampling)
```

### INA128 Instrumentation Amplifier
- **Gain resistor (RG):** 1kΩ between pins 1 and 8 → Gain = 51x
- **Pin 3:** Connected to voltage divider junction (1MΩ + 220Ω)
- **Pin 6 (VOUT):** Connected to Arduino A0 (confirmed)
- Gain formula: G = 1 + (50kΩ / RG)

### Voltage Divider (on breadboard)
- D9 (PWM) → 1MΩ → junction → 220Ω → GND
- Junction → INA128 pin 3 (non-inverting input)
- Output: ~1.1 mV from 5V PWM, ~56 mV after 51x amplification

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
- `software/processing/realtime_analyzer.py` — Live visualization, STFT, spike detection
- `software/processing/signal_processor.py` — Core DSP module (Buffi et al. STFT methodology)
- `software/processing/analyze_recording.py` — Offline CSV analysis
- `software/acquisition/read_signal.py` — Acquisition + CSV logging

## Signal Parameters (from literature)
- Frequency: 0.01–1 Hz
- Amplitude: 0.5–2.1 mV
- Sample rate: 10 Hz
- Bandpass filter: 0.01–2.0 Hz
- Spike threshold: baseline + 3σ

## Future Implementation (before mycelium inoculation)
- **Long-duration logger**: Headless script for 7+ day continuous recording
  - No matplotlib (memory leak risk) — CSV-only logging
  - Daily file rotation (new CSV each day)
  - Auto-reconnect on USB disconnect
  - Watchdog/auto-restart on crash
  - Run with `caffeinate -s` to prevent Mac sleep
  - Target: ~175 MB/week at 10 Hz
  - Analyze offline afterward with `analyze_recording.py`

## User Setup
- Mac (Darwin), basic soldering iron, AstroAI multimeter
- Arduino Uno R4, mini breadboard
