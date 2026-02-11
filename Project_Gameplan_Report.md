# EE297B Research Project: Signal Processing Fungi Propagation
## Project Gameplan & Technical Blueprint
### Anthony Contreras & Alex Wong | San Jose State University

**Date:** January 28, 2026
**Deadline:** May 1, 2026 (~13 weeks remaining)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Timeline Assessment](#3-timeline-assessment)
4. [Reference Verification](#4-reference-verification)
5. [Hardware Design: Signal Simulator](#5-hardware-design-signal-simulator)
6. [Shopping List](#6-shopping-list)
7. [4-Day Action Plan](#7-4-day-action-plan)
8. [Software Architecture](#8-software-architecture)
9. [Code Templates](#9-code-templates)
10. [Risk Assessment](#10-risk-assessment)
11. [Task Division](#11-task-division)

---

## 1. Executive Summary

We have **13 weeks** to complete a 16-week project plan. The critical path bottleneck is **biology** - Pleurotus mycelium takes 2-4 weeks to colonize the FPC electrodes.

**Key Strategy:** Decouple software development from biological timeline by building a **voltage divider signal simulator** that mimics fungal electrical signals (0.5-2.1 mV, 0.01-1 Hz). This allows us to develop the entire signal processing pipeline while waiting for mycelium growth.

**Immediate Actions:**
- Order Pleurotus spawn TODAY
- Order AD8237 instrumentation amplifier
- Build voltage divider simulator on breadboard
- Develop Python/Arduino data acquisition pipeline

---

## 2. Project Overview

### Objectives (from proposal)

1. Demonstrate fungal-based bioelectrical sensing with Pleurotus ostreatus
2. Collect and create labeled datasets of fungal bioelectrical responses
3. Develop real-time signal processing pipeline using STFT and machine learning
4. Design closed-loop environmental control system with Arduino actuators
5. Compare open-loop and closed-loop system performance
6. Complete technical report with results and analysis

### Target Signal Characteristics

| Parameter | Value | Source |
|-----------|-------|--------|
| Amplitude | 0.5 - 2.1 mV peak | Adamatzky et al., Buffi et al. |
| Frequency | 0.01 - 1 Hz | Literature consensus |
| Waveform | Irregular spikes/oscillations | Not sinusoidal |
| DC offset | Variable (electrode drift) | Expected behavior |

### Hardware Stack

| Component | Model | Purpose |
|-----------|-------|---------|
| Sensing PCB | Custom FPC with ENIG finish | Electrode interface |
| Amplifier | AD8237 | Instrumentation amplifier (G=1000) |
| Microcontroller | Arduino Uno R4 | ADC + actuator control |
| Actuators | Mister, fan, LED | Environmental control |
| Shielding | Faraday cage | Noise rejection |

---

## 3. Timeline Assessment

### Original Plan: 16 weeks
### Available Time: 13 weeks (Jan 27 - May 1)

### Biology Bottleneck

| Stage | Duration |
|-------|----------|
| Order spawn, shipping | 3-7 days |
| Substrate colonization on FPC | 2-4 weeks |
| Stabilization before recording | 3-7 days |
| **Minimum bio-ready time** | **4-6 weeks** |

**Implication:** First live recordings likely mid-March at earliest.

### Revised Timeline

| Week | Dates | Activities |
|------|-------|------------|
| 1-2 | Jan 27 - Feb 9 | Order spawn. Build Faraday cage. Assemble amplifier circuit. **Build voltage divider simulator.** Verify full signal chain. |
| 3-4 | Feb 10 - Feb 23 | Develop Python/MATLAB acquisition + STFT pipeline. Inoculate FPCs. Start Arduino actuator code. |
| 5-6 | Feb 24 - Mar 9 | Continue software pipeline. Monitor FPC colonization. Test actuator hardware. |
| 7-8 | Mar 10 - Mar 23 | **First live recordings.** Compare to simulated baseline. Debug electrode interface. |
| 9-10 | Mar 24 - Apr 6 | Data collection under controlled stimuli. Build labeled dataset. Open-loop characterization. |
| 11-12 | Apr 7 - Apr 20 | Implement closed-loop control. Test feedback algorithms. Compare open vs closed loop. |
| 13 | Apr 21 - May 1 | Final evaluation. Write report. Prepare presentation. |

---

## 4. Reference Verification

All key references from the proposal have been verified:

| Reference | Status | Notes |
|-----------|--------|-------|
| Buffi et al. (2025) iScience | **Confirmed** | Open access. FPC design, ENIG electrodes, STFT analysis verified. |
| Buffi et al. (2025) FEMS Microbiol Rev | **Confirmed** | Comprehensive review of fungal electrophysiology. |
| Schyck et al. (2024) Global Challenges | **Confirmed** | Fungal signaling in living composites. |
| AD8237 specifications | **Confirmed** | Zero-drift, 106 dB CMRR, suitable for mV signals. |
| Pleurotus signal amplitude 0.5-2.1 mV | **Confirmed** | Consistent with Adamatzky group publications. |

### Important Note on Wood Wide Web

The "Wood Wide Web" concept (mycorrhizal network communication) is **scientifically contested**. A 2023 Nature Ecology & Evolution paper challenged core claims. For our project:

- **Do:** Frame as "exploring fungal electrical signals"
- **Don't:** Claim to demonstrate ecosystem-wide communication
- **Future work:** Mycorrhizal species experiments can be proposed but not promised

---

## 5. Hardware Design: Signal Simulator

### Purpose

Generate ~1 mV signals at 0.01-1 Hz to simulate fungal bioelectrical activity. This allows full pipeline development without waiting for biology.

### Circuit Schematic

```
                                    +5V (Arduino)
                                     |
                              +------+------+
                              |             |
                           [10k]         [10k]
                              |             |
                              +------+------+
                                     |
                              REF (2.5V)----+
                                            |
                                     +------+------+
                                     |             |
Arduino D9 (PWM)                     |          [0.1uF]
      |                              |             |
   [4.7M]                            |            GND
      |                              |
      +----[1k]----GND               |
      |                              |
   Signal                            |
   (~1mV)                            |
      |                              |
      +--------> +IN (pin 2)         |
                    |                |
                 AD8237              |
                    |                |
                 -IN (pin 1)---GND   |
                    |                |
                   FB (pin 3)        |
                    |                |
                 [R1=100]            |
                    |                |
                   RG (pin 5)        |
                    |                |
                [R2=100k]------------+
                    |                |
                   REF (pin 6)-------+
                    |
                  OUT (pin 7)-------> Arduino A0
                    |
                  +VS (pin 8)----+5V
                  -VS (pin 4)----GND
```

### AD8237 Pin Reference

```
        +-------U-------+
   -IN  | 1           8 | +VS
   +IN  | 2           7 | OUT
    FB  | 3           6 | REF
   -VS  | 4           5 | RG
        +---------------+
```

### Gain Calculation

```
Gain = 1 + (R2 / R1)
     = 1 + (100kΩ / 100Ω)
     = 1001

Input: 1 mV  →  Output: ~1 V (centered at 2.5V reference)
Output range: 1.5V to 3.5V (well within 0-5V ADC range)
```

### Voltage Divider Math

```
Vout = Vin × (R2 / (R1 + R2))
     = 5V × (1kΩ / (4.7MΩ + 1kΩ))
     = 5V × (1k / 4.701M)
     ≈ 1.06 mV peak

With PWM (0-255): Output ranges 0 to ~1 mV
```

---

## 6. Shopping List

### Priority 1: Order Immediately

| Item | Qty | Purpose | Est. Price | Source |
|------|-----|---------|------------|--------|
| **AD8237ARMZ** (MSOP-8) or **AD8237ARZ** (SOIC-8) | 2 | Instrumentation amplifier | $5-7 each | DigiKey, Mouser |
| **SOIC-8 to DIP adapter PCB** | 2 | Mount SMD on breadboard | $1-2 each | Amazon, DigiKey |
| **Pleurotus ostreatus spawn** | 1 bag | Biological component | $15-25 | Fungi Perfecti, North Spore |

### Priority 2: Check Inventory First

| Item | Qty | Purpose | Notes |
|------|-----|---------|-------|
| 4.7MΩ resistor (1/4W, 1%) | 5 | Voltage divider | May not have |
| 100kΩ resistor (1/4W, 1%) | 10 | Gain resistor R2 | Common value |
| 100Ω resistor (1/4W, 1%) | 10 | Gain resistor R1 | Common value |
| 10kΩ resistor (1/4W, 1%) | 10 | Reference divider | Common value |
| 1kΩ resistor (1/4W, 1%) | 10 | Divider low-side | Common value |
| 0.1µF ceramic capacitor | 10 | Bypass/filter | Common value |
| 10µF electrolytic | 5 | Power filtering | Common value |
| Arduino Uno | 1 | Controller | May have |
| Breadboard | 1 | Prototyping | May have |
| Jumper wires | 1 kit | Connections | May have |

### Priority 3: Faraday Cage (Week 2)

| Item | Qty | Purpose | Est. Price |
|------|-----|---------|------------|
| Metal ammo can or cookie tin | 1 | Shielded enclosure | $10-20 |
| BNC panel-mount connectors | 2-4 | Signal feedthrough | $3 each |
| Copper tape (conductive adhesive) | 1 roll | Seal gaps | $8 |
| Banana-to-alligator ground cable | 1 | Earth ground | $5 |

### Optional (Quality Improvements)

| Item | Purpose | Est. Price |
|------|---------|------------|
| USB isolator | Reduce ground loop noise | $15-25 |
| 9V battery + barrel jack | Clean isolated power | $5 |
| Shielded 2-conductor cable | Low-noise signal path | $5 |

### Estimated Total: $60-100

---

## 7. 4-Day Action Plan

### Day 1 (January 28-29)

#### Anthony
- [ ] Place orders: AD8237 + SOIC-to-DIP adapters (DigiKey/Mouser)
- [ ] Order Pleurotus spawn (Fungi Perfecti or North Spore)
- [ ] Inventory resistor/capacitor kit - list what's available
- [ ] Download AD8237 datasheet, read pages 1-12
- [ ] Create shared GitHub repo or Google Drive folder

#### Alex
- [ ] Set up Arduino IDE on development machine
- [ ] Write basic Arduino sketch: PWM on D9 + analogRead on A0 + Serial output
- [ ] Install Python environment: `pyserial`, `numpy`, `scipy`, `matplotlib`
- [ ] Write Python script: read serial, plot real-time data

### Day 2 (January 29-30)

#### Anthony
- [ ] Design PCB layout sketch for AD8237 circuit (for documentation)
- [ ] Research Faraday cage options - decide on enclosure
- [ ] Set up project documentation structure
- [ ] Review Buffi et al. iScience paper methodology section

#### Alex
- [ ] Implement STFT analysis: `scipy.signal.stft()`
- [ ] Test with synthetic signals (generate 0.1 Hz sine in Python)
- [ ] Add data logging: save timestamped readings to CSV
- [ ] Verify spectrogram output looks correct

### Day 3 (January 30-31)

#### Anthony
- [ ] Build voltage divider on breadboard (use available resistors)
- [ ] Measure output with multimeter - verify mV range
- [ ] Document circuit with photos
- [ ] Test divider output stability over 10 minutes

#### Alex
- [ ] Add bandpass filter: 0.01-2 Hz (fungal signal range)
- [ ] Implement spike detection (threshold crossing algorithm)
- [ ] Create dual visualization: time-domain + spectrogram
- [ ] Add configurable parameters (threshold, window size)

### Day 4 (January 31 - February 1)

#### Both Together
- [ ] Integrate hardware + software
- [ ] Connect voltage divider → Arduino A0 (no amp yet)
- [ ] Verify: PWM changes → divider → ADC → Python plot
- [ ] Measure baseline noise floor (inputs shorted)
- [ ] Document system performance
- [ ] Plan Week 2 tasks (amp integration)
- [ ] Update project timeline/Gantt chart

---

## 8. Software Architecture

### Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Fungal    │     │   AD8237    │     │   Arduino   │
│  Mycelium   │────>│ Amplifier   │────>│    ADC      │
│  (or sim)   │     │  (G=1000)   │     │  (10-bit)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               │ Serial (115200 baud)
                                               ▼
                                        ┌─────────────┐
                                        │   Python    │
                                        │  Pipeline   │
                                        └─────────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
             ┌─────────────┐           ┌─────────────┐           ┌─────────────┐
             │  Bandpass   │           │    STFT     │           │    Data     │
             │   Filter    │           │  Analysis   │           │   Logging   │
             │ 0.01-2 Hz   │           │             │           │    CSV      │
             └─────────────┘           └─────────────┘           └─────────────┘
                    │                          │
                    ▼                          ▼
             ┌─────────────┐           ┌─────────────┐
             │   Spike     │           │ Spectrogram │
             │ Detection   │           │    Plot     │
             └─────────────┘           └─────────────┘
                    │
                    ▼
             ┌─────────────┐
             │  Actuator   │
             │  Commands   │
             └─────────────┘
                    │
                    │ Serial
                    ▼
             ┌─────────────┐
             │   Arduino   │
             │  Actuators  │
             │ (mist/fan)  │
             └─────────────┘
```

### File Structure

```
EE297B_ResearchProject/
├── docs/
│   ├── Project_Gameplan_Report.md    (this document)
│   ├── MSEE Final Written Proposal.pdf
│   └── datasheets/
│       └── AD8237.pdf
├── hardware/
│   ├── schematics/
│   └── pcb/
├── firmware/
│   └── fungal_signal_acquisition/
│       └── fungal_signal_acquisition.ino
├── software/
│   ├── acquisition/
│   │   └── read_signal.py
│   ├── processing/
│   │   ├── stft_analysis.py
│   │   └── spike_detection.py
│   └── visualization/
│       └── live_plot.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── labeled/
└── results/
    └── figures/
```

---

## 9. Code Templates

### Arduino Firmware

```cpp
// fungal_signal_acquisition.ino
// EE297B Research Project - Signal Processing Fungi Propagation
// Anthony Contreras & Alex Wong

const int PWM_PIN = 9;        // Simulator output
const int ADC_PIN = A0;       // Signal input
const int MISTER_PIN = 7;     // Actuator control
const int FAN_PIN = 8;        // Actuator control
const int LED_PIN = 6;        // Actuator control

const int SAMPLE_PERIOD_MS = 100;  // 10 Hz sampling rate
unsigned long lastSampleTime = 0;

// Simulator state
int pwmValue = 0;
int pwmDirection = 1;
bool simulatorEnabled = true;

void setup() {
    Serial.begin(115200);

    pinMode(PWM_PIN, OUTPUT);
    pinMode(MISTER_PIN, OUTPUT);
    pinMode(FAN_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);

    analogReference(DEFAULT);  // 5V reference

    Serial.println("# Fungal Signal Acquisition System");
    Serial.println("# Format: timestamp_ms,adc_raw,voltage_mV");
}

void loop() {
    // Check for commands from Python
    if (Serial.available()) {
        char cmd = Serial.read();
        handleCommand(cmd);
    }

    // Sample at fixed rate
    unsigned long now = millis();
    if (now - lastSampleTime >= SAMPLE_PERIOD_MS) {
        lastSampleTime = now;

        // Update simulator (slow ramp)
        if (simulatorEnabled) {
            pwmValue += pwmDirection;
            if (pwmValue >= 255 || pwmValue <= 0) {
                pwmDirection *= -1;
            }
            analogWrite(PWM_PIN, pwmValue);
        }

        // Read and transmit
        int adcRaw = analogRead(ADC_PIN);
        float voltage_mV = (adcRaw / 1023.0) * 5000.0;

        Serial.print(now);
        Serial.print(",");
        Serial.print(adcRaw);
        Serial.print(",");
        Serial.println(voltage_mV, 3);
    }
}

void handleCommand(char cmd) {
    switch (cmd) {
        case 'M': digitalWrite(MISTER_PIN, HIGH); break;  // Mister ON
        case 'm': digitalWrite(MISTER_PIN, LOW); break;   // Mister OFF
        case 'F': digitalWrite(FAN_PIN, HIGH); break;     // Fan ON
        case 'f': digitalWrite(FAN_PIN, LOW); break;      // Fan OFF
        case 'L': digitalWrite(LED_PIN, HIGH); break;     // LED ON
        case 'l': digitalWrite(LED_PIN, LOW); break;      // LED OFF
        case 'S': simulatorEnabled = true; break;         // Simulator ON
        case 's': simulatorEnabled = false; break;        // Simulator OFF
    }
}
```

### Python Acquisition Script

```python
#!/usr/bin/env python3
"""
read_signal.py
EE297B Research Project - Signal Processing Fungi Propagation
Anthony Contreras & Alex Wong

Real-time acquisition and visualization of fungal bioelectrical signals.
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import stft, butter, filtfilt
from datetime import datetime
import csv
import os

# ============== CONFIGURATION ==============
SERIAL_PORT = '/dev/tty.usbmodem14101'  # Update for your system
# Windows: 'COM3', 'COM4', etc.
# Mac: '/dev/tty.usbmodem*'
# Linux: '/dev/ttyACM0', '/dev/ttyUSB0'

BAUD_RATE = 115200
SAMPLE_RATE = 10  # Hz (must match Arduino)
BUFFER_SECONDS = 60  # Rolling window size
BUFFER_SIZE = SAMPLE_RATE * BUFFER_SECONDS

# Filter settings
FILTER_LOW = 0.01   # Hz
FILTER_HIGH = 2.0   # Hz

# Spike detection
SPIKE_THRESHOLD_MV = 0.5  # mV above baseline

# Data logging
LOG_DIR = '../data/raw'
# ============================================


def create_bandpass_filter(lowcut, highcut, fs, order=2):
    """Create Butterworth bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Handle edge case for very low frequencies
    if low < 0.001:
        low = 0.001
    b, a = butter(order, [low, high], btype='band')
    return b, a


def detect_spikes(signal, threshold):
    """Simple threshold-based spike detection."""
    baseline = np.median(signal)
    spikes = np.where(signal > baseline + threshold)[0]
    return spikes, baseline


def main():
    # Setup data logging
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'recording_{timestamp}.csv')
    log_file = open(log_filename, 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['timestamp_ms', 'adc_raw', 'voltage_mV'])

    # Setup serial connection
    print(f"Connecting to {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print("Connected. Starting acquisition...")

    # Setup data buffers
    times = deque(maxlen=BUFFER_SIZE)
    voltages = deque(maxlen=BUFFER_SIZE)

    # Setup filter
    b, a = create_bandpass_filter(FILTER_LOW, FILTER_HIGH, SAMPLE_RATE)

    # Setup plot
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Fungal Bioelectrical Signal Acquisition', fontsize=14)

    ax_raw, ax_filtered, ax_stft = axes

    print(f"Logging to: {log_filename}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            # Read line from Arduino
            line = ser.readline().decode('utf-8').strip()

            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            try:
                parts = line.split(',')
                t_ms = int(parts[0])
                adc_raw = int(parts[1])
                voltage_mv = float(parts[2])

                # Store data
                times.append(t_ms / 1000.0)  # Convert to seconds
                voltages.append(voltage_mv)

                # Log to file
                csv_writer.writerow([t_ms, adc_raw, voltage_mv])

            except (ValueError, IndexError):
                continue

            # Update plots every 10 samples
            if len(voltages) >= 50 and len(voltages) % 10 == 0:
                t_arr = np.array(times)
                v_arr = np.array(voltages)

                # Normalize time to start at 0
                t_arr = t_arr - t_arr[0]

                # Plot 1: Raw signal
                ax_raw.clear()
                ax_raw.plot(t_arr, v_arr, 'b-', linewidth=0.5)
                ax_raw.set_ylabel('Voltage (mV)')
                ax_raw.set_title('Raw Signal')
                ax_raw.grid(True, alpha=0.3)

                # Plot 2: Filtered signal with spike detection
                if len(v_arr) > 15:  # Need enough samples for filter
                    try:
                        v_filtered = filtfilt(b, a, v_arr)
                        spikes, baseline = detect_spikes(v_filtered, SPIKE_THRESHOLD_MV)

                        ax_filtered.clear()
                        ax_filtered.plot(t_arr, v_filtered, 'g-', linewidth=0.5)
                        ax_filtered.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5)
                        ax_filtered.axhline(y=baseline + SPIKE_THRESHOLD_MV, color='r', linestyle='--', alpha=0.5)

                        if len(spikes) > 0:
                            ax_filtered.scatter(t_arr[spikes], v_filtered[spikes], c='red', s=20, zorder=5)

                        ax_filtered.set_ylabel('Voltage (mV)')
                        ax_filtered.set_title(f'Filtered ({FILTER_LOW}-{FILTER_HIGH} Hz) | Spikes: {len(spikes)}')
                        ax_filtered.grid(True, alpha=0.3)
                    except Exception:
                        pass

                # Plot 3: STFT Spectrogram
                if len(v_arr) >= 64:
                    try:
                        f, t_stft, Zxx = stft(v_arr, fs=SAMPLE_RATE, nperseg=32, noverlap=16)

                        ax_stft.clear()
                        ax_stft.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
                        ax_stft.set_ylabel('Frequency (Hz)')
                        ax_stft.set_xlabel('Time (s)')
                        ax_stft.set_title('STFT Spectrogram')
                        ax_stft.set_ylim([0, 2])  # Focus on 0-2 Hz
                    except Exception:
                        pass

                plt.tight_layout()
                plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n\nStopping acquisition...")
    finally:
        ser.close()
        log_file.close()
        print(f"Data saved to: {log_filename}")
        print(f"Total samples: {len(voltages)}")


if __name__ == '__main__':
    main()
```

---

## 10. Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mycelium doesn't colonize FPC electrodes | No live data | Start inoculation ASAP; prepare backup substrate approach |
| AD8237 noise floor too high | Can't detect 0.5 mV signals | Verify with simulator first; have INA121 as backup |
| Timeline slip | Incomplete deliverables | Weekly check-ins; scope reduction plan ready |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Electrode-mycelium interface impedance issues | Poor signal quality | Characterize impedance; adjust amp gain |
| Environmental noise in lab | Corrupted signals | Build proper Faraday cage; use differential measurement |
| Not enough labeled data for ML | Weak classifier | Fall back to threshold-based detection |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Arduino ADC resolution insufficient | Quantization noise | 10-bit is adequate for mV-range signals |
| Python real-time performance | Dropped samples | Reduce plot update rate; optimize code |

### Scope Reduction Plan (if needed)

If timeline pressure builds, reduce scope in this order:

1. **Drop:** TCN/ML classifier → use threshold-based detection instead
2. **Simplify:** Closed-loop control → demonstrate manual trigger response
3. **Reduce:** Multiple stimuli → focus on humidity only
4. **Last resort:** Reduce data collection period

---

## 11. Task Division

### Suggested Role Split

| Area | Owner | Responsibilities |
|------|-------|------------------|
| **Hardware** | Anthony | FPC handling, amplifier circuit, Faraday cage, actuators, breadboard builds |
| **Software** | Alex | Arduino firmware, Python pipeline, STFT analysis, visualization, data logging |
| **Biology** | Anthony | Spawn ordering, inoculation, growth monitoring, environmental conditions |
| **Documentation** | Both | Weekly updates, final report sections, presentation |

### Weekly Sync Schedule

- **Monday:** Brief async update (Slack/Discord)
- **Wednesday:** 30-min video call - progress check, blocker discussion
- **Friday:** Code/hardware review, planning next week

### Deliverable Ownership

| Deliverable | Primary | Support |
|-------------|---------|---------|
| Working signal acquisition chain | Alex | Anthony |
| FPC with colonized mycelium | Anthony | - |
| STFT analysis pipeline | Alex | - |
| Closed-loop controller | Both | - |
| Final report | Both | - |
| Presentation | Both | - |

---

## Appendix A: Quick Reference

### AD8237 Pinout (SOIC-8)

```
        +-------U-------+
   -IN  | 1           8 | +VS
   +IN  | 2           7 | OUT
    FB  | 3           6 | REF
   -VS  | 4           5 | RG
        +---------------+
```

### Gain Formula
```
G = 1 + (R2 / R1)
```

### Useful Commands

```bash
# Find Arduino port (Mac)
ls /dev/tty.usb*

# Find Arduino port (Linux)
ls /dev/ttyACM* /dev/ttyUSB*

# Install Python dependencies
pip install pyserial numpy scipy matplotlib

# Run acquisition script
python software/acquisition/read_signal.py
```

### Key Datasheets

- AD8237: https://www.analog.com/media/en/technical-documentation/data-sheets/ad8237.pdf
- Arduino Uno R4: https://docs.arduino.cc/hardware/uno-r4-minima

---

## Appendix B: Contact Information

| Name | Email | Phone |
|------|-------|-------|
| Anthony Contreras | anthony.r.contreras@sjsu.edu | (408) 373-4714 |
| Alex Wong | alex.a.wong@sjsu.edu | (510) 599-0460 |

---

*Document generated: January 28, 2026*
*Project deadline: May 1, 2026*
