# Signal Processing Pipeline
## EE297B Research Project - Signal Processing Fungi Propagation
### Anthony Contreras & Alex Wong | San Jose State University

---

## Quick Start

### 1. Install Dependencies

```bash
pip install pyserial numpy scipy matplotlib pandas
```

### 2. Real-Time Analysis (with Arduino)

```bash
# Connect your Arduino with voltage divider running
python realtime_analyzer.py

# Or specify port
python realtime_analyzer.py --port /dev/cu.usbmodem14101  # Mac
python realtime_analyzer.py --port COM3                   # Windows

# Demo mode (no Arduino needed)
python realtime_analyzer.py --demo
```

### 3. Analyze Recorded Data

```bash
# Basic analysis
python analyze_recording.py path/to/recording.csv

# Save plots to folder
python analyze_recording.py recording.csv --output-dir ./results
```

---

## Files

| File | Description |
|------|-------------|
| `signal_processor.py` | Core signal processing module (filters, STFT, spike detection) |
| `realtime_analyzer.py` | Real-time visualization with Arduino |
| `analyze_recording.py` | Offline analysis of CSV recordings |

---

## Signal Processing Pipeline

### Filters Implemented

| Filter | Cutoff | Purpose |
|--------|--------|---------|
| **High-pass** | 0.01 Hz | Removes DC drift, electrode offset |
| **Low-pass** | 2.0 Hz | Removes high-frequency noise |
| **Bandpass** | 0.01-2.0 Hz | Isolates fungal signal band |
| **Notch** | 60 Hz | Removes mains interference (if fs > 120 Hz) |

### Why These Values?

Based on Buffi et al. 2025 and Adamatzky et al.:
- Fungal signals are **0.5-2.1 mV** amplitude
- Frequency range is **0.01-1 Hz** (very slow oscillations)
- Spike durations are **seconds to minutes** (not milliseconds like neurons)

### STFT Analysis

Short-Time Fourier Transform parameters:
- Window: Hanning
- Segment size: 64 samples
- Overlap: 75%
- FFT size: 128 (zero-padded)

---

## Usage Examples

### Using SignalProcessor as a Module

```python
from signal_processor import SignalProcessor
import numpy as np

# Create processor (10 Hz sample rate)
processor = SignalProcessor(sample_rate=10)

# Load your data
data = np.loadtxt('recording.csv', delimiter=',', skiprows=1)
voltage = data[:, 2]  # voltage_mV column

# Apply full processing pipeline
filtered = processor.process(voltage)

# Detect spikes
spikes, info = processor.detect_spikes(filtered)
print(f"Found {info['count']} spikes")
print(f"Spike rate: {info['rate_per_minute']:.2f} per minute")

# Get frequency analysis
bands = processor.get_frequency_bands(voltage)
print(f"Low frequency power: {bands['low']:.4e}")

# Generate plots
processor.plot_processing_pipeline(voltage)
processor.plot_spectrogram(voltage)
plt.show()
```

### Custom Filter Settings

```python
from signal_processor import SignalProcessor, FilterConfig

# Custom filter configuration
config = FilterConfig(
    sample_rate=10.0,
    highpass_cutoff=0.05,    # Higher HP to remove more drift
    lowpass_cutoff=1.0,      # Lower LP for cleaner signal
    bandpass_low=0.05,
    bandpass_high=1.0,
)

processor = SignalProcessor(filter_config=config)
```

---

## Understanding the Output

### Serial Data Format (from Arduino)

```
timestamp_ms,pwm,adc_raw,voltage_mV
100,128,45,1.374
200,129,46,1.404
...
```

### Real-Time Display

The analyzer shows:
1. **Raw Signal** - Direct ADC readings
2. **Filtered Signal** - After bandpass filter with spike detection
3. **STFT Spectrogram** - Frequency content over time
4. **Statistics** - Sample rate, signal stats, spike count

### Spike Detection

Spikes are detected when:
- Signal exceeds **baseline + 3×std**
- Minimum **0.5 seconds** between spikes
- Minimum **0.1 mV** prominence

---

## Troubleshooting

### "No serial port found"
```bash
# List available ports
python realtime_analyzer.py --list-ports

# Specify port manually
python realtime_analyzer.py --port /dev/ttyACM0
```

### "Insufficient data for STFT"
- Need at least 64 samples for STFT
- Wait ~6 seconds at 10 Hz sample rate

### Noisy spectrogram
- Check your electrical connections
- Ensure good grounding
- Consider adding shielding

---

## Integration with Arduino

The expected Arduino output format (from `mycelium_signal_simulator.ino`):

```
timestamp_ms,pwm_value,adc_raw,voltage_mV
```

Make sure:
- Baud rate is **115200**
- Arduino is outputting CSV format
- Sample rate matches (default: 10 Hz)

---

## Next Steps

1. **Test with voltage divider** - Verify pipeline works with simulated signals
2. **When INA128PA arrives** - Connect to real FPC electrodes
3. **Inoculate FPC** - Wait 2-4 weeks for mycelium colonization
4. **First real recording!** - Use these tools to analyze actual fungal signals

---

*Last Updated: February 3, 2026*
