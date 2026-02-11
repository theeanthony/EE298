# QUICK START: Voltage Divider Simulator
## You have: Arduino Uno R4 + 1MΩ resistors + basic resistors

---

## Build This (5 minutes)

### Parts
- 1× 1MΩ resistor
- 1× 220Ω resistor (or 2× 470Ω in parallel, or 2× 100Ω in series)
- 3× jumper wires

### Wiring

```
Arduino Uno R4                 Breadboard

     D9  ●----[wire]----[  1MΩ  ]----+----[  220Ω  ]----●  GND rail
                                     |
     A0  ●----[wire]-----------------+

    GND  ●----[wire]--------------------------------●  GND rail
```

### Schematic
```
D9 (PWM output)
      |
    [1MΩ]     ← High resistance (limits current)
      |
      +-------→ A0 (ADC input)
      |
    [220Ω]    ← Low resistance (sets output voltage)
      |
     GND
```

---

## Expected Output

| PWM Value | Output Voltage |
|-----------|----------------|
| 0 | 0 mV |
| 128 | ~0.55 mV |
| 255 | ~1.1 mV |

**This matches fungal signal range (0.5-2.1 mV)!**

---

## Upload Code

1. Open Arduino IDE
2. Open file: `firmware/mycelium_signal_simulator/mycelium_signal_simulator.ino`
3. Select Board: **Arduino Uno R4 Minima** (or WiFi)
4. Click Upload

---

## Test It

1. Open Serial Monitor (115200 baud)
2. You should see data streaming:
   ```
   # Mycelium Signal Simulator v1.0
   # Detected: Arduino Uno R4 (14-bit ADC)
   100,128,45,1.374
   200,129,46,1.404
   ...
   ```

3. Type these commands:
   - `0` = Slow sine wave
   - `1` = Random walk
   - `2` = Spikes
   - `3` = Composite (most realistic)
   - `s` = Show statistics

---

## Verify with Multimeter

1. Set multimeter to **mV DC** range
2. Put probes on:
   - **Red (+):** The junction point (between 1MΩ and 220Ω)
   - **Black (-):** GND
3. You should see ~0-1.1 mV as the signal varies

---

## Arduino Uno R4 Advantage

Your R4 has a **14-bit ADC** (16,384 steps) vs the R3's 10-bit (1,024 steps).

| Board | Resolution | mV per step (5V ref) |
|-------|------------|----------------------|
| Uno R3 | 10-bit | 4.88 mV |
| **Uno R4** | **14-bit** | **0.31 mV** |

**The R4 can actually resolve your ~1mV signals directly!** (The R3 cannot.)

---

## Don't Have 220Ω?

| Alternative | Result |
|-------------|--------|
| 2× 100Ω in series | 200Ω → ~1.0 mV output |
| 2× 470Ω in parallel | 235Ω → ~1.2 mV output |
| 330Ω | ~1.6 mV output |
| 1kΩ | ~5 mV output (too high, but usable for testing) |

---

## Next Steps

When your **AD8237 amplifier** arrives:

```
                        ┌─────────────┐
D9 --[1MΩ]--+--[220Ω]--GND           │
            |                         │
            +------→ AD8237 +IN ──────┤
                     AD8237 -IN ← GND │
                     AD8237 OUT ──────┼──→ A0
                        └─────────────┘
```

The amplifier will boost the 1mV signal to ~1V for clean ADC reading.

---

**Total build time: 5 minutes**
**Total cost: $0 (parts you already have)**
