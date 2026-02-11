# Voltage Divider Mycelium Signal Simulator
## Build Guide - Using Only Basic Components
### EE297B Research Project | Anthony Contreras & Alex Wong

**Purpose:** Simulate the ~0.5-2.1 mV bioelectrical signals from fungal mycelium so you can develop and test your signal processing pipeline while waiting for biological samples.

**What You Have:**
- Arduino Uno R4 (3.3V and 5V outputs!)
- Breadboard
- Resistors: **10× 1MΩ** + various others
- Jumper wires
- LEDs

**What You DON'T Have Yet:**
- AD8237 amplifier (on order)
- FPC board (on order)

---

## RECOMMENDED: Simple Single-Stage Design (You Have 1MΩ!)

Since you have 1MΩ resistors, use this **much simpler** single-stage design:

### Circuit (Super Simple)

```
Arduino D9 (PWM) ----[1MΩ]----+----[220Ω]---- GND
                              |
                              +-------------> Arduino A0
```

### Math
```
Vout = Vin × (R2 / (R1 + R2))
     = 5V × (220Ω / (1,000,000Ω + 220Ω))
     = 5V × (220 / 1,000,220)
     = 5V × 0.00022
     ≈ 1.1 mV
```

**That's it! One 1MΩ + one 220Ω = done.**

### Even Better: Use 3.3V Output

Your Uno R4 has a 3.3V pin! Using 3.3V gives you:
```
Vout = 3.3V × (220 / 1,000,220) ≈ 0.73 mV
```

This is closer to the lower end of fungal signals (0.5-2.1 mV).

### Breadboard Wiring (3 wires + 2 resistors)

```
Arduino Uno R4                    Breadboard
+-------------+
|          D9 |---wire---[1MΩ]---+---[220Ω]---GND rail
|          A0 |---wire-----------+
|         GND |---wire-----------------------GND rail
|         3V3 |  (optional: use instead of D9 for fixed 0.73mV)
+-------------+
```

---

## Part 1: Understanding the Target Signal

### Real Fungal Signal Characteristics (from literature)

| Parameter | Value | Source |
|-----------|-------|--------|
| Amplitude | 0.5 - 2.1 mV | Adamatzky et al., Buffi et al. |
| Frequency | 0.01 - 1 Hz | Very slow oscillations |
| Waveform | Irregular spikes, not sinusoidal | Biological variability |
| DC offset | Variable | Electrode drift |

### What We're Simulating

We need to create a **millivolt-level signal** from the Arduino's 5V output. This requires dividing the voltage by ~2500-5000x.

---

## Part 2: The Problem with Standard Resistor Kits

Most resistor kits max out at **1MΩ**. To get from 5V down to 1mV, you'd ideally want:

```
Vout = Vin × (R2 / (R1 + R2))
1mV = 5000mV × (R2 / (R1 + R2))
```

This requires R1/R2 ratio of ~5000:1, which is hard with typical resistor values.

**Solution:** Use a **two-stage voltage divider** with resistors you likely have.

---

## Part 3: Two-Stage Voltage Divider Design

### Stage 1: 5V → 50mV (100:1 division)

```
5V ----[100kΩ]----+----[1kΩ]---- GND
                  |
                  +---- ~50mV output
```

**Math:** 5V × (1k / (100k + 1k)) = 5V × (1/101) ≈ **49.5 mV**

### Stage 2: 50mV → ~1mV (50:1 division)

```
~50mV ----[47kΩ]----+----[1kΩ]---- GND
                    |
                    +---- ~1mV output (to Arduino A0)
```

**Math:** 49.5mV × (1k / (47k + 1k)) = 49.5mV × (1/48) ≈ **1.03 mV**

---

## Part 4: Complete Circuit Schematic

```
                    STAGE 1                      STAGE 2
                    (100:1)                      (50:1)

Arduino D9 ----[100kΩ]----+----[1kΩ]----+----[47kΩ]----+----[1kΩ]---- GND
  (PWM)                   |             |              |
                          |            GND             |
                          |                            |
                      (≈50mV)                      (≈1mV)
                          |                            |
                          |                            +-----> Arduino A0
                          |                                    (ADC input)
                          |
                     [OPTIONAL]
                     Connect LED + 330Ω here
                     to visualize PWM activity
                     (won't affect signal much)
```

### Breadboard Layout

```
    Arduino                          Breadboard
   +-------+
   |    D9 |----wire-----------[100kΩ]---+---[1kΩ]---+---[47kΩ]---+---[1kΩ]---GND
   |       |                             |           |            |
   |    A0 |----wire--------------------------------------------- +
   |       |                             |           |
   |   GND |----wire--------------------|-----------|---------------------------GND
   |       |                            |           |
   |    5V |                           LED         GND
   +-------+                          (opt)
                                        |
                                      [330Ω]
                                        |
                                       GND
```

---

## Part 5: Resistor Substitutions

**Don't have the exact values?** Here are alternatives:

### For 100kΩ (Stage 1 high-side):
| Alternative | Result |
|-------------|--------|
| 2× 47kΩ in series | 94kΩ (close enough) |
| 100kΩ (if you have it) | Perfect |
| 82kΩ + 22kΩ in series | 104kΩ (close enough) |

### For 47kΩ (Stage 2 high-side):
| Alternative | Result |
|-------------|--------|
| 2× 22kΩ in series | 44kΩ (close enough) |
| 47kΩ (if you have it) | Perfect |
| 33kΩ + 10kΩ in series | 43kΩ (close enough) |
| 10kΩ + 10kΩ + 10kΩ + 10kΩ in series | 40kΩ (usable) |

### For 1kΩ (both stages low-side):
| Alternative | Result |
|-------------|--------|
| 1kΩ | Perfect (usually in every kit) |
| 2× 2.2kΩ in parallel | ~1.1kΩ (close enough) |
| 1.5kΩ | Will give slightly different voltage (still usable) |

---

## Part 6: Resistor Options with Your 1MΩ Resistors

You have **10× 1MΩ resistors**. Here are different output voltages you can achieve:

### Option A: 1MΩ + 220Ω = ~1.1 mV (from 5V)
```
D9 ----[1MΩ]----+----[220Ω]---- GND
                |
                +--> A0
```

### Option B: 1MΩ + 100Ω = ~0.5 mV (from 5V)
```
D9 ----[1MΩ]----+----[100Ω]---- GND
                |
                +--> A0
```
This matches the LOW end of fungal signals.

### Option C: 1MΩ + 470Ω = ~2.3 mV (from 5V)
```
D9 ----[1MΩ]----+----[470Ω]---- GND
                |
                +--> A0
```
This matches the HIGH end of fungal signals.

### Option D: 2× 1MΩ in series + 220Ω = ~0.55 mV (from 5V)
```
D9 ----[1MΩ]----[1MΩ]----+----[220Ω]---- GND
                         |
                         +--> A0
```
For even smaller signals.

### Using 3.3V Instead of 5V (Uno R4 Advantage!)

| R1 | R2 | From 5V | From 3.3V |
|----|-----|---------|-----------|
| 1MΩ | 100Ω | 0.5 mV | 0.33 mV |
| 1MΩ | 220Ω | 1.1 mV | **0.73 mV** |
| 1MΩ | 470Ω | 2.3 mV | **1.5 mV** |
| 1MΩ | 1kΩ | 5.0 mV | 3.3 mV |

**Recommendation:** Use **1MΩ + 220Ω from 3.3V** for ~0.73 mV - right in the middle of the fungal signal range.

---

## Part 7: Arduino Code for Signal Generator

```cpp
/*
 * mycelium_signal_simulator.ino
 * Generates fake fungal bioelectrical signals for pipeline testing
 *
 * NO EXTERNAL COMPONENTS NEEDED BEYOND RESISTORS
 */

const int PWM_PIN = 9;      // Output pin (to voltage divider)
const int ADC_PIN = A0;     // Input pin (from voltage divider)
const int LED_PIN = 13;     // Built-in LED for activity indicator

// Timing
const int SAMPLE_RATE_MS = 100;  // 10 Hz sampling
unsigned long lastSample = 0;

// Signal generation state
float phase = 0;
int signalMode = 0;  // 0=slow sine, 1=random walk, 2=spikes

void setup() {
    Serial.begin(115200);
    pinMode(PWM_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);

    // Use internal 1.1V reference for better resolution at low voltages
    // Comment this out if your signals are too small to read
    // analogReference(INTERNAL);  // 1.1V reference (Uno)

    Serial.println("# Mycelium Signal Simulator v1.0");
    Serial.println("# Format: timestamp_ms,pwm_value,adc_raw,adc_mV");
    Serial.println("# Commands: 0=sine, 1=random, 2=spikes, r=reset");
}

void loop() {
    // Check for mode change commands
    if (Serial.available()) {
        char cmd = Serial.read();
        switch(cmd) {
            case '0': signalMode = 0; Serial.println("# Mode: Slow Sine"); break;
            case '1': signalMode = 1; Serial.println("# Mode: Random Walk"); break;
            case '2': signalMode = 2; Serial.println("# Mode: Spikes"); break;
            case 'r': phase = 0; Serial.println("# Reset"); break;
        }
    }

    unsigned long now = millis();

    if (now - lastSample >= SAMPLE_RATE_MS) {
        lastSample = now;

        // Generate signal based on mode
        int pwmValue = generateSignal();

        // Output to voltage divider
        analogWrite(PWM_PIN, pwmValue);

        // Blink LED to show activity
        digitalWrite(LED_PIN, pwmValue > 127);

        // Read back through ADC
        int adcRaw = analogRead(ADC_PIN);

        // Convert to millivolts (assuming 5V reference, 10-bit ADC)
        // 5000mV / 1024 steps = 4.883 mV per step
        float adcMillivolts = (adcRaw / 1023.0) * 5000.0;

        // Output CSV data
        Serial.print(now);
        Serial.print(",");
        Serial.print(pwmValue);
        Serial.print(",");
        Serial.print(adcRaw);
        Serial.print(",");
        Serial.println(adcMillivolts, 3);
    }
}

int generateSignal() {
    static int randomValue = 128;
    static unsigned long lastSpike = 0;

    int pwmValue = 0;

    switch(signalMode) {
        case 0:  // Slow sine wave (~0.05 Hz = 20 second period)
            phase += 0.0314;  // 2*PI / 200 samples = 20 sec at 10Hz
            if (phase > 6.28) phase = 0;
            pwmValue = 128 + (int)(127 * sin(phase));
            break;

        case 1:  // Random walk (mimics biological noise)
            randomValue += random(-10, 11);  // Random step -10 to +10
            randomValue = constrain(randomValue, 0, 255);
            pwmValue = randomValue;
            break;

        case 2:  // Occasional spikes (mimics action potentials)
            pwmValue = 20;  // Low baseline
            if (millis() - lastSpike > random(3000, 8000)) {  // Spike every 3-8 sec
                pwmValue = random(180, 255);  // Spike amplitude
                lastSpike = millis();
            }
            break;
    }

    return pwmValue;
}
```

---

## Part 8: Testing Without the Amplifier

Since you don't have the AD8237 yet, the Arduino's 10-bit ADC (4.88 mV resolution) **cannot directly resolve 1 mV signals**.

### Options:

**Option A: Test the Divider Output with Multimeter**
1. Build the voltage divider
2. Set Arduino PWM to fixed values (0, 128, 255)
3. Measure output voltage with multimeter on mV scale
4. Verify you get ~0-1mV range

**Option B: Use Higher Voltage for Now**
Skip Stage 2 and read the ~50mV signal directly:
```
Arduino D9 ----[100kΩ]----+----[1kΩ]---- GND
                          |
                          +-----> Arduino A0
```
This gives ~50mV which the Arduino CAN read (≈10 ADC counts).

**Option C: Use Arduino's 1.1V Internal Reference**
Uncomment `analogReference(INTERNAL);` in the code.
- Changes ADC range from 0-5V to 0-1.1V
- Resolution improves to ~1.07 mV per step
- Still marginal for 1mV signals, but better

**Option D: Wait for AD8237**
Build the divider now, verify with multimeter, then add the amplifier when it arrives.

---

## Part 9: Quick Build Checklist

### Before You Start
- [ ] Identify resistors: 100kΩ (or substitute), 47kΩ (or substitute), 1kΩ × 2
- [ ] Clear space on breadboard
- [ ] Have multimeter ready (mV range)

### Build Steps
- [ ] Insert 100kΩ resistor from D9 row to middle of board
- [ ] Insert 1kΩ from middle to GND rail (Stage 1 complete)
- [ ] Insert 47kΩ from Stage 1 output to new row
- [ ] Insert 1kΩ from that row to GND rail (Stage 2 complete)
- [ ] Connect Stage 2 output to A0
- [ ] Connect Arduino GND to breadboard GND rail
- [ ] Double-check all connections

### Verify
- [ ] Upload code to Arduino
- [ ] Open Serial Monitor (115200 baud)
- [ ] See data streaming
- [ ] Measure voltage at Stage 2 output with multimeter
- [ ] Confirm ~0-1mV swing as PWM changes

---

## Part 10: Expected Results

### With Multimeter at Stage 2 Output:

| PWM Value | Expected Voltage |
|-----------|------------------|
| 0 | ~0 mV |
| 64 | ~0.25 mV |
| 128 | ~0.5 mV |
| 192 | ~0.75 mV |
| 255 | ~1.0 mV |

### In Serial Monitor (without amplifier):

The ADC readings will be very small (0-2 counts) because Arduino can't resolve 1mV directly. This is expected! The important thing is:

1. The divider is working (verify with multimeter)
2. The code is running
3. When the AD8237 arrives, you just add it in the middle

---

## Part 11: Wiring Diagram for When AD8237 Arrives

```
                    VOLTAGE DIVIDER                    AMPLIFIER                 ARDUINO

Arduino D9 --[100k]--+--[1k]--+--[47k]--+--[1k]-- GND
                     |        |         |
                    GND      GND        |
                                        |
                              Simulated Signal (~1mV)
                                        |
                                        v
                               +----------------+
                               |    AD8237      |
                        GND ---|1 -IN     +VS 8|--- +5V
               Signal In ------|2 +IN     OUT 7|-----------> Arduino A0
                               |3 FB      REF 6|---+--[10k]--+5V
                        GND ---|4 -VS      RG 5|   |
                               +----------------+   +--[10k]--GND
                                    |      |       |
                                 [100Ω]    +-------+
                                    |              |
                                    +---[100k]-----+

                               (Gain = 1 + 100k/100 = 1001)
```

---

## Quick Reference Card

### Resistor Values for ~1mV Output

| Stage | High-Side R | Low-Side R | Division |
|-------|-------------|------------|----------|
| 1 | 100kΩ | 1kΩ | ÷101 |
| 2 | 47kΩ | 1kΩ | ÷48 |
| **Total** | - | - | **÷4848** |

**5V ÷ 4848 ≈ 1.03 mV** ✓

### Arduino Pins

| Pin | Function |
|-----|----------|
| D9 | PWM output (to voltage divider) |
| A0 | ADC input (from voltage divider or amplifier) |
| GND | Common ground |
| D13 | Activity LED (built-in) |

### Serial Commands

| Command | Effect |
|---------|--------|
| `0` | Slow sine wave mode |
| `1` | Random walk mode |
| `2` | Spike mode |
| `r` | Reset phase |

---

*Created: January 28, 2026*
*For: EE297B Research Project - Signal Processing Fungi Propagation*
