# Component Checklist
## EE297B Research Project - Signal Processing Fungi Propagation
### Anthony Contreras & Alex Wong

**Date Created:** January 28, 2026
**Last Updated:** February 3, 2026

---

## Current Status Summary

| Item | Status |
|------|--------|
| FPC PCB boards (ENIG electrodes) | **HAVE** |
| Blue Oyster mycelium | **HAVE** (just arrived) |
| Arduino Uno R4 WiFi | **HAVE** |
| 1MΩ resistors (10x) | **HAVE** |
| Basic resistors, wires, breadboard | **HAVE** |
| INA128PA Amplifier | **NEED TO ORDER** |

---

## Signal Chain (Arduino R4 Setup)

```
FPC Electrodes → JP1 Header → INA128PA Amp → Arduino A0
     (on PCB)    (solder on)   (breadboard)   (14-bit ADC)
```

**NOTE:** We are using Arduino Uno R4 (14-bit ADC) instead of PicoLog ADC-24.
**NOTE:** DB-25 connector NOT needed - that was for PicoLog. Use JP1 header instead.

---

## Priority 1: Order Immediately (Critical Path)

### Instrumentation Amplifier - MUST ORDER

| Status | Item | Qty | Source | Est. Price | Notes |
|:------:|------|-----|--------|------------|-------|
| [ ] | **INA128PA** (DIP-8 package) | 2 | [DigiKey](https://www.digikey.com/en/products/detail/texas-instruments/INA128PA/275721) | $8.90 ea | **Breadboard-ready, no soldering!** |

**Why INA128PA?**
- DIP-8 package plugs directly into breadboard
- No SOIC-to-DIP adapter needed
- No soldering required for prototyping
- Excellent performance (120dB CMRR, low noise)
- Industry standard instrumentation amplifier

### Headers for FPC (JP1 Connection) - MUST ORDER

| Status | Item | Qty | Source | Est. Price | Notes |
|:------:|------|-----|--------|------------|-------|
| [ ] | 40-pin male header strip (2.54mm, breakaway) | 1 | [Amazon](https://www.amazon.com/dp/B07VP63Y4H) | $5.99/pack | Solder to FPC JP1 |
| [ ] | 40-pin female header strip (2.54mm, breakaway) | 1 | [Amazon](https://www.amazon.com/dp/B07BS126FK) | $6.99/pack | For breadboard connection |
| [ ] | Dupont jumper wires M-M/M-F kit | 1 pack | [Amazon](https://www.amazon.com/dp/B01EV70C78) | $6.99 | Connect everything |

---

## Priority 2: Check Your Inventory First

### Resistors for INA128PA Gain Setting (G=1000)

**Gain Formula:** G = 1 + (50kΩ / RG)
**For G=1000:** RG = 50kΩ / 999 ≈ **50Ω**

| Have? | Item | Qty Needed | Purpose | Notes |
|:-----:|------|------------|---------|-------|
| [ ] | **50Ω (1%)** | 5 | Gain resistor RG | For G=1000x |
| [ ] | 10kΩ (1%) | 10 | Reference voltage divider | Common, likely have |

**Alternative Gain Options:**
| RG Value | Gain | Output for 1mV input |
|----------|------|----------------------|
| 50Ω | 1001x | 1.0V |
| 100Ω | 501x | 0.5V |
| 500Ω | 101x | 0.1V |

### Capacitors

| Have? | Item | Qty Needed | Purpose | Notes |
|:-----:|------|------------|---------|-------|
| [x] | 0.1µF ceramic | 10 | Bypass/decoupling | Likely have |
| [x] | 10µF electrolytic | 5 | Power supply filtering | Likely have |

### Development Hardware (VERIFY YOU HAVE)

| Have? | Item | Status |
|:-----:|------|--------|
| [x] | Arduino Uno R4 (WiFi or Minima) | **HAVE** |
| [x] | USB-C cable (for R4) | **HAVE** |
| [x] | Breadboard (830 pts) | **HAVE** |
| [x] | Jumper wire kit | **HAVE** |
| [x] | 1MΩ resistors | **HAVE** (10x) |
| [ ] | Multimeter | Check if have |

---

## Priority 3: Faraday Cage Materials (Week 2)

| Status | Item | Qty | Source | Est. Price | Notes |
|:------:|------|-----|--------|------------|-------|
| [ ] | Metal project box or ammo can | 1 | [Amazon](https://www.amazon.com/dp/B07WCLP59X) | $12-15 | Must be conductive |
| [ ] | Copper tape (conductive adhesive) | 1 roll | [Amazon](https://www.amazon.com/dp/B01MR5DSCM) | $7.99 | Seal gaps/seams |
| [ ] | BNC panel-mount female connectors | 2-4 | [Amazon](https://www.amazon.com/dp/B07FQDLX7Z) | $7.99/5-pack | Signal feedthrough |
| [ ] | Alligator clip to banana plug | 1 | [Amazon](https://www.amazon.com/dp/B0B21SPM74) | $6.99 | Earth ground |

---

## Priority 4: Inoculation Supplies (If needed)

| Status | Item | Qty | Source | Est. Price | Notes |
|:------:|------|-----|--------|------------|-------|
| [x] | Blue Oyster mycelium | 1 | - | - | **HAVE** (just arrived) |
| [ ] | EZ BioResearch PDA Plates (10-pack) | 1 | [Amazon](https://www.amazon.com/dp/B074GG4L95) | $19.99 | Pre-poured, no cooking |
| [ ] | Sterile Transfer Pipettes | 1 pack | [Amazon](https://www.amazon.com/dp/B07MH4YNCS) | $6.99 | For liquid inoculation |
| [ ] | Parafilm | 1 roll | [Amazon](https://www.amazon.com/dp/B07RKWLG4B) | $9.99 | Seal plates |
| [ ] | Sterile Microcentrifuge Tubes | 1 pack | [Amazon](https://www.amazon.com/dp/B0BKHDFWNM) | $8.99 | Store liquid culture |
| [ ] | 70% Isopropyl Alcohol | 1 bottle | Local pharmacy | $5 | Sterilization |

---

## Priority 5: Nice-to-Have (Quality Improvements)

| Status | Item | Qty | Source | Est. Price | Notes |
|:------:|------|-----|--------|------------|-------|
| [ ] | USB isolator | 1 | Amazon | $15-25 | Reduce ground loops |
| [ ] | 9V battery + barrel jack | 1 | Amazon | $5 | Clean isolated power |
| [ ] | Shielded 2-conductor cable | 1m | DigiKey | $5 | Low-noise signal path |

---

## Complete Shopping List (Copy-Paste Ready)

### DigiKey Order (~$20)

```
INA128PA (DIP-8) × 2 ..................... $17.80
   https://www.digikey.com/en/products/detail/texas-instruments/INA128PA/275721

50Ω 1% resistor (if not in kit) × 5 ...... ~$2
                                    Total: ~$20
```

### Amazon Order - Essential (~$48)

```
1. Male/Female Header Kit ................ $6.99
   https://www.amazon.com/dp/B07VP63Y4H

2. Dupont Jumper Wire Kit ................ $6.99
   https://www.amazon.com/dp/B01EV70C78

3. Resistor Kit 1% (for 50Ω, 10kΩ) ....... $5.99
   https://www.amazon.com/dp/B08FD1XVL6

4. 0.1µF Capacitors ...................... $5.99
   https://www.amazon.com/dp/B07WCLQNCD

5. Copper Tape ........................... $7.99
   https://www.amazon.com/dp/B01MR5DSCM

6. Metal Project Box ..................... $12.00
   https://www.amazon.com/dp/B07WCLP59X

                                    Total: ~$48
```

### Amazon Order - Inoculation (~$52)

```
7. EZ BioResearch PDA Plates ............. $19.99
   https://www.amazon.com/dp/B074GG4L95

8. Sterile Transfer Pipettes ............. $6.99
   https://www.amazon.com/dp/B07MH4YNCS

9. Parafilm .............................. $9.99
   https://www.amazon.com/dp/B07RKWLG4B

10. Microcentrifuge Tubes ................ $8.99
    https://www.amazon.com/dp/B0BKHDFWNM

11. 70% Isopropyl Alcohol ................ $5.00
    Local pharmacy

                                    Total: ~$52
```

---

## INA128PA Wiring Diagram

```
YOUR FPC BOARD                    BREADBOARD                      ARDUINO R4
┌─────────────────┐              ┌─────────────────┐             ┌──────────┐
│                 │              │                 │             │          │
│  JP1 Header     │              │    INA128PA     │             │          │
│  ┌───────────┐  │              │   (DIP-8)       │             │          │
│  │ PAIR1_POS ├──┼──────────────┼──►│+IN   3│    │             │          │
│  │ PAIR1_NEG ├──┼──────────────┼──►│-IN   2│    │             │          │
│  │ ...       │  │              │   │       │    │             │          │
│  └───────────┘  │              │   │Vout  6├────┼─────────────┼──► A0    │
│                 │              │   │       │    │             │          │
│  Electrode      │              │   │V+    7├────┼─────────────┼──► 5V    │
│  Pads           │              │   │V-    4├────┼─────────────┼──► GND   │
│  (for mycelium) │              │   │       │    │             │          │
│                 │              │   │RG    1├─┐  │             │          │
│                 │              │   │RG    8├─┼──┼──[50Ω]      │          │
│                 │              │   └───────┘ │  │             │          │
└─────────────────┘              │             │  │             └──────────┘
                                 │  For G=1000x│  │
                                 │  use 50Ω RG │  │
                                 └─────────────────┘

INA128PA Pin Configuration:
  Pin 1: RG (gain resistor)
  Pin 2: -IN (negative input)
  Pin 3: +IN (positive input)
  Pin 4: V- (ground)
  Pin 5: REF (reference, tie to GND or mid-supply)
  Pin 6: Vout (output)
  Pin 7: V+ (5V)
  Pin 8: RG (gain resistor)

GAIN = 1 + (50kΩ / RG)
For G=1000: RG = 50Ω
```

---

## FPC Header Pinout Reference (JP1)

Your FPC board (11600498a_y2) has **16 signal lines** from 8 electrode pairs:

| FPC Pin | Signal | Description |
|---------|--------|-------------|
| 1 | PAIR1_NEG | Electrode Pair 1 Negative |
| 2 | PAIR1_POS | Electrode Pair 1 Positive |
| 3 | PAIR2_NEG | Electrode Pair 2 Negative |
| 4 | PAIR2_POS | Electrode Pair 2 Positive |
| 5 | PAIR3_NEG | Electrode Pair 3 Negative |
| 6 | PAIR3_POS | Electrode Pair 3 Positive |
| 7 | PAIR4_NEG | Electrode Pair 4 Negative |
| 8 | PAIR4_POS | Electrode Pair 4 Positive |
| 9 | PAIR5_NEG | Electrode Pair 5 Negative |
| 10 | PAIR5_POS | Electrode Pair 5 Positive |
| 11 | PAIR6_NEG | Electrode Pair 6 Negative |
| 12 | PAIR6_POS | Electrode Pair 6 Positive |
| 13 | PAIR7_NEG | Electrode Pair 7 Negative |
| 14 | PAIR7_POS | Electrode Pair 7 Positive |
| 15 | PAIR8_NEG | Electrode Pair 8 Negative |
| 16 | PAIR8_POS | Electrode Pair 8 Positive |

**Wiring to INA128PA:**
- Connect PAIR_POS → INA128PA +IN (pin 3)
- Connect PAIR_NEG → INA128PA -IN (pin 2)
- This gives you differential measurement across each electrode pair

---

## Order Tracking

| Order # | Source | Items | Date Ordered | Est. Arrival | Actual Arrival |
|---------|--------|-------|--------------|--------------|----------------|
| _______ | DigiKey | INA128PA × 2 | ______ | ______ | ______ |
| _______ | Amazon | Headers, wires, cage | ______ | ______ | ______ |
| _______ | Amazon | Inoculation supplies | ______ | ______ | ______ |

---

## What You DON'T Need

Since you're using Arduino R4 instead of PicoLog:

| Item | Why Not Needed |
|------|----------------|
| ~~AD8237ARZ~~ | Using INA128PA (DIP) instead - no adapter needed |
| ~~SOIC-8 to DIP Adapter~~ | INA128PA is already DIP-8 |
| ~~PicoLog ADC-24~~ | Using Arduino R4 14-bit ADC |
| ~~DB-25 Female Connector~~ | Was for PicoLog connection |
| ~~DB-25 Male Breakout~~ | Was for PicoLog connection |
| ~~DB-25 Cable~~ | Was for PicoLog connection |

**Savings: ~$850+ (no PicoLog) + ~$30 (no DB-25 parts) + ~$10 (no adapter)**

---

## Budget Summary

| Category | Cost |
|----------|------|
| INA128PA × 2 (DigiKey) | ~$20 |
| Headers + Wires (Amazon) | ~$20 |
| Resistors + Caps (Amazon) | ~$12 |
| Faraday Cage Materials | ~$28 |
| **Essential Total** | **~$80** |
| Inoculation Supplies (optional) | ~$52 |
| **Full Total** | **~$132** |

---

## Notes

```
_____________________________________________________________

_____________________________________________________________

_____________________________________________________________

_____________________________________________________________

_____________________________________________________________
```

---

*Last updated: February 3, 2026*
