# FINAL Component List - INA128PA + Arduino R4 Setup
## (No PicoLog Data Logger, No Soldering Required)
### EE297B Research Project | Anthony Contreras & Alex Wong

**Last Updated:** February 3, 2026

---

## What You Already Have

| Item | Status |
|------|--------|
| FPC PCB boards (ENIG electrodes) | **HAVE** |
| Blue Oyster mycelium | **HAVE** (just arrived!) |
| Arduino Uno R4 WiFi | **HAVE** |
| 1MΩ resistors (10x) | **HAVE** |
| Basic resistors, wires, breadboard | **HAVE** |

---

## What You Need to Order

### Signal Chain Overview

```
FPC Electrodes → JP1 Header → INA128PA Amp → Arduino A0
     (on PCB)    (solder on)   (breadboard)   (14-bit ADC)
```

**You do NOT need:**
- DB-25 connector (that was for PicoLog)
- SOIC-to-DIP adapter (INA128PA is already DIP-8)
- AD8237 (replaced by INA128PA)

---

## PRIORITY 1: Must Order Now

### Amplifier (Critical) - NO SOLDERING NEEDED!

| Item | Qty | Link | Price |
|------|-----|------|-------|
| **INA128PA** (DIP-8 package) | 2 | [DigiKey](https://www.digikey.com/en/products/detail/texas-instruments/INA128PA/275721) | $8.90 ea |

**Why INA128PA over AD8237?**
| Feature | INA128PA | AD8237 |
|---------|----------|--------|
| Package | **DIP-8 (breadboard-ready!)** | SOIC-8 (needs adapter) |
| Soldering | **NONE** | Required |
| Price | $8.90 | $6.50 + $4 adapter |
| CMRR | 120dB | 106dB |
| Performance | Excellent | Excellent |

### Headers for FPC (JP1 Connection)

| Item | Qty | Link | Price |
|------|-----|------|-------|
| **Male Header 2.54mm (40-pin breakaway)** | 1 | [Amazon](https://www.amazon.com/dp/B07VP63Y4H) | $5.99 (pack) |
| **Female Header 2.54mm (40-pin breakaway)** | 1 | [Amazon](https://www.amazon.com/dp/B07BS126FK) | $6.99 (pack) |

### Wiring

| Item | Qty | Link | Price |
|------|-----|------|-------|
| **Dupont Jumper Wires M-M/M-F** | 1 pack | [Amazon](https://www.amazon.com/dp/B01EV70C78) | $6.99 |

---

## PRIORITY 2: Amplifier Circuit Components

### Resistor for INA128PA Gain Setting

**Gain Formula:** G = 1 + (50kΩ / RG)

| RG Value | Gain | Output for 1mV input | Best For |
|----------|------|----------------------|----------|
| **50Ω** | 1001x | 1.0V | **Recommended** |
| 100Ω | 501x | 0.5V | If signals are stronger |
| 500Ω | 101x | 0.1V | High amplitude signals |

| Item | Qty | Purpose | Link | Price |
|------|-----|---------|------|-------|
| **50Ω resistor (1%)** | 5 | RG (gain) | [Amazon Kit](https://www.amazon.com/dp/B08FD1XVL6) | $5.99 (kit) |
| **10kΩ resistor (1%)** | 10 | Reference divider | (same kit) | - |

### Capacitors

| Item | Qty | Purpose | Link | Price |
|------|-----|---------|------|-------|
| **0.1µF ceramic** | 10 | Bypass/decoupling | [Amazon](https://www.amazon.com/dp/B07WCLQNCD) | $5.99 |
| **10µF electrolytic** | 5 | Power filtering | (likely have) | - |

---

## PRIORITY 3: Faraday Cage / Shielding

| Item | Qty | Link | Price |
|------|-----|------|-------|
| **Aluminum Project Box** | 1 | [Amazon](https://www.amazon.com/dp/B07WCLP59X) | $12-15 |
| **Copper Tape (conductive adhesive)** | 1 roll | [Amazon](https://www.amazon.com/dp/B01MR5DSCM) | $7.99 |
| **BNC Female Panel Mount** | 2 | [Amazon](https://www.amazon.com/dp/B07FQDLX7Z) | $7.99 (5 pack) |
| **Alligator Clip to Banana Plug** | 1 | [Amazon](https://www.amazon.com/dp/B0B21SPM74) | $6.99 |

---

## PRIORITY 4: Inoculation Supplies (If not using pre-poured plates)

| Item | Qty | Link | Price |
|------|-----|------|-------|
| **EZ BioResearch PDA Plates (10-pack)** | 1 | [Amazon](https://www.amazon.com/dp/B074GG4L95) | $19.99 |
| **Sterile Microcentrifuge Tubes** | 1 pack | [Amazon](https://www.amazon.com/dp/B0BKHDFWNM) | $8.99 |
| **Sterile Transfer Pipettes** | 1 pack | [Amazon](https://www.amazon.com/dp/B07MH4YNCS) | $6.99 |
| **Parafilm** | 1 roll | [Amazon](https://www.amazon.com/dp/B07RKWLG4B) | $9.99 |
| **70% Isopropyl Alcohol** | 1 bottle | Local pharmacy | $5 |

---

## Complete Shopping List (Copy-Paste Ready)

### DigiKey Order (~$20)

```
INA128PA (DIP-8) × 2 ..................... $17.80
   https://www.digikey.com/en/products/detail/texas-instruments/INA128PA/275721

                              TOTAL: ~$18
```

### Amazon Order - Essential (~$50)

```
1. Male/Female Header Kit ................ $6.99
   https://www.amazon.com/dp/B07VP63Y4H

2. Dupont Jumper Wire Kit ................ $6.99
   https://www.amazon.com/dp/B01EV70C78

3. Resistor Kit 1% (includes 50Ω) ........ $5.99
   https://www.amazon.com/dp/B08FD1XVL6

4. 0.1µF Capacitors ...................... $5.99
   https://www.amazon.com/dp/B07WCLQNCD

5. Copper Tape ........................... $7.99
   https://www.amazon.com/dp/B01MR5DSCM

6. Aluminum Project Box .................. $12.00
   https://www.amazon.com/dp/B07WCLP59X

                              TOTAL: ~$46
```

### Amazon Order - Inoculation Supplies (~$50)

```
7. EZ BioResearch PDA Plates ............. $19.99
   https://www.amazon.com/dp/B074GG4L95

8. Sterile Transfer Pipettes ............. $6.99
   https://www.amazon.com/dp/B07MH4YNCS

9. Parafilm .............................. $9.99
   https://www.amazon.com/dp/B07RKWLG4B

10. Microcentrifuge Tubes ................ $8.99
    https://www.amazon.com/dp/B0BKHDFWNM

                              TOTAL: ~$46
```

---

## Wiring Diagram: FPC → INA128PA → Arduino

```
YOUR FPC BOARD                    BREADBOARD                      ARDUINO R4
┌─────────────────┐              ┌─────────────────┐             ┌──────────┐
│                 │              │                 │             │          │
│  JP1 Header     │              │    INA128PA     │             │          │
│  ┌───────────┐  │              │     (DIP-8)     │             │          │
│  │ PAIR1_POS ├──┼──────────────┼──► +IN   (3)    │             │          │
│  │ PAIR1_NEG ├──┼──────────────┼──► -IN   (2)    │             │          │
│  │ ...       │  │              │                 │             │          │
│  └───────────┘  │              │     Vout (6) ───┼─────────────┼──► A0    │
│                 │              │                 │             │          │
│  Electrode      │              │     V+   (7) ───┼─────────────┼──► 5V    │
│  Pads           │              │     V-   (4) ───┼─────────────┼──► GND   │
│  (for mycelium) │              │     REF  (5) ───┼─────────────┼──► GND   │
│                 │              │                 │             │          │
│                 │              │     RG   (1) ─┐ │             │          │
│                 │              │     RG   (8) ─┼─┼──[50Ω]      │          │
└─────────────────┘              │               │ │             └──────────┘
                                 │               │ │
                                 │    ┌──────────┘ │
                                 │    │ GAIN=1001x │
                                 └────┴────────────┘

INA128PA PINOUT (DIP-8, TOP VIEW):

        ┌───────┐
   RG  1│●      │8  RG      ← Connect 50Ω between pins 1 & 8
  -IN  2│       │7  V+      ← Connect to 5V
  +IN  3│       │6  Vout    ← Connect to Arduino A0
   V-  4│       │5  REF     ← Connect to GND (or mid-supply for bipolar)
        └───────┘

GAIN = 1 + (50kΩ / RG) = 1 + (50000 / 50) = 1001x
```

---

## What You DON'T Need Anymore

Since you're using Arduino + INA128PA instead of PicoLog + AD8237:

| Item | Why Not Needed | Savings |
|------|----------------|---------|
| ~~PicoLog ADC-24~~ | Using Arduino R4 14-bit ADC | ~$850 |
| ~~AD8237ARZ~~ | Using INA128PA (DIP) instead | - |
| ~~SOIC-8 to DIP Adapter~~ | INA128PA is already DIP-8 | ~$8 |
| ~~DB-25 Female Connector~~ | Was for PicoLog connection | ~$8 |
| ~~DB-25 Male Breakout~~ | Was for PicoLog connection | ~$10 |
| ~~DB-25 Cable~~ | Was for PicoLog connection | ~$10 |

**Total Savings: ~$886**

---

## INA128PA vs AD8237 - Quick Comparison

| Feature | INA128PA (YOUR CHOICE) | AD8237 |
|---------|------------------------|--------|
| Package | **DIP-8 (plug & play!)** | SOIC-8 (needs adapter) |
| Soldering Required | **NO** | YES |
| Breadboard Compatible | **YES, directly** | Needs adapter board |
| Gain Formula | G = 1 + (50kΩ/RG) | G = 1 + (R2/R1) |
| For G=1000 | RG = 50Ω | R1=100Ω, R2=100kΩ |
| CMRR | 120 dB | 106 dB |
| Input Impedance | 10^10 Ω | 10^10 Ω |
| Price | $8.90 | $6.50 + $4 adapter |
| **Verdict** | **Best for prototyping** | Better for final PCB |

---

## Order Summary

| Priority | Items | Cost | Where |
|----------|-------|------|-------|
| **NOW** | INA128PA × 2 | ~$18 | DigiKey |
| **NOW** | Headers + wires | ~$20 | Amazon |
| **NOW** | Resistors + caps | ~$12 | Amazon |
| **Week 2** | Faraday cage materials | ~$28 | Amazon |
| **If needed** | Inoculation supplies | ~$46 | Amazon |

**Essential Total: ~$78**
**Full Total: ~$124**

---

## Next Steps

1. **Order INA128PA from DigiKey** (ships fast, usually 2-3 days)
2. **Order Amazon essentials** (headers, wires, resistors)
3. **While waiting:** Continue testing with voltage divider simulator
4. **When parts arrive:** Build INA128PA circuit on breadboard
5. **Week 2:** Inoculate FPC with Blue Oyster mycelium
6. **Weeks 3-5:** Wait for mycelium colonization
7. **Week 5-6:** First real recording attempt!

---

*You're all set! The INA128PA will plug directly into your breadboard - no soldering, no adapters needed. Just connect 50Ω between pins 1 & 8 for 1000x gain, hook up power and ground, and you're ready to amplify those fungal signals!*
