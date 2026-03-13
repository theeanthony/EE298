# Inoculation Log — EE297B Mycelium Experiment
**Anthony Contreras & Alex Wong | SJSU MSEE**

---

## Day 0 — Inoculation Event

| Field | Value |
|---|---|
| **Date** | 2026-02-24 |
| **Time (UTC)** | 05:45 UTC |
| **Time (PST)** | 9:45 PM PST |
| **Logger seq at inoculation** | ~210,600 (recording_20260223.csv) |
| **Baseline voltage pre-inoculation** | ~2379 mV (stable, mid-rail) |

---

## Inoculation Details

| Field | Value |
|---|---|
| **Organism** | Oyster mushroom (*Pleurotus ostreatus*) |
| **Inoculum type** | Liquid culture (LC) syringe |
| **Volume per pair** | ~20 µL |
| **Pairs inoculated** | 2, 3, 4, 5 (JP1 pins 3–10) |
| **Control pairs** | 6, 7 (JP1 pins 11–14) — dry, uninoculated |
| **PCB cleaning** | Isopropyl alcohol wipe, dried fully before inoculation |
| **LC temperature** | Room temperature (warmed from fridge ~2 hours) |

---

## Hardware Configuration at Inoculation

| Component | State |
|---|---|
| INA128 #1 | Pair 2 → Arduino A0 (active, measuring) |
| REF pin (pin 5) | 2.23V mid-rail (10kΩ+10kΩ divider) |
| Decoupling caps | 0.1µF on V+ (pin 7) and V− (pin 4) |
| Voltage divider simulator | Removed |
| Computer B power | Laptop battery (wall charger unplugged — noise reduction) |
| Logger | Running persistently via nohup + caffeinate |
| Sample rate | 10 Hz, 14-bit ADC |

---

## JP1 Wire Map (confirmed 2026-02-23)

| JP1 Pin | Electrode | Wire Colors | Connected To |
|---|---|---|---|
| 3 | Pair 2 negative | maroon → maroon | INA128 pin 2 (IN−) |
| 4 | Pair 2 positive | maroon → blue | INA128 pin 3 (IN+) |
| 5 | Pair 3 negative | white → white | Dangling (connect Day 3) |
| 6 | Pair 3 positive | brown → black | Dangling (connect Day 3) |
| 11 | Pair 6 negative | orange → orange | Dangling (connect Day 4+) |
| 12 | Pair 6 positive | yellow → red | Dangling (connect Day 4+) |
| 13 | Pair 7 negative | green → green | Dangling (connect Day 4+) |
| 14 | Pair 7 positive | yellow → red | Dangling (connect Day 4+) |

---

## Humidity Chamber

| Field | Value |
|---|---|
| Container | Plastic food storage container inside aluminum Faraday cage |
| Substrate | 1cm vermiculite layer (Sukh fine horticultural grade) |
| Moisture source | Boiled tap water, cooled to room temperature |
| PCB position | Elevated above vermiculite — no contact |
| Faraday cage insulation | Cardboard lining (dry, protected from moisture by plastic container) |

---

## Expected Milestones

| Day | Expected |
|---|---|
| Days 1–3 | Flat signal — colonization only, no electrical activity yet |
| Day 3 | Wire pair 3 to INA128 #2 → Arduino A1 (second channel) |
| Day 4+ | Signal changes begin (per Buffi et al. 2025) |
| Day 4+ | Wire pair 6 or 7 to INA128 #3 → Arduino A2 (control comparison) |

---

## Daily Check Command (run from Computer A)

```bash
ssh anthonycontreras@anthonys-mbp.lan "tail -5 ~/EE298/data/raw/recording_\$(date +%Y%m%d).csv"
```

## Live Monitor (run directly on Computer B)

```bash
watch -n 5 "tail -5 ~/EE298/data/raw/recording_\$(date +%Y%m%d).csv"
```

---

## Inoculation Notes (as-happened)
- LC syringe came out fast — pair 2 has large puddle near negative electrode
- Pairs 3-5 have smaller puddles, not precisely on pads
- Pair 4 has bridging between puddles — electrically irrelevant (no wires connected)
- Logger briefly unplugged during inoculation — GAP row written, reconnected immediately
- Signal dropped from ~2379 mV (dry baseline) to ~100 mV after LC application — expected (ionic solution creating electrochemical potential at electrode pads)
- Cage sealed at ~05:58 UTC 2026-02-24

## What This Means
Messy inoculation is normal in lab settings. Oyster mushroom LC is aggressive —
mycelium will spread from wherever the LC landed and colonize toward the electrodes.
Signal at ~100 mV is the new wet baseline. Expect it to shift and stabilize over
the next 12-24 hours as LC absorbs. Watch for Day 4+ signal changes per Buffi.

## Outcome — FAILED (confirmed Day 4)
Signal returned to dry baseline (~2379 mV) by Day 4. LC dried before mycelium established.
Root cause: bare FR4 is not nutritious; no growth medium to anchor LC to electrode pads.
Proceeded to Second Attempt (see below).

---

# Second Inoculation Attempt

## Day 0 — 2026-03-05

| Field | Value |
|---|---|
| **Date** | 2026-03-05 |
| **Time (UTC)** | ~20:41 UTC (logger start) |
| **Time (PST)** | ~12:41 PM PST |
| **Logger seq at inoculation** | ~1 (fresh logger start) |
| **Baseline voltage post-equilibration** | ~2378 mV all 4 channels |

## Improved Inoculation Method

| Field | Value |
|---|---|
| **Scaffold** | Agar agar powder (telephone brand) + sugar, boiled, deposited on electrode pads first |
| **Inoculum** | Oyster mushroom (*Pleurotus ostreatus*) liquid culture syringe (fresh, from fridge) |
| **Application** | LC applied to **positive electrode only** per pair (per Buffi — fungus grows toward second electrode) |
| **Volume per pair** | ~20 µL |
| **Pairs inoculated** | 2, 3 (JP1 pins 3–6) |
| **Control pairs** | 6, 7 (JP1 pins 11–14) — dry, uninoculated |
| **Note** | Pair 3 had large LC blob on both sides (syringe squirted initially) |

## Hardware Configuration

| Component | State |
|---|---|
| INA128 #1 (A0) | Pair 2 → Arduino A0 |
| INA128 #2 (A1) | Pair 3 → Arduino A1 |
| INA128 #3 (A2) | Pair 6 (ctrl) → Arduino A2 |
| INA128 #4 (A3) | Pair 7 (ctrl) → Arduino A3 |
| REF pin (pin 5) | ~2.23V mid-rail (10kΩ+10kΩ divider) on each amp |
| Decoupling caps | 0.1µF on V+ and V− of each amp |
| Computer B power | Laptop battery (wall charger unplugged) |
| Logger | 14-column CSV, nohup + caffeinate, 10 Hz 14-bit |
| Firmware | fungal_signal_acquisition.ino v2.0 (4-channel) |

## Expected Milestones

| Day | Date | Expected |
|---|---|---|
| Days 1–3 | Mar 6–8 | Flat signal — colonization only |
| Day 4+ | Mar 9+ | Signal changes on pairs 2, 3 vs controls 6, 7 |

## Daily Check Commands

```bash
# From Computer A:
ssh anthonycontreras@anthonys-mbp.lan "tail -3 ~/EE298/data/raw/recording_$(date +%Y%m%d).csv"
ssh anthonycontreras@anthonys-mbp.lan "cat ~/EE298/data/raw/logger_status.txt"
```

## Day 3 — 2026-03-08

| Channel | Morning baseline | Evening (disturbed) | Notes |
|---|---|---|---|
| Pair 2 (inoculated) | ~2366 mV | ~4261 mV | Elevated — agar electrochemical potential or probing disturbance |
| Pair 3 (inoculated) | ~2369 mV | ~2180 mV | Stable throughout |
| Pair 6 ctrl | ~2396 mV | drifting | Floating input drift — 1MΩ bias resistors added |
| Pair 7 ctrl | ~2393 mV | drifting | Floating input drift — 1MΩ bias resistors added |

**Events:**
- Control channels drifted far from baseline (dry floating inputs on high-impedance INA128)
- 1MΩ bias resistors added to both control INA128s (pin 2 → REF, pin 3 → REF) to anchor inputs
- Resistor mismatch amplified by 51× gain causes ~560 mV offset — controls stabilized but not at exact REF
- Extensive multimeter probing + wire pulls disturbed circuit in evening
- Pair 2 VOUT wire briefly disconnected from Arduino A0 during diagnosis, reconnected
- Circuit left overnight to settle before Day 4 check

**Day 4 expected:** 2026-03-09 ~12:41 PM PST — watch for slow oscillations on pairs 2, 3 diverging from controls 6, 7

## Day 4 — 2026-03-09

| Channel | Voltage | Notes |
|---|---|---|
| Pair 2 | ~3400–3520 mV | Elevated +1050 mV above baseline — residual disturbance or early agar electrochemistry |
| Pair 3 | ~2372–2387 mV | Rock solid at baseline |
| Pair 6 ctrl | ~4238 mV ±2 mV | Bias resistors working — ultra stable |
| Pair 7 ctrl | ~3888–3966 mV | Somewhat stable |

No clear biological signal yet. Pair 2 elevation resolved overnight (disturbance, not signal).

## Day 5 — 2026-03-10

| Channel | Voltage | Notes |
|---|---|---|
| Pair 2 | ~2379–2380 mV | Back at baseline — Day 4 elevation confirmed as disturbance |
| Pair 3 | ~2371–2372 mV | Rock solid at baseline, ±1 mV |
| Pair 6 ctrl | ~1122–1164 mV | Drifted lower — trace moisture on dry electrode pads |
| Pair 7 ctrl | ~1885–1976 mV | Drifted lower |

Clean Day 5 baseline on both inoculated channels. Controls less reliable due to humidity.

## Day 7 — 2026-03-12

| Channel | Voltage | Notes |
|---|---|---|
| Pair 2 | ~3893–3960 mV, rising | **Elevated +1550 mV above baseline, active trend** |
| Pair 3 | ~2094–2107 mV | Near baseline, stable |
| Pair 6 ctrl | ~4242–4244 mV | Near ADC ceiling, drifted up |
| Pair 7 ctrl | ~4219–4248 mV | Near ADC ceiling, drifted up |

Pair 2 significantly elevated compared to flat pair 3. Controls near ceiling — unreliable.
Need full time-series analysis to confirm whether pair 2 rise is gradual (signal) or sudden (disturbance).

## Data Transfer & Analysis — 2026-03-12

- All 21 CSV files (1.34 GB, recording_20260220 through recording_20260312) transferred from Computer B to Computer A via rsync
- `analyze_recording.py` rewritten for 4-channel 14-column format — generates 5 PNG plots (added zoom window)
- Analysis files saved to `~/Downloads/mycelium_data/analysis/`
- Relevant experiment window: recording_20260305.csv through recording_20260312.csv

---

## Signal Analysis — Day 7 (2026-03-12)

### Channel Summary

| Channel | Raw Day 7 | Filtered 3σ (output) | At-Electrode (÷51) | vs Literature |
|---|---|---|---|---|
| Pair 2 (inoculated, one-side LC) | ~3930 mV, rising | ±1442 mV | ~28 mV | ~15× above (0.5–2.1 mV) |
| Pair 3 (inoculated, both-side LC) | ~2100 mV, flat | ±283 mV | ~5.5 mV | ~2–3× above |
| Pair 6 (control) | ~4242 mV | — | — | Unreliable (ADC ceiling) |
| Pair 7 (control) | ~4230 mV | — | — | Unreliable (ADC ceiling) |

### PSD Analysis

| Channel | Peak Frequency | Period | PSD at Peak | Shape |
|---|---|---|---|---|
| Pair 2 | 0.039 Hz | ~25 seconds | ~2×10⁶ mV²/Hz | 1/f (pink noise) |
| Pair 3 | 0.029 Hz | ~34 seconds | ~10⁵ mV²/Hz | 1/f (pink noise) |

- Pair 2 has **~20× more spectral power** than pair 3 across the full 0–2 Hz band
- Both peak frequencies are within the fungal signal range (0.01–1 Hz per Buffi/Adamatzky)
- 1/f spectral shape is characteristic of both biological slow oscillations AND slow electrochemical drift — cannot distinguish from PSD alone

### Interpretation

**Why is pair 2 amplitude so large (~28 mV at electrode vs 0.5–2.1 mV expected)?**

Two non-mutually-exclusive explanations:

1. **Agar scaffold electrochemistry (likely dominant):** The agar+sugar scaffold deposited on pair 2's electrode pads creates a slow-evolving ionic concentration gradient. As the agar equilibrates, dries at the edges, and is modified by any microbial activity, the ionic potential at the electrode-solution interface shifts over hours and days. This slow evolution directly shows up in the 0.01–2 Hz band and would produce exactly this type of large-amplitude, broadband signal increase.

2. **Asymmetric LC application (amplified by differential measurement):** Pair 2 had LC applied to the positive electrode only (per Buffi protocol — fungus grows toward second electrode). Pair 3 had LC on both sides (accidental squirt). The INA128 measures the *difference* between IN+ and IN−. If LC/agar creates different electrochemical potentials at the two electrodes (which it does — wet vs dry, active vs passive), the asymmetry is amplified 51× into the output. Pair 3's symmetric environment created near-equal potentials at both inputs → difference ≈ 0 → lower output variation.

3. **Possible early mycelial contribution (cannot rule out):** The elevation began Day 5–6, matching Buffi's Day 4+ prediction. If mycelium has colonized the agar near electrode pads on pair 2 but not yet pair 3, the growing hyphal network would alter the electrode-solution interface impedance and create slow oscillations. Cannot separate from agar electrochemistry at this stage.

### What Would Confirm Biological Origin

For the paper, the key discriminator between agar electrochemistry and biological signal is **temporal structure**:

- **Agar electrochemistry:** slow monotonic drift, no periodic structure, should stabilize as agar fully dries/equilibrates
- **Biological signal:** quasi-periodic slow oscillations (consistent period ~25–60s per Adamatzky), irregular spike trains, should *increase* in amplitude and regularity as colonization progresses

**Action:** Run `05_zoom_*.png` on different 12–24 hour windows from Day 5 through Day 7 on pair 2. If the filtered signal shows recurring peaks with consistent spacing (~25s), that's biological. If it's aperiodic continuous variance, that's electrochemical drift.

---

## Control Measurement Improvement Plan

### Current Problem
Control electrode pairs (6, 7) have dry bare FR4 copper pads → extremely high impedance (~GΩ) → INA128 input bias current (~5 nA) slowly charges stray capacitance → output drifts to ADC rail over hours. 1MΩ bias resistors partially anchored the inputs but resistor mismatch (±5% tolerance) × 51× gain = up to 560 mV systematic offset. Both controls sat near ADC ceiling (~4240 mV) throughout Days 5–7 and are effectively unusable.

### Immediate Fix (can do during Day 8–11 window)

**Apply sterile agar (no LC) to pairs 6 and 7 electrode pads.**

- Prepare sterile agar solution: same agar+sugar recipe used for pairs 2/3, but **do not add any LC inoculum**
- Deposit ~20 µL droplet onto each electrode pad (same volume and placement as inoculated pairs)
- Close cage immediately, let equilibrate 15–30 min before checking

**Why this works:**
- Ionic solution provides a low-impedance conductive path for bias current — eliminates floating input drift
- The 1MΩ bias resistors can be **removed** once agar is applied (they're no longer needed)
- Removes the ~560 mV resistor-mismatch offset
- Creates a true **substrate control**: same agar chemistry, same humidity exposure, no fungus

**Expected result after applying sterile agar:** controls should return to near-REF (2230 mV) ± small offset from agar ionic potential (~50–100 mV). This gives a valid baseline to compare against pairs 2/3.

### Ranking of Control Options

| Option | Effort | Controls For | Notes |
|---|---|---|---|
| **Sterile agar on pairs 6/7** | Low (do now) | Substrate chemistry, humidity | Best immediate fix |
| Replace 1MΩ with 0.1% matched resistors | Low (order parts) | Bias drift offset | Use only if agar option fails |
| Swap RG to 10kΩ on control INA128s | Low | Gain headroom for drift | Gain drops 51× → 6× |
| **Heat-killed inoculum control** | Medium (next experiment) | Full biological chemistry | Gold standard for paper |
| Tied-input hardware zero reference | Low (wire change) | Amp offset + noise floor | Good sanity check channel |

### For the Paper

The honest statement is: *"Control channels exhibited floating-input drift due to high electrode-solution impedance in dry conditions; controls were stabilized using substrate-matched wet controls (sterile agar, no inoculum) from Day 8 onward."* This is a legitimate methodological adaptation documented in the literature.

---

*Second attempt log started 2026-03-05. Analysis section added 2026-03-12.*
