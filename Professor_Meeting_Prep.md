# Professor Meeting Preparation
## EE297B Research Project - Signal Processing Fungi Propagation
### Anthony Contreras & Alex Wong | Response to Prof. Juzi

---

## 1. Hardware Signal-Acquisition Chain

### Our Planned Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────┐
│  FPC Electrodes │────▶│  Instrumentation│────▶│   Data Logger   │────▶│   PC/Python │
│  (ENIG, 8 pairs)│     │   Amplifier     │     │   (ADC)         │     │  Processing │
│                 │     │   (AD8237)      │     │                 │     │             │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────┘
        │                       │                       │
        │                       │                       │
   Differential            Gain: 1000x            24-bit or 14-bit
   measurement             0.5mV → 500mV          resolution
```

### Component Details

| Stage | Component | Specification | Status |
|-------|-----------|---------------|--------|
| **Electrodes** | Custom FPC with ENIG finish | 8 differential pairs, 2mm pads, 10mm spacing | PCB ordered ✓ |
| **Amplifier** | AD8237 (instrumentation amp) | Gain 1000x, 106dB CMRR, zero-drift | Need to order |
| **Shielding** | Faraday cage | Aluminum enclosure + copper tape | Need to build/acquire |
| **Data Logger** | Options below | 14-24 bit resolution | Need to determine |
| **Software** | Python + Arduino | STFT analysis (from Buffi paper) | Code ready ✓ |

---

## 2. Expected Signal Amplitude Range

### From Literature (Buffi et al. 2025, Adamatzky et al.)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Amplitude** | 0.5 - 2.1 mV | Extracellular, differential measurement |
| **Frequency** | 0.01 - 1 Hz | Very slow oscillations |
| **Spike duration** | Seconds to minutes | Not millisecond-scale like neurons |
| **DC offset** | Variable | Electrode drift expected |

### After Amplification (AD8237, G=1000)

| Input | Output | ADC Requirement |
|-------|--------|-----------------|
| 0.5 mV | 500 mV | Easily readable |
| 2.1 mV | 2.1 V | Well within range |

---

## 3. Plan to Reduce Noise and Interference

### Noise Sources & Mitigations

| Noise Source | Mitigation Strategy |
|--------------|---------------------|
| **60 Hz mains** | Faraday cage, differential measurement, software notch filter |
| **EMI/RFI** | Shielded cables, Faraday cage, short cable runs |
| **Electrode drift** | High-pass filter (>0.01 Hz), AC coupling |
| **Amplifier noise** | AD8237 has zero-drift architecture, low 1/f noise |
| **Ground loops** | Single-point grounding, USB isolator |
| **Agar artifacts** | Buffi method: liquid droplet inoculation, NO agar on electrodes |

### Shielding Plan

```
┌────────────────────────────────────────────┐
│            FARADAY CAGE                    │
│  (metal enclosure, earth grounded)         │
│                                            │
│   ┌─────────┐      ┌─────────────┐        │
│   │   FPC   │─────▶│   AD8237    │        │
│   │ + humid │      │   Amp PCB   │        │
│   │ chamber │      └──────┬──────┘        │
│   └─────────┘             │               │
│                           │               │
└───────────────────────────┼───────────────┘
                            │ Shielded cable
                            ▼
                    ┌───────────────┐
                    │  Data Logger  │
                    │  (outside)    │
                    └───────────────┘
```

---

## 4. Data Logger Options Analysis

### What Buffi et al. Used

**PicoLog ADC-24** (Pico Technology, UK)
- 24-bit resolution
- 8 differential inputs
- Input impedance: 2 MΩ
- Galvanic isolation
- Price: **~$800-900 USD**

### Our Options

| Option | Resolution | Differential? | Price | Pros | Cons |
|--------|------------|---------------|-------|------|------|
| **PicoLog ADC-24** | 24-bit | Yes (8 ch) | ~$850 | Gold standard, matches paper | Expensive |
| **Arduino Uno R4** | 14-bit | No (but amp provides) | ~$30 | Already have, 0.3mV resolution | Less precision |
| **ADS1256 module** | 24-bit | Yes (4 ch) | ~$15-25 | High resolution, cheap | Requires integration work |
| **ADS1115 module** | 16-bit | Yes (2 ch) | ~$5-10 | Easy I2C, PGA built-in | Lower resolution |

### Recommendation

**For initial proof-of-concept:** Use Arduino Uno R4 (14-bit ADC)
- We already have it
- After AD8237 amplification (1000x), 0.5mV becomes 500mV → easily readable
- 14-bit gives 0.3mV resolution at 5V reference → adequate for amplified signal

**For publication-quality data:** Request PicoLog ADC-24 from department
- Matches methodology of Buffi et al.
- Better for reproducibility claims
- Professional software for long-term logging

**Budget alternative:** ADS1256 24-bit ADC module (~$20)
- Same 24-bit resolution as ADC-24
- Requires custom integration with Arduino/Pi
- Good middle ground

---

## 5. Questions for Professor Juzi

1. **Lab equipment access:** Can we borrow/access a Faraday cage from the EE labs? Or should we build one from a metal enclosure?

2. **Data logger:** Does the department have a PicoLog ADC-24 or equivalent precision data logger we could use? If not, is the Arduino Uno R4 (14-bit) acceptable for initial recordings?

3. **Amplifier:** We're planning to use AD8237 instrumentation amplifier. Does the department have any in stock, or should we order?

4. **Budget:** Is there any project budget available for components (~$100-150 for amp, connectors, shielding materials)?

---

## 6. Current Progress Summary

### Completed ✓
- [x] FPC board designed and ordered (ENIG finish, 8 electrode pairs)
- [x] Signal simulation working (voltage divider + Arduino R4)
- [x] Python acquisition code written and tested
- [x] STFT analysis code identified (from Buffi paper, ready to adapt)
- [x] Blue Oyster liquid culture source identified

### In Progress
- [ ] Ordering AD8237 amplifier + SOIC-to-DIP adapter
- [ ] Ordering liquid culture for inoculation
- [ ] Acquiring/building Faraday cage
- [ ] Determining final data logger solution

### Next Steps (Next 2 Weeks)
1. Receive FPC boards and solder headers/connectors
2. Build amplifier circuit on breadboard
3. Acquire Faraday cage (build or borrow)
4. Inoculate first FPC with liquid culture
5. Attempt first recordings (even if noisy)

---

## 7. Timeline to First Recording

| Week | Milestone |
|------|-----------|
| Week 1-2 | Receive FPC, build amplifier circuit, acquire shielding |
| Week 2-3 | Inoculate FPC electrodes with Blue Oyster |
| Week 3-5 | Wait for mycelium to bridge electrode gap |
| Week 5-6 | **First recording attempt** |

**Biological bottleneck:** Mycelium colonization takes 2-4 weeks. We're starting inoculation ASAP.

---

## 8. Draft Email Response to Professor

> Dear Professor Juzi,
>
> Thank you for your guidance. We agree that achieving the first real recording is the highest priority milestone.
>
> **Our signal acquisition chain:**
> - Electrodes: Custom FPC with ENIG finish (8 differential pairs) - ordered
> - Amplifier: AD8237 instrumentation amplifier (gain 1000x) - ordering this week
> - Data logger: Arduino Uno R4 (14-bit ADC) for initial tests; we'd like to discuss if the department has access to a higher-precision logger like the PicoLog ADC-24 used in the Buffi paper
> - Shielding: Planning to build/acquire Faraday cage - would appreciate guidance on lab resources
>
> **Expected signal range:** 0.5-2.1 mV (from literature). After 1000x amplification → 500mV-2.1V, well within ADC range.
>
> **Noise reduction plan:**
> - Faraday cage for EMI shielding
> - Differential measurement (rejects common-mode noise)
> - Shielded cables
> - Software filtering (notch at 60Hz, bandpass 0.01-2Hz)
> - Buffi's liquid inoculation method (avoids agar artifacts on electrodes)
>
> **Questions for our meeting:**
> 1. Does the department have a Faraday cage or precision data logger (ADC-24) we could use?
> 2. Is there any project budget for components (~$100-150)?
>
> We'll have the FPC boards and amplifier circuit ready to demonstrate at our meeting.
>
> Best regards,
> Anthony & Alex

---

*Prepared: January 28, 2026*
