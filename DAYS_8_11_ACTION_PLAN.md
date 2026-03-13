# Days 8–11 Action Plan — EE297B Mycelium Experiment
**Anthony Contreras & Alex Wong | SJSU MSEE**
**Current status: Day 7 of second inoculation (2026-03-12). Experiment ends ~Day 14.**

---

## Context

- **Pair 2** (one-side LC inoculation): elevated +1550 mV by Day 7, 5× more signal energy than pair 3
- **Pair 3** (both-side LC): flat, stable, near baseline — good internal reference
- **Controls (pairs 6, 7)**: near ADC ceiling, unreliable — fix described below
- **Deadline**: mid-semester (March/April) — paper + demo + presentation
- **Repo**: https://github.com/theeanthony/EE298 (pull latest for analysis code + plots)

---

## Priority 1 — Fix the Controls (Anthony, 30 min, Day 8)

Apply **20 µL sterile agar (no LC)** to pairs 6 and 7 electrode pads.

- Same agar+sugar recipe used for pairs 2/3, but with **zero inoculum**
- Removes floating-input drift (ionic solution provides impedance path for INA128 bias current)
- Remove the 1MΩ bias resistors from pairs 6/7 INA128s after agar is applied — they're no longer needed and cause ~560 mV offset
- Controls should stabilize near 2230 mV (±~100 mV) within a few hours

**Why this matters for the paper:** Without valid controls, the results section has no comparison baseline. This is the single highest-value 30 minutes of the experiment.

---

## Priority 2 — Run Zoom Plots to Characterize Pair 2 (Either, 20 min, Day 8)

Run analyze_recording.py three times on different time windows to look for slow-wave periodicity in pair 2.

```bash
# Day 5 window (before elevation started)
python3 software/processing/analyze_recording.py ~/Downloads/mycelium_data/recording_2026030*.csv \
  --output-dir ~/Downloads/mycelium_data/analysis \
  --start-date 2026-03-05 \
  --zoom-start 2026-03-09T12:00:00+00:00 --zoom-hours 12

# Day 6 window (during elevation)
python3 software/processing/analyze_recording.py ~/Downloads/mycelium_data/recording_2026030*.csv \
  --output-dir ~/Downloads/mycelium_data/analysis \
  --start-date 2026-03-05 \
  --zoom-start 2026-03-10T12:00:00+00:00 --zoom-hours 12

# Day 7 window (latest data)
python3 software/processing/analyze_recording.py ~/Downloads/mycelium_data/recording_2026030*.csv \
  --output-dir ~/Downloads/mycelium_data/analysis \
  --start-date 2026-03-05 \
  --zoom-start 2026-03-11T12:00:00+00:00 --zoom-hours 12
```

**What to look for in the filtered panels (panels 2 and 4 of each plot):**
- Repeating peaks at consistent ~25–60s intervals → **biological signal** (temporal coherence)
- Continuous high-amplitude noise with no rhythm → **electrochemical drift** (agar scaffold)
- This single observation determines how pair 2 is framed in the paper's Discussion section

---

## Priority 3 — Add STFT Spectrogram to analyze_recording.py (Alex, 2–3 hrs, Days 8–9)

The Buffi et al. 2025 paper (which this experiment replicates) uses **STFT as the primary signal characterization method**. We need this plot for the paper.

Target output: a 2-panel spectrogram (pair 2 top, pair 3 bottom)
- X-axis: time (days)
- Y-axis: frequency (0–2 Hz)
- Color: power at each frequency over time
- Expected signature: if biological activity is present, the <0.1 Hz band should "light up" on pair 2 after Day 4 while pair 3 stays dark

Ask Claude Code to implement this — prompt: *"Add a plot_stft_spectrogram() function to analyze_recording.py that generates a 2-panel STFT spectrogram for pair 2 and pair 3 using scipy.signal.spectrogram with ~10-minute windows sliding every 1 minute. Frequency axis 0–2 Hz. Use log-scale color (dB). Save as 06_stft_spectrogram.png."*

---

## Priority 4 — Stimulus-Response Test (Anthony, Days 9–10)

The only way to **confirm biological origin** is stimulus-response: apply a stimulus, observe if inoculated pair responds and control does not.

**Protocol:**
1. Day 8: let signal run completely undisturbed (establish clean baseline)
2. Day 9, ~noon PST: send a 5-second misting event via serial command
3. Record exact UTC time of mist in lab notebook
4. Monitor CSV data for the 30–60 minutes before and after the event
5. If pair 2 shows a deflection 5–30 min post-mist while pair 3 and controls do not → **confirmed biological response**, publishable

**Send misting command from Computer B:**
```bash
python3 -c "
import serial, time
s = serial.Serial('/dev/cu.usbmodemF412FA6FA0802', 115200)
time.sleep(2)
s.write(b'M')
print('Mist ON')
time.sleep(5)
s.write(b'm')
print('Mist OFF')
s.close()
"
```

**Critical:** Note the exact UTC timestamp. Cross-reference against the CSV `wall_clock_utc` column afterward.

This is the same class of experiment as the Cornell mycelium robot paper (UV-light stimulus → bioelectric response). It's the strongest evidence of biological origin possible with the current setup.

---

## Priority 5 — Start Writing the Paper (Both, Days 8–11)

You have enough data to write ~80% of the paper right now. Don't wait for Day 14.

### Section Split

| Section | Who | Source Material |
|---|---|---|
| Introduction + Related Work | Alex | `research-findings.md` (already a full lit review) |
| Methods: Hardware | Anthony | `INOCULATION_LOG.md` + `CLAUDE.md` hardware section |
| Methods: Software/DSP | Alex | `analyze_recording.py`, `signal_processor.py` |
| Results: Signal characterization | Both | Day 7 analysis plots (already in `data/analysis/`) |
| Results: ML pipeline | Alex | Needs training run on real data (see Priority 6) |
| Discussion | Both | After Day 11 data is in |

### Figures You Already Have

| # | File | Caption |
|---|---|---|
| 1 | *(draw)* | Hardware block diagram — signal chain from electrode to ADC |
| 2 | `data/analysis/01_overview_all_channels.png` | Full 7-day 4-channel recording |
| 3 | `data/analysis/02_inoculated_comparison.png` | Pair 2 vs Pair 3 voltage divergence |
| 4 | `data/analysis/03_filtered_signals.png` | Bandpass-filtered signal energy (3σ comparison) |
| 5 | `data/analysis/04_power_spectral_density.png` | PSD: pair 2 has 20× more power, peak at 0.039 Hz |
| 6 | *(generate)* | STFT spectrogram — frequency content over time |
| 7 | *(generate)* | Zoom window — fine structure of pair 2 slow oscillations |

---

## Priority 6 — Train ML Pipeline on Real Data (Alex, 2–3 hrs, Days 9–10)

The pipeline was designed for simulated data. Run it on the real 7-day recording.

**Label strategy (weak labels):**
- Days 1–3 (pairs 2 and 3) = `background`
- Days 5–7 pair 2 = `candidate_signal`
- Days 5–7 pair 3 = `background_wet` (inoculated but quiet — useful negative)

Even with weak labels and moderate F1, demonstrating the full pipeline on live mycelium data satisfies the "working demo" deliverable.

Ask Claude Code to write a CSV-to-windowed-array converter: *"Write a script that reads the 14-column recording CSVs, extracts pair 2 voltage column (v_p2_mV), segments into 600-sample windows (60s at 10 Hz) with 50% overlap, and saves as a .npz file compatible with ml/train.py --from-cache."*

---

## Day-by-Day Summary

### Days 8–9 (Anthony)
- [ ] Apply sterile agar to pairs 6/7 — remove 1MΩ bias resistors after
- [ ] Run 3× zoom window plots (Day 5, 6, 7 windows)
- [ ] Write Methods: Hardware section of paper

### Days 8–9 (Alex)
- [ ] Pull latest from GitHub (`git pull origin main`)
- [ ] Download raw CSVs from Google Drive link (Anthony will share)
- [ ] Add STFT spectrogram to `analyze_recording.py`
- [ ] Write Introduction + Related Work section of paper

### Days 10–11 (Anthony)
- [ ] Stimulus-response misting test (Day 9 or 10 noon)
- [ ] Sync final CSVs from Computer B: `rsync -av --progress "anthonycontreras@192.168.12.103:~/EE298/data/raw/recording_202603*.csv" ~/Downloads/mycelium_data/`
- [ ] Write Results: Signal characterization section (use the 5 existing figures)

### Days 10–11 (Alex)
- [ ] Train ML pipeline on real data (weak labels)
- [ ] Write Methods: Software/DSP section
- [ ] Generate STFT spectrogram figure

---

## What Makes the Strongest Paper

Ranked by impact:

1. **Valid controls** (sterile agar on 6/7) — without this, results section has no baseline
2. **STFT spectrogram** — Buffi's core methodology, expected figure in the paper
3. **Stimulus-response test** — only proof of biological origin, very publishable
4. **Zoom window periodicity check** — determines Discussion framing for pair 2
5. **ML pipeline on real data** — completes the "full pipeline" claim

---

## Key Analysis Numbers (Day 7)

| Metric | Pair 2 (inoculated) | Pair 3 (inoculated) | Literature |
|---|---|---|---|
| Raw voltage Day 7 | ~3930 mV (+1550 mV above baseline) | ~2100 mV (near baseline) | — |
| Filtered 3σ (output) | ±1442 mV | ±283 mV | — |
| At-electrode equivalent | ~28 mV | ~5.5 mV | 0.5–2.1 mV |
| PSD peak frequency | 0.039 Hz (~25s period) | 0.029 Hz (~34s period) | 0.01–1 Hz |
| Relative PSD power | 20× higher than pair 3 | baseline | — |

Pair 2 amplitude (~28 mV at electrode) is ~15× above published fungal spike values. Most likely cause: agar scaffold electrochemistry evolving over days, amplified by asymmetric LC application (one-side on pair 2 vs both-sides on pair 3). Biological contribution cannot be ruled out — stimulus-response test is the discriminator.

---

*Generated 2026-03-12 | Experiment Day 7*
