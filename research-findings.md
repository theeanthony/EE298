# Research Findings & Dataset Inventory

## EE297B — Signal Processing for Fungi Propagation
**Last updated:** February 16, 2026

---

## Literature Review

### Primary References

| Reference | Key Findings | Relevance |
|-----------|-------------|-----------|
| Buffi et al. (2025) iScience | FPC with ENIG electrodes, STFT analysis, differential pairs, Faraday cage. Signals from *Pleurotus ostreatus* and *Fusarium oxysporum*. Signal change at day 4 post-inoculation. | **Core methodology** — our project replicates this setup |
| Buffi et al. (2025) FEMS Microbiol Rev | Comprehensive review of fungal electrophysiology | Background/context |
| Adamatzky (2022) Royal Society Open Science | "Language of fungi" — spike trains from 4 species (0.03–2.1 mV, duration 1–21 hrs). Proposed fungal spike "words" via clustering. | Spike characterization, dataset source |
| Schyck et al. (2024) Global Challenges | Fungal signaling in living composites | Supplementary context |
| Cornell / Science Robotics (2024) | *Pleurotus eryngii* bioelectric signals used for robot sensorimotor control. UV-light stimulus response. | Stimulus-response dataset, analysis code |
| Fukasawa et al. (2023) | Field recordings from *Laccaria bicolor* sporocarps — >100 mV potentials triggered by rainfall | Environmental response signals |

### Signal Parameters (consensus from literature)

| Parameter | Value | Source |
|-----------|-------|--------|
| Amplitude | 0.5–2.1 mV peak | Adamatzky, Buffi |
| Frequency | 0.01–1 Hz | Literature consensus |
| Waveform | Irregular spikes/oscillations | Not sinusoidal |
| Spike duration | 1–21 hours | Adamatzky (varies by species) |
| Colonization to signal | ~4 days post-inoculation | Buffi et al. |
| DC offset | Variable (electrode drift) | Expected behavior |

### Ionic Basis of Fungal Signals
- Resting potential: -100 to -200 mV (driven by H+-ATPase proton pump)
- Spikes involve Ca2+, Cl-, K+ channels
- Calcium acts as master regulator — tip-high gradient linked to growth
- Signal propagation may couple with hydraulic mass flow

---

## Publicly Available Datasets

### Tier 1: Fungal Electrophysiology (Target Domain)

#### Adamatzky — Language of Fungi (Zenodo)
- **URL:** https://zenodo.org/records/5790768
- **Species:** Ghost fungi, Enoki, Split gill, Caterpillar fungi (4 species)
- **Signal:** 0.03–2.1 mV, 8 electrode pairs
- **Sample rate:** 1 Hz
- **Duration:** 1.5–5 days per species
- **Format:** .txt.zip (4 files)
- **Size:** 84.6 MB
- **License:** CC-BY 4.0

#### Cornell Mycelium Robot Control (Zenodo)
- **URL:** https://zenodo.org/records/12812074
- **GitHub:** https://github.com/Alchemist77/fungi-data-analasis
- **Species:** *Pleurotus eryngii* (King Oyster)
- **Signal:** Native spiking + UV-light-stimulated recordings
- **Format:** RAR archive + Python analysis code
- **Size:** 172 MB compressed / 11.9 GB uncompressed
- **Relevance:** Same genus as our *P. ostreatus*, includes stimulus-response pairs

#### Buffi et al. — Mendeley Data
- **URL:** https://data.mendeley.com/datasets/srkxbkh6sp/1
- **Species:** *Fusarium oxysporum* (also *Pleurotus* in paper)
- **Signal:** Raw voltage potentials, pre/post biocide stimuli
- **Format:** Downloadable files
- **Relevance:** Exact methodology we're replicating (FPC + STFT)

### Tier 2: Plant Electrophysiology (Domain Adaptation)

#### Library of Electrophysiological Responses in Plants (2024)
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10950275/ (see Online Resources for data)
- **Species:** 16+ species (Venus flytrap, Mimosa pudica, tomato, basil, etc.)
- **Signal:** Action potentials from tactile/flame stimulation
- **Sample rate:** 10 kHz (as .wav files)
- **Bandpass:** 0.07–8.8 Hz (overlaps our 0.01–2.0 Hz)
- **Count:** 398 recordings over 7 months
- **Format:** .wav + .txt event markers
- **License:** CC-BY
- **Note:** Downsample from 10 kHz to 10 Hz for transfer learning

### Tier 3: ECG / Muscle Signals (Pre-training)

#### ECG Heartbeat Categorization (Kaggle)
- **URL:** https://www.kaggle.com/datasets/shayanfazeli/heartbeat
- **Source:** Preprocessed from MIT-BIH + PTB databases
- **Samples:** 109,446 (MIT-BIH, 5 classes) + 14,552 (PTB, 2 classes)
- **Sample rate:** 125 Hz (downsampled)
- **Length:** 188 samples per heartbeat (fixed-length, zero-padded)
- **Format:** CSV
- **Size:** ~104 MB
- **Relevance:** Already in 1D-CNN format. Paper demonstrates transfer learning across ECG domains.

#### MIT-BIH Arrhythmia Database (PhysioNet)
- **URL:** https://physionet.org/content/mitdb/1.0.0/
- **Records:** 48 half-hour 2-channel ECG, 360 Hz, ~110K beat annotations
- **Size:** 104 MB
- **License:** ODC-BY 1.0

#### Apnea-ECG Database (PhysioNet)
- **URL:** https://physionet.org/content/apnea-ecg/1.0.0/
- **Records:** 70 recordings (7–10 hrs each)
- **Relevance:** Respiratory modulation creates 0.15–0.5 Hz oscillations — overlaps fungal range

### Tier 4: EEG / Brain Signals (Pre-training)

#### Sleep-EDF Expanded (PhysioNet)
- **URL:** https://physionet.org/content/sleep-edfx/1.0.0/
- **Records:** 197 whole-night polysomnographic recordings
- **Sample rate:** 100 Hz
- **Relevance:** Delta waves at 0.5–2 Hz directly overlap our fungal signal range. Sleep staging = oscillation pattern classification.
- **Size:** ~7.7 GB

#### CHB-MIT Scalp EEG — Seizure Detection (PhysioNet)
- **URL:** https://physionet.org/content/chbmit/1.0.0/
- **Records:** 22 subjects, 198 annotated seizures
- **Sample rate:** 256 Hz
- **Relevance:** Seizure onset detection ≈ fungal spike detection (detect abnormal electrical events in continuous time series)
- **Size:** 42.6 GB

#### EEG Motor Movement/Imagery (PhysioNet)
- **URL:** https://physionet.org/content/eegmmidb/1.0.0/
- **Records:** 109 subjects, 64-channel, 160 Hz
- **Size:** ~3.4 GB

#### BCI Competition IV — Dataset 2a
- **URL:** https://www.bbci.de/competition/iv/
- **Records:** 9 subjects, 22 EEG + 3 EOG, 250 Hz
- **Relevance:** Standard benchmark for evaluating 1D-CNN/TCN architectures on EEG

---

## Key Insight: Why Transfer Learning Works Here

All bioelectrical signals share fundamental properties:
1. **Spike-train patterns** — discrete depolarization events in continuous time series
2. **Low-frequency oscillations** — slow rhythmic patterns (seconds to minutes)
3. **Noise characteristics** — environmental interference, electrode drift, motion artifacts
4. **Detection task** — find events of interest in noisy, continuous recordings

A model pre-trained to detect spikes in ECG/EEG learns general temporal feature extraction that transfers to fungal spike detection, even though the biological source differs.
