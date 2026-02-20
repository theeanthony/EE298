# Inoculation Prep — Verification & Launch Checklist
**EE297B | Anthony Contreras & Alex Wong | SJSU**
**Started: 2026-02-20 | Target inoculation: ASAP this week**

---

## Setup Overview

| Machine | Role | What runs on it |
|---|---|---|
| **Computer A** | Your main Mac (this one) | Arduino IDE — flash firmware only |
| **Computer B** | Older laptop (dedicated logger) | `start_logger.sh` — runs 24/7 unattended |

The Arduino stays physically connected to **Computer B** during the entire inoculation run.
Computer A is only needed to flash firmware once before you hand off the Arduino.

---

## Phase 0 — Hardware Prep (Before Any Firmware Flash)

Do this at your bench now, before connecting anything to Computer B.

### Step 0A — Place decoupling caps on INA128PA (mandatory)

These prevent power-rail noise from coupling into your mV-level signal.
At 51× gain, skipping this step means power supply hash dominates your noise floor.

```
Materials needed: 2× 0.1 µF ceramic caps, breadboard
```

**INA128PA DIP-8 pinout — power pins:**
```
        ┌──────U──────┐
  RG    │ 1         8 │ RG
  IN−   │ 2         7 │ V+  ← +5V from Arduino (NOT GND)
  IN+   │ 3         6 │ VOUT → Arduino A0
  V−    │ 4         5 │ REF
        └─────────────┘
  ↑ GND (single supply)
```

**⚠ CRITICAL: Pin 7 is V+ — it MUST connect to +5V, not GND.**
Pin 4 is V− — this one connects to GND.
Swapping these means the chip has no power and outputs nothing.

**The caps do NOT replace the power wires — they are added alongside them:**

```
Pin 7 row on breadboard:
  [ chip leg | +5V wire to Arduino | cap leg 1 | (free) | (free) ]
                                         |
                                    0.1µF cap
                                         |
                                     GND rail

Pin 4 row on breadboard:
  [ chip leg | GND wire           | cap leg 1 | (free) | (free) ]
                                         |
                                    0.1µF cap
                                         |
                                     GND rail
```

Cap 1 (pin 7): one leg in pin 7's row (same node as +5V), other leg to GND rail.
Cap 2 (pin 4): one leg in pin 4's row (same node as GND), other leg to GND rail.
→ In single-supply, cap 2 is GND-to-GND and does nothing. Still harmless. Cap 1 is the one that matters.

Optional Cap 3: one leg on pin 2 (IN−), other leg on pin 3 (IN+). Add only after signal confirmed working with live electrodes — may clip fungal band if electrode impedance is high.

**Verify with multimeter:**
- Pin 7 row → Arduino 5V pin: should read ~5V DC (not 0V)
- Pin 4 row → GND: should read 0V

### Step 0B — Wire voltage divider simulator (pair 2 position)

Use this circuit to test everything before inoculation. Mycelium comes later.

```
D9 (PWM) ──── 1MΩ ──── junction ──── 220Ω ──── GND
                           │
                      INA128 pin 3 (IN+)
```

INA128 pin 2 (IN−) → GND
INA128 pin 5 (REF) → GND
INA128 pin 4 (V−) → GND
INA128 pin 7 (V+) → +5V (Arduino 5V rail)
INA128 pin 6 (VOUT) → Arduino A0
INA128 pins 1 & 8 → 1kΩ resistor across them (RG, sets gain = 51×)

---

## Phase 1 — Flash Firmware (Computer A)

### Step 1A — Open Arduino IDE on Computer A

Open:
```
firmware/fungal_signal_acquisition/fungal_signal_acquisition.ino
```

Verify `PAIR_ID = 2` at the top of the file (line 28). Leave it.

### Step 1B — Select board and port

- Board: **Arduino Uno R4 Minima** (Tools → Board → Arduino UNO R4 Boards)
- Port: whichever port appears when Arduino is plugged into Computer A

### Step 1C — Upload

Click Upload. Wait for "Done uploading."

Open **Serial Monitor** (115200 baud). You should see exactly this startup banner:

```
# ==========================================
# Fungal Signal Acquisition System v2.0
# EE297B Research Project - SJSU
# ==========================================
# Pair ID: 2
# ADC: 14-bit (Uno R4) — 0.006 mV/count at 51x gain
# Commands: M/m=mister, F/f=fan, L/l=LED, S/s=simulator
# Format: timestamp_ms,adc_raw,voltage_mV,mister,fan,led
# ==========================================
```

If you see `10-bit` in that line, the board selection is wrong — re-check the board.

Close Serial Monitor before unplugging.

### Step 1D — Move Arduino to Computer B

Unplug from Computer A. Plug USB cable into Computer B.
Computer A is done with the Arduino for now.

---

## Phase 2 — Set Up Computer B

### Step 2A — Get the project onto Computer B

Option 1 — Git (cleanest):
```bash
# On Computer B
git clone <your-repo-url> ~/EE297B_ResearchProject
# OR if already cloned:
cd ~/EE297B_ResearchProject && git pull
```

Option 2 — USB drive / AirDrop:
Copy the entire `EE297B_ResearchProject` folder. Make sure it includes:
- `software/acquisition/long_duration_logger.py`
- `software/acquisition/start_logger.sh`

### Step 2B — Install Python dependencies on Computer B

```bash
pip3 install pyserial
```
That's the only dependency for `long_duration_logger.py`.

### Step 2C — Make the logger executable and create the data directory

```bash
cd ~/EE297B_ResearchProject
chmod +x software/acquisition/start_logger.sh
mkdir -p data/raw
```

### Step 2D — Find the Arduino port on Computer B

```bash
ls /dev/cu.usb*
# Expected output: /dev/cu.usbmodem1401  (or similar number)
```

Note the exact port — you'll pass it with `--port` to avoid auto-detect delays on restart.

---

## Phase 3 — Verification Tests

Run all 5 tests in order. Do not inoculate until all 5 pass.

---

### TEST 1 — 14-bit ADC resolution

**What you're checking:** `adc_raw` should span 0–16383, not 0–1023.

On Computer B, run the logger for 60 seconds with the voltage divider active:

```bash
cd ~/EE297B_ResearchProject
python3 software/acquisition/long_duration_logger.py \
  --port /dev/cu.usbmodem1401 \
  --pair-id 2 --adc-bits 14 --gain 51
```

Wait ~60 seconds, then Ctrl+C.

Open the CSV:
```bash
cat data/raw/recording_$(date +%Y%m%d).csv | grep -v "^#" | head -20
```

**Expected columns:**
```
seq,wall_clock_utc,timestamp_ms,adc_raw,voltage_mV,mister,fan,led
1,2026-02-20T18:00:01.234+00:00,1234,8192,2500.123,0,0,0
2,2026-02-20T18:00:01.334+00:00,1334,8240,2514.751,0,0,0
```

**Check `adc_raw` range:**
```bash
awk -F',' 'NR>1 && $1~/^[0-9]/ {print $4}' data/raw/recording_$(date +%Y%m%d).csv \
  | sort -n | awk 'NR==1{min=$1} END{print "min="min, "max="$1}'
```

**Pass:** max adc_raw is in range 6000–16383 (depends on simulator ramp position)
**Fail:** max adc_raw ≤ 1023 → firmware didn't upload correctly, or wrong board selected

**Check voltage resolution:**
At gain=51, 14-bit: 5000/16383/51 = 0.006 mV/count.
Consecutive rows should differ by ~0.006–0.06 mV, not large jumps.

---

### TEST 2 — Actuator state columns

**What you're checking:** Sending `M` (mister ON) makes `mister=1` appear in CSV within one sample (100ms).

Start a fresh logger run. In a **second terminal on Computer B:**

```bash
# Terminal 1 — logger running
python3 software/acquisition/long_duration_logger.py --port /dev/cu.usbmodem1401

# Terminal 2 — send mister ON command
echo -n "M" > /dev/cu.usbmodem1401
sleep 2
echo -n "m" > /dev/cu.usbmodem1401
```

Wait 5 seconds, Ctrl+C the logger.

Check the CSV:
```bash
awk -F',' '$6=="1"' data/raw/recording_$(date +%Y%m%d).csv | head -5
```

**Pass:** Several rows show `mister=1` in column 6, surrounded by rows with `mister=0`
**Fail:** All rows show `mister=0` → check port name in the echo command

Also verify fan and LED:
```bash
echo -n "F" > /dev/cu.usbmodem1401   # fan ON
echo -n "f" > /dev/cu.usbmodem1401   # fan OFF
echo -n "L" > /dev/cu.usbmodem1401   # LED ON
echo -n "l" > /dev/cu.usbmodem1401   # LED OFF
```

---

### TEST 3 — GAP detection on USB disconnect

**What you're checking:** Unplugging the Arduino mid-run creates a `# GAP:` comment row
in the CSV, not silent missing data.

Start the logger:
```bash
python3 software/acquisition/long_duration_logger.py --port /dev/cu.usbmodem1401
```

Wait 15 seconds so ~150 samples accumulate. Then:
1. **Physically unplug the Arduino USB** from Computer B
2. Wait 10 seconds — logger should print `No data received (timeout) — reconnecting`
3. **Replug the USB**
4. Wait 15 more seconds
5. Ctrl+C

Search for the GAP row:
```bash
grep "^# GAP" data/raw/recording_$(date +%Y%m%d).csv
```

**Pass output looks like:**
```
# GAP: disconnected ~10s, seq_before=152, reconnected_at=2026-02-20T18:15:30.123+00:00
```

**Pass:** GAP line is present with a timestamp and seq number
**Fail:** No GAP line → check that `disconnect_time` logic is triggering (logger should print warning to terminal)

Also verify data continues after the GAP:
```bash
grep -v "^#" data/raw/recording_$(date +%Y%m%d).csv | tail -10
```
Seq numbers should continue from where they left off (no reset to 1).

---

### TEST 4 — Metadata header

**What you're checking:** The first 8 lines of every new daily CSV are `#` comment metadata.

```bash
head -10 data/raw/recording_$(date +%Y%m%d).csv
```

**Pass output looks like:**
```
# experiment: mycelium_colonization_001
# start_utc: 2026-02-20T18:00:01.234+00:00
# sample_rate_hz: 10
# adc_bits: 14
# gain: 51
# pair_id: 2
# active_pairs: 2,3,4,5
# control_pairs: 6,7
seq,wall_clock_utc,timestamp_ms,adc_raw,voltage_mV,mister,fan,led
1,...
```

**Fail:** First line is `seq,wall_clock_utc,...` with no metadata → the CSV file already
existed from a previous run. Delete it and start fresh:
```bash
rm data/raw/recording_$(date +%Y%m%d).csv
python3 software/acquisition/long_duration_logger.py --port /dev/cu.usbmodem1401
```

---

### TEST 5 — 48-hour headless run (start now, inoculate after it passes)

**Start the long-duration test using start_logger.sh:**

```bash
cd ~/EE297B_ResearchProject
bash software/acquisition/start_logger.sh \
  --port /dev/cu.usbmodem1401 \
  --pair-id 2 --adc-bits 14 --gain 51
```

Leave Computer B running unattended. Come back every ~8 hours and check:

**Check 1 — Is logger still alive?**
```bash
cat data/raw/logger_status.txt
```
Expected: `connected: True`, `avg_rate_hz: ~10.0`, file size growing.

**Check 2 — Did midnight rotation work?** (check next morning)
```bash
ls -lh data/raw/recording_*.csv
```
Expected: Two files — one for today, one for tomorrow. Each has metadata header.

**Check 3 — No memory growth** (on Computer B):
```bash
ps aux | grep long_duration
```
RSS memory should be flat (not growing over hours).

**Check 4 — Simulate a USB drop mid-run:**
Unplug and replug once during the 48-hr test.
```bash
grep "^# GAP" data/raw/recording_$(date +%Y%m%d).csv
```
GAP row should appear, logging resumes automatically.

**Pass criteria for 48-hr test:**
- [ ] Logger runs continuously without manual intervention
- [ ] Daily rotation creates new file at midnight with fresh metadata header
- [ ] Status file updates every ~60s
- [ ] No crash or memory blowup
- [ ] GAP row appears after intentional USB disconnect

**Do not inoculate until this test completes.**

---

## Phase 4 — Inoculation Day (Day 0)

When Tests 1–5 all pass, proceed.

### Step 4A — Stop the test logger cleanly

On Computer B:
```
Ctrl+C
```
Logger prints `Logger stopped cleanly` and exits 0 (no auto-restart).

### Step 4B — Hardware transition: simulator → live electrodes

1. Remove the voltage divider wires from INA128 pin 3 (IN+) and pin 2 (IN−)
2. Route **pair 2** JP1 wires to INA128:
   - JP1 pin 3 → INA128 pin 3 (IN+)
   - JP1 pin 4 → INA128 pin 2 (IN−)
3. Verify continuity: multimeter between JP1 pin 3 and INA128 pin 3 (should beep)
4. Leave decoupling caps in place — they stay for the entire experiment

### Step 4C — Inoculate the PCB

Inoculate **pairs 2, 3, 4, 5** (active).
Leave **pairs 6, 7** dry — these are your control electrodes.

After inoculation, immediately close the Faraday cage.

### Step 4D — Start the production logger

```bash
cd ~/EE297B_ResearchProject
bash software/acquisition/start_logger.sh \
  --port /dev/cu.usbmodem1401 \
  --pair-id 2 --adc-bits 14 --gain 51
```

Note the wall-clock UTC start time from the metadata header — this is Day 0, T=0.

```bash
head -3 data/raw/recording_$(date +%Y%m%d).csv
# start_utc: <that timestamp is your Day 0 reference>
```

Leave Computer B running. Do not disturb.

---

## Phase 5 — Daily Monitoring Protocol

### Morning check (takes 2 minutes, from Computer A via SSH or in person)

```bash
# Check logger health
cat data/raw/logger_status.txt

# Count today's samples (should be ~864,000 per day at 10 Hz)
grep -v "^#" data/raw/recording_$(date +%Y%m%d).csv | wc -l

# Check for any GAP events
grep "^# GAP" data/raw/recording_$(date +%Y%m%d).csv
```

### Day 3–4 — When to expect first signals (per Buffi et al.)

Buffi found signal changes beginning at **day 4 post-inoculation**.
Do not worry if the first 3 days look flat — that is normal colonization time.

At day 3, plan the second amplifier addition (see below).

### Day 3 — Adding pair 3 (second amplifier)

**This is the plan-approved procedure. Follow exactly.**

1. **On Computer A:** Update firmware `PAIR_ID = 3`, add `const int ADC_PIN2 = A1;`
   — but only the channel expansion portion. Upload to Arduino.
   Arduino resets. Logger detects disconnect → writes GAP row → reconnects automatically.

2. **Confirm on Computer B:**
   ```bash
   tail -5 data/raw/recording_$(date +%Y%m%d).csv
   grep "^# GAP" data/raw/recording_$(date +%Y%m%d).csv | tail -1
   ```
   Verify GAP row written, logging resumed.

3. **Open cage:**
   - Tape/zip-tie existing pair 2 wires first — don't touch them
   - Route pair 3 JP1 wires (pins 5–6) → second INA128PA → Arduino A1
   - Close cage. Total open time: target < 2 minutes.

4. **Confirm both channels** logging in CSV.

---

## Quick Reference — Commands

### Computer B — start/stop logger

```bash
# Start (production run)
bash software/acquisition/start_logger.sh --port /dev/cu.usbmodem1401 --pair-id 2 --adc-bits 14 --gain 51

# Stop cleanly
Ctrl+C

# Check status without stopping
cat data/raw/logger_status.txt

# Watch live (last 5 rows updating)
watch -n 5 "tail -5 data/raw/recording_$(date +%Y%m%d).csv"
```

### Computer A — send actuator commands mid-run

```bash
# Replace port with your actual port on Computer A
echo -n "M" > /dev/cu.usbmodem1401   # mister ON
echo -n "m" > /dev/cu.usbmodem1401   # mister OFF
echo -n "F" > /dev/cu.usbmodem1401   # fan ON
echo -n "f" > /dev/cu.usbmodem1401   # fan OFF
echo -n "L" > /dev/cu.usbmodem1401   # LED ON (100%)
echo -n "l" > /dev/cu.usbmodem1401   # LED OFF
echo -n "?" > /dev/cu.usbmodem1401   # print status to Arduino serial
```

### Quick data sanity check

```bash
# ADC range (should be 0–16383 territory)
awk -F',' 'NR>1 && $1~/^[0-9]/{print $4}' data/raw/recording_$(date +%Y%m%d).csv \
  | sort -n | awk 'NR==1{min=$1} END{print "adc_raw range: "min" to "$1}'

# Sample count today
grep -vc "^#" data/raw/recording_$(date +%Y%m%d).csv

# Any gaps today
grep "^# GAP" data/raw/recording_$(date +%Y%m%d).csv
```

---

## Verification Checklist Summary

| Test | What it proves | Pass criteria |
|---|---|---|
| 1. 14-bit ADC | Firmware fix works | `adc_raw` max > 6000 in 60s run |
| 2. Actuator state | mister/fan/led logged | `mister=1` rows appear after `M` command |
| 3. GAP detection | Reconnect is silent-safe | `# GAP:` line present after USB unplug |
| 4. Metadata header | New CSV has experiment context | First 8 lines are `#` comments |
| 5. 48-hr headless | System stable for multi-day run | All sub-checks pass, no crash |

**Status:**
- [ ] TEST 1 — 14-bit ADC resolution — PENDING
- [ ] TEST 2 — Actuator state columns — PENDING
- [ ] TEST 3 — GAP detection — PENDING
- [ ] TEST 4 — Metadata header — PENDING
- [ ] TEST 5 — 48-hr headless run — PENDING
- [ ] Hardware caps placed (INA128 V+ and V−) — PENDING
- [ ] Firmware flashed (v2.0 banner confirmed) — PENDING
- [ ] Logger transferred to Computer B — PENDING
- [ ] **INOCULATE** — BLOCKED until all above checked

---

*Document created: 2026-02-20*
*Inoculation target: week of 2026-02-16 (already delayed — proceed ASAP)*
