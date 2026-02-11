/*
 * mycelium_signal_simulator.ino
 * Generates fake fungal bioelectrical signals for pipeline testing
 *
 * EE297B Research Project - Signal Processing Fungi Propagation
 * Anthony Contreras & Alex Wong | San Jose State University
 *
 * HARDWARE: Arduino Uno R4 + 1MΩ resistor + 220Ω resistor
 *
 * SIMPLE WIRING (recommended):
 * D9 (PWM) ----[1MΩ]----+----[220Ω]---- GND
 *                       |
 *                       +--> A0
 *
 * OUTPUT: ~0-1.1mV signal (from 5V PWM) or ~0-0.73mV (from 3.3V)
 *
 * SIGNAL MODES (0-9):
 * 0 = Slow sine wave (healthy baseline)
 * 1 = Random walk (biological noise)
 * 2 = Occasional spikes (action potentials)
 * 3 = Composite (most realistic healthy signal)
 * 4 = NOTHING / Dead signal (no mycelium / dead culture)
 * 5 = NOISE ONLY (poor electrode contact / 60Hz interference)
 * 6 = DRIFT (electrode polarization / temperature change)
 * 7 = SATURATION (amplifier clipping / signal too strong)
 * 8 = INTERMITTENT (loose connection / dying mycelium)
 * 9 = STIMULUS RESPONSE (light/touch response simulation)
 */

// ============== PIN DEFINITIONS ==============
const int PWM_PIN = 9;      // Output pin (to voltage divider)
const int ADC_PIN = A0;     // Input pin (from voltage divider)
const int LED_PIN = 13;     // Built-in LED for activity indicator

// ============== TIMING ==============
const int SAMPLE_RATE_MS = 100;  // 10 Hz sampling
unsigned long lastSample = 0;
unsigned long startTime = 0;

// ============== SIGNAL GENERATION ==============
float phase = 0;
int signalMode = 3;  // Default to composite (realistic)
int randomValue = 128;
unsigned long lastSpike = 0;
unsigned long lastEvent = 0;

// For intermittent mode
bool connectionGood = true;
unsigned long lastConnectionChange = 0;

// For stimulus response mode
bool stimulusActive = false;
unsigned long stimulusStart = 0;

// ============== STATISTICS ==============
long sampleCount = 0;
float adcSum = 0;
int adcMin = 16383;
int adcMax = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        ; // Wait for serial port
    }

    pinMode(PWM_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);

    #if defined(ARDUINO_UNOR4_MINIMA) || defined(ARDUINO_UNOR4_WIFI)
        analogReadResolution(14);  // Uno R4: Use 14-bit ADC
        Serial.println(F("# Detected: Arduino Uno R4 (14-bit ADC)"));
    #else
        Serial.println(F("# Detected: Arduino Uno R3 or compatible (10-bit ADC)"));
    #endif

    startTime = millis();
    printHeader();
}

void printHeader() {
    Serial.println(F("# ================================================"));
    Serial.println(F("# Mycelium Signal Simulator v2.0"));
    Serial.println(F("# EE297B Research Project - SJSU"));
    Serial.println(F("# ================================================"));
    Serial.println(F("#"));
    Serial.println(F("# === HEALTHY SIGNALS ==="));
    Serial.println(F("#   0 = Slow sine wave (baseline oscillation)"));
    Serial.println(F("#   1 = Random walk (biological noise)"));
    Serial.println(F("#   2 = Occasional spikes (action potentials)"));
    Serial.println(F("#   3 = Composite (realistic healthy mycelium)"));
    Serial.println(F("#"));
    Serial.println(F("# === PROBLEM SCENARIOS ==="));
    Serial.println(F("#   4 = NOTHING (dead/no mycelium)"));
    Serial.println(F("#   5 = NOISE ONLY (bad electrode/60Hz)"));
    Serial.println(F("#   6 = DRIFT (electrode polarization)"));
    Serial.println(F("#   7 = SATURATION (amplifier clipping)"));
    Serial.println(F("#   8 = INTERMITTENT (loose wire/dying)"));
    Serial.println(F("#"));
    Serial.println(F("# === EXPERIMENT SCENARIOS ==="));
    Serial.println(F("#   9 = STIMULUS RESPONSE (light/touch)"));
    Serial.println(F("#"));
    Serial.println(F("# COMMANDS: 0-9=mode, r=reset, s=stats, ?=help"));
    Serial.println(F("# OUTPUT: timestamp_ms,pwm,adc_raw,voltage_mV"));
    Serial.println(F("# ================================================"));
    Serial.print(F("# Current mode: "));
    Serial.print(signalMode);
    Serial.print(F(" ("));
    printModeName(signalMode);
    Serial.println(F(")"));
    Serial.println(F("# ================================================"));
}

void printModeName(int mode) {
    switch(mode) {
        case 0: Serial.print(F("Sine")); break;
        case 1: Serial.print(F("Random Walk")); break;
        case 2: Serial.print(F("Spikes")); break;
        case 3: Serial.print(F("Composite")); break;
        case 4: Serial.print(F("NOTHING")); break;
        case 5: Serial.print(F("NOISE")); break;
        case 6: Serial.print(F("DRIFT")); break;
        case 7: Serial.print(F("SATURATION")); break;
        case 8: Serial.print(F("INTERMITTENT")); break;
        case 9: Serial.print(F("STIMULUS")); break;
        default: Serial.print(F("Unknown")); break;
    }
}

void loop() {
    handleSerialCommands();

    unsigned long now = millis();

    if (now - lastSample >= SAMPLE_RATE_MS) {
        lastSample = now;
        sampleCount++;

        int pwmValue = generateSignal();
        analogWrite(PWM_PIN, pwmValue);

        // LED indicator
        digitalWrite(LED_PIN, (pwmValue > 127) ? HIGH : LOW);

        int adcRaw = readADCAverage(4);

        // Update statistics
        adcSum += adcRaw;
        if (adcRaw < adcMin) adcMin = adcRaw;
        if (adcRaw > adcMax) adcMax = adcRaw;

        // Convert to millivolts
        #if defined(ARDUINO_UNOR4_MINIMA) || defined(ARDUINO_UNOR4_WIFI)
            float adcMillivolts = (adcRaw / 16383.0) * 5000.0;
        #else
            float adcMillivolts = (adcRaw / 1023.0) * 5000.0;
        #endif

        // Output CSV
        Serial.print(now - startTime);
        Serial.print(",");
        Serial.print(pwmValue);
        Serial.print(",");
        Serial.print(adcRaw);
        Serial.print(",");
        Serial.println(adcMillivolts, 3);
    }
}

int readADCAverage(int numSamples) {
    long sum = 0;
    for (int i = 0; i < numSamples; i++) {
        sum += analogRead(ADC_PIN);
        delayMicroseconds(100);
    }
    return sum / numSamples;
}

int generateSignal() {
    int pwmValue = 0;

    switch (signalMode) {
        case 0: pwmValue = generateSine(); break;
        case 1: pwmValue = generateRandomWalk(); break;
        case 2: pwmValue = generateSpikes(); break;
        case 3: pwmValue = generateComposite(); break;
        case 4: pwmValue = generateNothing(); break;
        case 5: pwmValue = generateNoiseOnly(); break;
        case 6: pwmValue = generateDrift(); break;
        case 7: pwmValue = generateSaturation(); break;
        case 8: pwmValue = generateIntermittent(); break;
        case 9: pwmValue = generateStimulusResponse(); break;
        default: pwmValue = 128; break;
    }

    return constrain(pwmValue, 0, 255);
}

// ==================== HEALTHY SIGNALS ====================

int generateSine() {
    // Slow sine wave: ~0.05 Hz (20 second period)
    phase += 0.0314;
    if (phase > TWO_PI) phase -= TWO_PI;
    return 128 + (int)(100.0 * sin(phase));
}

int generateRandomWalk() {
    // Random walk with mean reversion
    randomValue += random(-5, 6);
    if (randomValue > 128) randomValue -= 1;
    if (randomValue < 128) randomValue += 1;
    randomValue = constrain(randomValue, 20, 235);
    return randomValue;
}

int generateSpikes() {
    // Low baseline with occasional spikes
    int baseline = 30;
    int pwmValue = baseline;

    if (millis() - lastSpike > (unsigned long)random(3000, 10000)) {
        pwmValue = random(180, 250);
        lastSpike = millis();
    }
    return pwmValue;
}

int generateComposite() {
    // Most realistic healthy signal
    float slowPhase = (millis() - startTime) / 60000.0 * TWO_PI;
    int base = 128 + (int)(40.0 * sin(slowPhase));
    int noise = random(-20, 21);
    int spike = 0;

    if (millis() - lastSpike > (unsigned long)random(5000, 15000)) {
        spike = random(80, 120);
        lastSpike = millis();
    }
    return constrain(base + noise + spike, 0, 255);
}

// ==================== PROBLEM SCENARIOS ====================

int generateNothing() {
    /*
     * SCENARIO: No signal / Dead mycelium / No mycelium present
     *
     * What you'd see in real life:
     * - Flat line at some DC offset
     * - Very small random noise from ADC
     * - No biological activity
     *
     * Causes:
     * - Mycelium hasn't colonized electrodes yet
     * - Mycelium is dead
     * - Electrodes not in contact with mycelium
     * - Amplifier not connected
     */

    // Flat DC with tiny ADC noise
    return 128 + random(-2, 3);
}

int generateNoiseOnly() {
    /*
     * SCENARIO: High noise, no clear signal
     *
     * What you'd see in real life:
     * - High amplitude random fluctuations
     * - Possible 60Hz component (mains interference)
     * - No discernible biological pattern
     *
     * Causes:
     * - Poor electrode contact
     * - 60 Hz mains interference (no Faraday cage)
     * - Ground loop
     * - Damaged shielding
     * - EMI from nearby electronics
     */

    // Simulate 60Hz interference (6 samples per cycle at 10Hz = aliased)
    float noise60Hz = 50.0 * sin(2 * PI * 6.0 * (millis() / 1000.0));

    // Add random broadband noise
    int broadbandNoise = random(-60, 61);

    // Small DC offset
    int dc = 128;

    return dc + (int)noise60Hz + broadbandNoise;
}

int generateDrift() {
    /*
     * SCENARIO: Slow baseline drift
     *
     * What you'd see in real life:
     * - Signal slowly moves up or down over minutes
     * - May eventually saturate at rail
     * - Small variations on top of drift
     *
     * Causes:
     * - Electrode polarization
     * - Temperature changes
     * - Electrode chemistry changing
     * - Moisture changes on electrode
     * - Long-term biological changes
     */

    // Very slow drift (over 2 minutes, drift from 50 to 200)
    float driftTime = (millis() - startTime) / 120000.0;  // 0 to 1 over 2 min
    driftTime = fmod(driftTime, 1.0);  // Repeat

    // Sawtooth drift pattern
    int drift;
    if (driftTime < 0.5) {
        drift = 50 + (int)(150.0 * (driftTime * 2));  // Rising
    } else {
        drift = 200 - (int)(150.0 * ((driftTime - 0.5) * 2));  // Falling
    }

    // Add small biological signal on top
    int bio = (int)(20.0 * sin(phase));
    phase += 0.0314;
    if (phase > TWO_PI) phase -= TWO_PI;

    return drift + bio + random(-5, 6);
}

int generateSaturation() {
    /*
     * SCENARIO: Amplifier saturation / clipping
     *
     * What you'd see in real life:
     * - Signal hits max or min and stays flat
     * - "Clipped" waveform tops/bottoms
     * - Lost information during saturation
     *
     * Causes:
     * - Gain too high on amplifier
     * - Signal amplitude larger than expected
     * - DC offset pushing signal to rail
     * - Power supply issues
     */

    // Generate a signal that's too large
    float largeSignal = 128 + 200.0 * sin(phase);
    phase += 0.05;  // Faster oscillation
    if (phase > TWO_PI) phase -= TWO_PI;

    // Clip at boundaries (simulates amplifier saturation)
    int pwm = (int)largeSignal;
    if (pwm > 250) pwm = 250;  // Clipped at top
    if (pwm < 5) pwm = 5;      // Clipped at bottom

    return pwm;
}

int generateIntermittent() {
    /*
     * SCENARIO: Intermittent signal / connection problems
     *
     * What you'd see in real life:
     * - Signal comes and goes
     * - Sudden jumps or dropouts
     * - May look normal then suddenly flat or noisy
     *
     * Causes:
     * - Loose wire connection
     * - Dying mycelium (sporadic activity)
     * - Intermittent electrode contact
     * - Cold solder joint
     * - Vibration affecting connections
     */

    // Randomly switch between good and bad connection
    if (millis() - lastConnectionChange > (unsigned long)random(2000, 8000)) {
        connectionGood = !connectionGood;
        lastConnectionChange = millis();

        if (connectionGood) {
            Serial.println(F("# [INTERMITTENT] Connection restored"));
        } else {
            Serial.println(F("# [INTERMITTENT] Connection lost!"));
        }
    }

    if (connectionGood) {
        // Normal composite signal
        return generateComposite();
    } else {
        // Disconnected: either flat or very noisy
        if (random(0, 2) == 0) {
            return random(0, 10);  // Near ground
        } else {
            return random(245, 256);  // Near rail
        }
    }
}

// ==================== EXPERIMENT SCENARIOS ====================

int generateStimulusResponse() {
    /*
     * SCENARIO: Response to environmental stimulus
     *
     * What you'd see in real life:
     * - Baseline activity
     * - Stimulus applied (light, touch, chemical)
     * - Increased activity / spike rate
     * - Gradual return to baseline
     *
     * Based on Adamatzky et al. research showing
     * mycelium responds to light and mechanical stimuli
     */

    // Check for stimulus trigger (every 15-30 seconds)
    if (!stimulusActive && (millis() - lastEvent > (unsigned long)random(15000, 30000))) {
        stimulusActive = true;
        stimulusStart = millis();
        lastEvent = millis();
        Serial.println(F("# [STIMULUS] Applied! (light/touch)"));
    }

    // Stimulus response lasts 5-10 seconds
    if (stimulusActive && (millis() - stimulusStart > (unsigned long)random(5000, 10000))) {
        stimulusActive = false;
        Serial.println(F("# [STIMULUS] Response complete, returning to baseline"));
    }

    int pwmValue;

    if (stimulusActive) {
        // DURING STIMULUS: increased activity
        // Higher baseline, more frequent spikes, larger amplitude

        float responsePhase = (millis() - stimulusStart) / 5000.0;  // 0 to ~1
        int excitation = (int)(80.0 * (1.0 - responsePhase));  // Decaying response

        // Elevated baseline with rapid oscillations
        pwmValue = 150 + excitation + (int)(30.0 * sin(phase * 3));

        // More frequent spikes during stimulus
        if (random(0, 10) < 3) {  // 30% chance per sample
            pwmValue += random(50, 100);
        }

        // Add noise
        pwmValue += random(-15, 16);

    } else {
        // BASELINE: normal quiet activity
        float slowPhase = (millis() - startTime) / 60000.0 * TWO_PI;
        pwmValue = 100 + (int)(20.0 * sin(slowPhase));
        pwmValue += random(-10, 11);

        // Occasional baseline spike
        if (millis() - lastSpike > (unsigned long)random(8000, 20000)) {
            pwmValue += random(40, 80);
            lastSpike = millis();
        }
    }

    phase += 0.1;
    if (phase > TWO_PI) phase -= TWO_PI;

    return pwmValue;
}

// ==================== SERIAL COMMANDS ====================

void handleSerialCommands() {
    while (Serial.available()) {
        char cmd = Serial.read();

        switch (cmd) {
            case '0': case '1': case '2': case '3':
            case '4': case '5': case '6': case '7':
            case '8': case '9':
                signalMode = cmd - '0';
                Serial.print(F("# Mode: "));
                Serial.print(signalMode);
                Serial.print(F(" ("));
                printModeName(signalMode);
                Serial.println(F(")"));
                resetState();
                break;

            case 'r': case 'R':
                resetState();
                Serial.println(F("# State Reset"));
                break;

            case 's': case 'S':
                printStatistics();
                break;

            case '?': case 'h': case 'H':
                printHeader();
                break;

            case 't': case 'T':
                // Manual trigger for stimulus mode
                if (signalMode == 9) {
                    stimulusActive = true;
                    stimulusStart = millis();
                    Serial.println(F("# [STIMULUS] Manual trigger!"));
                }
                break;

            case '\n': case '\r':
                break;

            default:
                Serial.print(F("# Unknown command: "));
                Serial.println(cmd);
                break;
        }
    }
}

void resetState() {
    phase = 0;
    randomValue = 128;
    lastSpike = millis();
    lastEvent = millis();
    sampleCount = 0;
    adcSum = 0;
    adcMin = 16383;
    adcMax = 0;
    startTime = millis();
    connectionGood = true;
    lastConnectionChange = millis();
    stimulusActive = false;
}

void printStatistics() {
    Serial.println(F("# --- STATISTICS ---"));
    Serial.print(F("# Mode: "));
    printModeName(signalMode);
    Serial.println();
    Serial.print(F("# Samples: "));
    Serial.println(sampleCount);
    Serial.print(F("# Runtime: "));
    Serial.print((millis() - startTime) / 1000.0, 1);
    Serial.println(F(" sec"));

    #if defined(ARDUINO_UNOR4_MINIMA) || defined(ARDUINO_UNOR4_WIFI)
        Serial.print(F("# ADC Min: "));
        Serial.print(adcMin);
        Serial.print(F(" ("));
        Serial.print((adcMin / 16383.0) * 5000.0, 3);
        Serial.println(F(" mV)"));
        Serial.print(F("# ADC Max: "));
        Serial.print(adcMax);
        Serial.print(F(" ("));
        Serial.print((adcMax / 16383.0) * 5000.0, 3);
        Serial.println(F(" mV)"));
        if (sampleCount > 0) {
            float avg = adcSum / sampleCount;
            Serial.print(F("# ADC Avg: "));
            Serial.print(avg, 1);
            Serial.print(F(" ("));
            Serial.print((avg / 16383.0) * 5000.0, 3);
            Serial.println(F(" mV)"));
        }
    #else
        Serial.print(F("# ADC Min: "));
        Serial.print(adcMin);
        Serial.print(F(" ("));
        Serial.print((adcMin / 1023.0) * 5000.0, 2);
        Serial.println(F(" mV)"));
        Serial.print(F("# ADC Max: "));
        Serial.print(adcMax);
        Serial.print(F(" ("));
        Serial.print((adcMax / 1023.0) * 5000.0, 2);
        Serial.println(F(" mV)"));
    #endif

    Serial.println(F("# -----------------"));
}
