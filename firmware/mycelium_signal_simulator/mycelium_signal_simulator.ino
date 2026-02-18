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
 * ARDUINO UNO R4 ADVANTAGES:
 * - 14-bit ADC (vs 10-bit on Uno R3) = 0.20mV resolution at 3.3V ref
 * - 3.3V output available for lower voltage divider input
 * - DAC output on A0 (if needed for analog signal generation)
 */

// ============== PIN DEFINITIONS ==============
const int PWM_PIN = 9;      // Output pin (to voltage divider)
const int ADC_PIN = A0;     // Input pin (from voltage divider)
const int LED_PIN = 13;     // Built-in LED for activity indicator

// ============== TIMING ==============
const int SAMPLE_RATE_MS = 100;  // 10 Hz sampling (matches real fungal signal bandwidth)
unsigned long lastSample = 0;
unsigned long startTime = 0;

// ============== SIGNAL GENERATION ==============
float phase = 0;
int signalMode = 3;  // Start with Composite mode
int randomValue = 128;
unsigned long lastSpike = 0;
float driftValue = 0;
int intermittentCounter = 0;
unsigned long stimulusTime = 0;

// ============== REALISTIC MYCELIUM PARAMETERS ==============
// Based on Buffi et al. 2025, Adamatzky et al.
// Real fungal signals: 0.01-1 Hz, 0.5-2.1 mV amplitude
float myceliumPhase1 = 0;      // Very slow oscillation (0.01 Hz)
float myceliumPhase2 = 0;      // Slow oscillation (0.05 Hz)
float myceliumPhase3 = 0;      // Medium oscillation (0.2 Hz)
int myceliumBaseline = 128;    // Wandering baseline
unsigned long lastActionPotential = 0;
int actionPotentialPhase = 0;  // 0 = none, >0 = in progress
float nutrientLevel = 1.0;     // Simulated nutrient availability

// ============== STATISTICS ==============
long sampleCount = 0;
float adcSum = 0;
int adcMin = 1023;
int adcMax = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial) {
        ; // Wait for serial port (needed for native USB boards)
    }

    pinMode(PWM_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);

    // ============== ADC CONFIGURATION ==============
    #if defined(ARDUINO_UNOR4_MINIMA) || defined(ARDUINO_UNOR4_WIFI)
        analogReadResolution(14);  // Uno R4: Use 14-bit ADC
        Serial.println(F("# Detected: Arduino Uno R4 (14-bit ADC)"));
    #else
        Serial.println(F("# Detected: Arduino Uno R3 or compatible (10-bit ADC)"));
    #endif

    startTime = millis();
    randomSeed(analogRead(A1));  // Seed random number generator

    printHeader();
}

void printHeader() {
    Serial.println(F("# ================================================"));
    Serial.println(F("# Mycelium Signal Simulator v2.0"));
    Serial.println(F("# EE297B Research Project - SJSU"));
    Serial.println(F("# ================================================"));
    Serial.println(F("#"));
    Serial.println(F("# SIGNAL MODES (0-9, m):"));
    Serial.println(F("#   0 = Sine          - Clean sine wave"));
    Serial.println(F("#   1 = Random Walk   - Biological noise"));
    Serial.println(F("#   2 = Spikes        - Action potentials"));
    Serial.println(F("#   3 = Composite     - Realistic (DEFAULT)"));
    Serial.println(F("#   4 = NOTHING       - Flat baseline"));
    Serial.println(F("#   5 = NOISE         - Pure random noise"));
    Serial.println(F("#   6 = DRIFT         - Slowly drifting signal"));
    Serial.println(F("#   7 = SATURATION    - Signal hits limits"));
    Serial.println(F("#   8 = INTERMITTENT  - On/off pattern"));
    Serial.println(F("#   9 = STIMULUS      - Response to stimulus"));
    Serial.println(F("#   m = MYCELIUM      - Realistic fungal signal!"));
    Serial.println(F("#"));
    Serial.println(F("# COMMANDS:"));
    Serial.println(F("#   0-9 = Change signal mode"));
    Serial.println(F("#   r   = Reset phase/state"));
    Serial.println(F("#   s   = Print statistics"));
    Serial.println(F("#"));
    Serial.println(F("# OUTPUT FORMAT: timestamp_ms,pwm,adc_raw,voltage_mV"));
    Serial.println(F("# ================================================"));
    printCurrentMode();
    Serial.println(F("# ================================================"));
}

void printCurrentMode() {
    Serial.print(F("# Mode: "));
    Serial.print(signalMode);
    Serial.print(F(" ("));
    switch (signalMode) {
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
        case 10: Serial.print(F("MYCELIUM")); break;
    }
    Serial.println(F(")"));
}

void loop() {
    // Check for serial commands
    handleSerialCommands();

    unsigned long now = millis();

    // Sample at fixed rate
    if (now - lastSample >= SAMPLE_RATE_MS) {
        lastSample = now;
        sampleCount++;

        // Generate signal based on current mode
        int pwmValue = generateSignal();

        // Output to voltage divider
        analogWrite(PWM_PIN, pwmValue);

        // Blink LED to show activity (blink faster for higher values)
        digitalWrite(LED_PIN, (pwmValue > 127) ? HIGH : LOW);

        // Read back through ADC (multiple samples for averaging)
        int adcRaw = readADCAverage(4);

        // Update statistics
        adcSum += adcRaw;
        if (adcRaw < adcMin) adcMin = adcRaw;
        if (adcRaw > adcMax) adcMax = adcRaw;

        // Convert to millivolts
        #if defined(ARDUINO_UNOR4_MINIMA) || defined(ARDUINO_UNOR4_WIFI)
            float adcMillivolts = (adcRaw / 16383.0) * 5000.0;  // 14-bit, 5V ref
        #else
            float adcMillivolts = (adcRaw / 1023.0) * 5000.0;   // 10-bit, 5V ref
        #endif

        // Output CSV data
        Serial.print(now - startTime);  // Relative timestamp
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
        case 0:  // Sine wave
            pwmValue = generateSine();
            break;

        case 1:  // Random walk
            pwmValue = generateRandomWalk();
            break;

        case 2:  // Spikes
            pwmValue = generateSpikes();
            break;

        case 3:  // Composite (most realistic)
            pwmValue = generateComposite();
            break;

        case 4:  // NOTHING - flat baseline
            pwmValue = generateNothing();
            break;

        case 5:  // NOISE - pure random
            pwmValue = generateNoise();
            break;

        case 6:  // DRIFT - slowly changing
            pwmValue = generateDrift();
            break;

        case 7:  // SATURATION - hits limits
            pwmValue = generateSaturation();
            break;

        case 8:  // INTERMITTENT - on/off
            pwmValue = generateIntermittent();
            break;

        case 9:  // STIMULUS - response pattern
            pwmValue = generateStimulus();
            break;

        case 10:  // MYCELIUM - realistic fungal bioelectrical signal
            pwmValue = generateMycelium();
            break;

        default:
            pwmValue = 128;
            break;
    }

    return constrain(pwmValue, 0, 255);
}

int generateSine() {
    // Slow sine wave: ~0.05 Hz (20 second period)
    phase += 0.0314;
    if (phase > TWO_PI) phase -= TWO_PI;

    // Center at 128, amplitude of 100 (gives PWM 28-228)
    return 128 + (int)(100.0 * sin(phase));
}

int generateRandomWalk() {
    // Random walk with mean reversion (mimics biological noise)
    randomValue += random(-5, 6);

    // Mean reversion: pull slightly toward center (128)
    if (randomValue > 128) randomValue -= 1;
    if (randomValue < 128) randomValue += 1;

    // Constrain to valid range
    randomValue = constrain(randomValue, 20, 235);

    return randomValue;
}

int generateSpikes() {
    // Low baseline with occasional spikes
    int baseline = 30;  // Low baseline (~12% PWM)
    int pwmValue = baseline;

    // Random spike every 3-10 seconds
    if (millis() - lastSpike > (unsigned long)random(3000, 10000)) {
        // Spike!
        pwmValue = random(180, 250);  // High spike (70-98% PWM)
        lastSpike = millis();
    }

    return pwmValue;
}

int generateComposite() {
    // Most realistic: combines all three modes
    // Slow underlying oscillation (very slow: 60 sec period)
    float slowPhase = (millis() - startTime) / 60000.0 * TWO_PI;
    int base = 128 + (int)(40.0 * sin(slowPhase));

    // Add random noise (±20)
    int noise = random(-20, 21);

    // Occasional spike
    int spike = 0;
    if (millis() - lastSpike > (unsigned long)random(5000, 15000)) {
        spike = random(80, 120);  // Spike adds 80-120 to current value
        lastSpike = millis();
        Serial.println(F("# [SPIKE EVENT]"));
    }

    return constrain(base + noise + spike, 0, 255);
}

int generateNothing() {
    // Flat baseline - mimics dead/no signal
    // Just slight noise to show it's "alive"
    return 50 + random(-3, 4);  // Very flat around 50
}

int generateNoise() {
    // Pure random noise - like electrical interference
    return random(0, 256);  // Full random range
}

int generateDrift() {
    // Slowly drifting signal - mimics electrode drift or temperature changes
    driftValue += 0.3;  // Slow upward drift
    
    // Reset when it gets too high
    if (driftValue > 200) {
        driftValue = 30;
    }
    
    // Add small noise
    int noise = random(-10, 11);
    
    return constrain((int)driftValue + noise, 0, 255);
}

int generateSaturation() {
    // Signal that frequently hits the limits (clipping)
    // Oscillates but clips at top and bottom
    float fastPhase = (millis() - startTime) / 3000.0 * TWO_PI;
    int base = 128 + (int)(150.0 * sin(fastPhase));  // Large amplitude - will clip!
    
    return constrain(base, 0, 255);  // This will saturate/clip
}

int generateIntermittent() {
    // Signal that cuts in and out - mimics loose connection
    intermittentCounter++;
    
    // Pattern: 30 samples ON, 20 samples OFF (3 sec on, 2 sec off)
    if ((intermittentCounter % 50) < 30) {
        // ON - generate normal composite signal
        int base = 128 + random(-30, 31);
        Serial.println(F("# [INTERMITTENT: ON]"));
        return base;
    } else {
        // OFF - near zero
        if (intermittentCounter % 50 == 30) {
            Serial.println(F("# [INTERMITTENT: OFF]"));
        }
        return random(0, 10);
    }
}

int generateStimulus() {
    // Simulates response to external stimulus
    // Quiet baseline, then big response when "stimulated"

    unsigned long timeSinceStimulus = millis() - stimulusTime;

    // Auto-trigger stimulus every 15 seconds
    if (timeSinceStimulus > 15000) {
        stimulusTime = millis();
        timeSinceStimulus = 0;
        Serial.println(F("# [STIMULUS APPLIED!]"));
    }

    // Response curve: rapid rise, slow decay
    if (timeSinceStimulus < 5000) {
        // First 5 seconds: active response
        float response = exp(-timeSinceStimulus / 2000.0);  // Exponential decay
        int magnitude = (int)(150.0 * response);
        return 80 + magnitude + random(-10, 11);
    } else {
        // After 5 sec: back to baseline
        return 80 + random(-10, 11);
    }
}

int generateMycelium() {
    /*
     * REALISTIC MYCELIUM BIOELECTRICAL SIGNAL SIMULATION
     * Based on research from:
     * - Buffi et al. 2025: STFT analysis of fungal signals
     * - Adamatzky et al.: Fungal action potentials
     * - Olsson & Hansson: Mycelial electrical oscillations
     *
     * Real characteristics:
     * - Frequency range: 0.01-1 Hz (very slow!)
     * - Amplitude: 0.5-2.1 mV
     * - Action potentials: 1-5 min duration, irregular intervals
     * - Multiple overlapping oscillations
     * - Baseline drift over hours
     * - Response to nutrients, light, touch
     */

    int pwmValue = 0;

    // ===== 1. MULTIPLE SLOW OSCILLATIONS =====
    // These represent different biological rhythms in the mycelium

    // Very slow circadian-like rhythm (0.01 Hz ~ 100 sec period)
    myceliumPhase1 += 0.000628;  // 2*PI / (100 sec * 10 Hz)
    if (myceliumPhase1 > TWO_PI) myceliumPhase1 -= TWO_PI;
    float wave1 = 30.0 * sin(myceliumPhase1);

    // Slow metabolic rhythm (0.05 Hz ~ 20 sec period)
    myceliumPhase2 += 0.00314;   // 2*PI / (20 sec * 10 Hz)
    if (myceliumPhase2 > TWO_PI) myceliumPhase2 -= TWO_PI;
    float wave2 = 20.0 * sin(myceliumPhase2);

    // Faster signaling rhythm (0.2 Hz ~ 5 sec period)
    myceliumPhase3 += 0.01257;   // 2*PI / (5 sec * 10 Hz)
    if (myceliumPhase3 > TWO_PI) myceliumPhase3 -= TWO_PI;
    float wave3 = 15.0 * sin(myceliumPhase3);

    // ===== 2. WANDERING BASELINE =====
    // Mycelium baseline slowly drifts (electrode polarization, growth, etc.)
    if (random(100) < 3) {  // 3% chance each sample
        myceliumBaseline += random(-2, 3);  // Small drift
        myceliumBaseline = constrain(myceliumBaseline, 100, 156);
    }

    // ===== 3. ACTION POTENTIALS (SPIKES) =====
    // Real mycelium shows irregular "spiking trains"
    // Duration: 1-5 minutes, Interval: 10-30 minutes (scaled down for demo)

    int actionPotential = 0;

    if (actionPotentialPhase > 0) {
        // Currently in an action potential event
        actionPotentialPhase--;

        // Action potential shape: fast rise, slow decay
        if (actionPotentialPhase > 40) {
            // Rising phase (first ~4 seconds)
            actionPotential = map(actionPotentialPhase, 50, 40, 0, 80);
        } else if (actionPotentialPhase > 10) {
            // Plateau phase (middle ~3 seconds)
            actionPotential = 80 + random(-5, 6);
        } else {
            // Decay phase (last ~1 second)
            actionPotential = map(actionPotentialPhase, 10, 0, 80, 0);
        }

        if (actionPotentialPhase == 0) {
            Serial.println(F("# [MYCELIUM] Action potential ended"));
        }
    } else {
        // Check for new action potential (random, ~every 20-60 seconds)
        if (millis() - lastActionPotential > (unsigned long)random(20000, 60000)) {
            actionPotentialPhase = 50;  // ~5 second duration
            lastActionPotential = millis();
            Serial.println(F("# [MYCELIUM] Action potential started!"));
        }
    }

    // ===== 4. BIOLOGICAL NOISE =====
    // Low-amplitude, slightly correlated noise
    static int bioNoise = 0;
    bioNoise += random(-3, 4);  // Random walk
    bioNoise = bioNoise * 0.9;   // Decay toward zero
    bioNoise = constrain(bioNoise, -15, 15);

    // ===== 5. NUTRIENT RESPONSE =====
    // Simulate periodic "feeding" events that increase activity
    static unsigned long lastNutrientTime = 0;
    int nutrientBoost = 0;

    if (millis() - lastNutrientTime > 45000) {  // Every 45 seconds
        lastNutrientTime = millis();
        nutrientLevel = 1.5;  // Boost activity
        Serial.println(F("# [MYCELIUM] Nutrient pulse detected"));
    }

    // Nutrient effect decays over time
    if (nutrientLevel > 1.0) {
        nutrientLevel -= 0.005;  // Slow decay
        nutrientBoost = (int)((nutrientLevel - 1.0) * 40.0);
    }

    // ===== COMBINE ALL COMPONENTS =====
    pwmValue = myceliumBaseline
             + (int)(wave1 * nutrientLevel)
             + (int)(wave2 * nutrientLevel)
             + (int)wave3
             + actionPotential
             + bioNoise
             + nutrientBoost;

    return constrain(pwmValue, 0, 255);
}

void handleSerialCommands() {
    while (Serial.available()) {
        char cmd = Serial.read();

        // Check if it's a digit 0-9
        if (cmd >= '0' && cmd <= '9') {
            signalMode = cmd - '0';  // Convert char to int
            resetState();  // Reset state when changing modes
            printCurrentMode();
        }
        else if (cmd == 'm' || cmd == 'M') {
            signalMode = 10;  // MYCELIUM mode
            resetState();
            printCurrentMode();
            Serial.println(F("# [MYCELIUM] Realistic fungal signal simulation"));
            Serial.println(F("# Features: slow oscillations, action potentials, nutrient response"));
        }
        else if (cmd == 'r' || cmd == 'R') {
            resetState();
            Serial.println(F("# State Reset"));
        }
        else if (cmd == 's' || cmd == 'S') {
            printStatistics();
        }
        else if (cmd == '?' || cmd == 'h' || cmd == 'H') {
            printHeader();
        }
        else if (cmd == '\n' || cmd == '\r') {
            // Ignore newlines
        }
        else {
            // Unknown command - just ignore
        }
    }
}

void resetState() {
    phase = 0;
    randomValue = 128;
    lastSpike = millis();
    driftValue = 30;
    intermittentCounter = 0;
    stimulusTime = millis();
    startTime = millis();  // Reset timer too

    // Reset mycelium state
    myceliumPhase1 = 0;
    myceliumPhase2 = 0;
    myceliumPhase3 = 0;
    myceliumBaseline = 128;
    lastActionPotential = millis();
    actionPotentialPhase = 0;
    nutrientLevel = 1.0;
}

void printStatistics() {
    Serial.println(F("# --- STATISTICS ---"));
    Serial.print(F("# Samples: "));
    Serial.println(sampleCount);
    Serial.print(F("# Runtime: "));
    Serial.print((millis() - startTime) / 1000.0, 1);
    Serial.println(F(" sec"));
    Serial.print(F("# ADC Min: "));
    Serial.print(adcMin);
    Serial.print(F(" ("));
    Serial.print((adcMin / 1023.0) * 5000.0, 2);
    Serial.println(F(" mV)"));
    Serial.print(F("# ADC Max: "));
    Serial.print(adcMax);
    Serial.print(F(" ("));
    #if defined(ARDUINO_UNOR4_MINIMA) || defined(ARDUINO_UNOR4_WIFI)
        Serial.print((adcMax / 16383.0) * 5000.0, 3);
    #else
        Serial.print((adcMax / 1023.0) * 5000.0, 2);
    #endif
    Serial.println(F(" mV)"));
    Serial.print(F("# ADC Avg: "));
    float avg = adcSum / sampleCount;
    Serial.print(avg, 1);
    Serial.print(F(" ("));
    #if defined(ARDUINO_UNOR4_MINIMA) || defined(ARDUINO_UNOR4_WIFI)
        Serial.print((avg / 16383.0) * 5000.0, 3);
    #else
        Serial.print((avg / 1023.0) * 5000.0, 2);
    #endif
    Serial.println(F(" mV)"));
    printCurrentMode();
    Serial.println(F("# -----------------"));
}