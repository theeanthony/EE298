/*
 * fungal_signal_acquisition.ino
 * EE297B Research Project - Signal Processing Fungi Propagation
 * Anthony Contreras & Alex Wong | San Jose State University
 *
 * This firmware handles:
 * - Signal simulation via PWM + voltage divider (for development)
 * - ADC acquisition at 10 Hz (14-bit on Uno R4, 10-bit on R3)
 * - Serial output to Python pipeline
 * - Actuator control (mister, fan, LED) via serial commands
 *
 * Hardware connections:
 * - D9: PWM output for signal simulator
 * - A0: ADC input from INA128 amplifier (or simulator)
 * - D6: LED actuator
 * - D7: Mister relay
 * - D8: Fan relay
 *
 * Output format: timestamp_ms,adc_raw,voltage_mV,mister,fan,led
 *
 * PAIR_ID: Set at compile time to identify which electrode pair is wired.
 * Recompile with a different PAIR_ID when manually rotating pairs.
 */

// ============== PAIR IDENTIFICATION ==============
// Change this when rotating to a different electrode pair.
// Printed in the startup banner and CSV header comment.
const int PAIR_ID = 2;

// ============== PIN DEFINITIONS ==============
const int PWM_PIN = 9;        // Simulator PWM output
const int ADC_PIN = A0;       // Signal input from amplifier
const int LED_PIN = 6;        // LED actuator (PWM capable)
const int MISTER_PIN = 7;     // Mister relay control
const int FAN_PIN = 8;        // Fan relay control

// ============== TIMING CONFIGURATION ==============
const int SAMPLE_PERIOD_MS = 100;   // 10 Hz sampling rate
const int SIMULATOR_STEP_MS = 50;   // Simulator update rate

// ============== ADC CONFIGURATION ==============
// Set in setup() based on detected board.
int ADC_MAX_VALUE = 1023;     // 10-bit default; overridden to 16383 on Uno R4
float ADC_VREF_MV = 5000.0;  // 5 V reference

// ============== STATE VARIABLES ==============
unsigned long lastSampleTime = 0;
unsigned long lastSimulatorTime = 0;

// Simulator state (generates slow-varying fake fungal signal)
int pwmValue = 0;
int pwmDirection = 1;
int pwmStepSize = 1;
bool simulatorEnabled = true;

// Actuator states
bool misterState = false;
bool fanState = false;
int ledBrightness = 0;

// ============== SETUP ==============
void setup() {
    // Initialize serial communication
    Serial.begin(115200);
    while (!Serial) {
        ; // Wait for serial port to connect (needed for native USB)
    }

    // Configure pins
    pinMode(PWM_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);
    pinMode(MISTER_PIN, OUTPUT);
    pinMode(FAN_PIN, OUTPUT);

    // Set initial actuator states (all OFF)
    digitalWrite(MISTER_PIN, LOW);
    digitalWrite(FAN_PIN, LOW);
    analogWrite(LED_PIN, 0);

    // Configure ADC
    #if defined(ARDUINO_UNOR4_MINIMA) || defined(ARDUINO_UNOR4_WIFI)
        analogReference(AR_DEFAULT);   // Uno R4: use AR_DEFAULT (not DEFAULT)
        analogReadResolution(14);
        ADC_MAX_VALUE = 16383;
        ADC_VREF_MV = 5000.0;
    #else
        analogReference(DEFAULT);      // Uno R3 / AVR boards
        ADC_MAX_VALUE = 1023;
        ADC_VREF_MV = 5000.0;
    #endif

    // Print header
    Serial.println("# ==========================================");
    Serial.println("# Fungal Signal Acquisition System v2.0");
    Serial.println("# EE297B Research Project - SJSU");
    Serial.println("# ==========================================");
    Serial.print("# Pair ID: ");
    Serial.println(PAIR_ID);
    #if defined(ARDUINO_UNOR4_MINIMA) || defined(ARDUINO_UNOR4_WIFI)
        Serial.println("# ADC: 14-bit (Uno R4) — 0.006 mV/count at 51x gain");
    #else
        Serial.println("# ADC: 10-bit (Uno R3 or compatible)");
    #endif
    Serial.println("# Commands: M/m=mister, F/f=fan, L/l=LED, S/s=simulator");
    Serial.println("# Format: timestamp_ms,adc_raw,voltage_mV,mister,fan,led");
    Serial.println("# ==========================================");
}

// ============== MAIN LOOP ==============
void loop() {
    unsigned long currentTime = millis();

    // Check for incoming commands
    handleSerialCommands();

    // Update simulator (if enabled)
    if (simulatorEnabled && (currentTime - lastSimulatorTime >= SIMULATOR_STEP_MS)) {
        lastSimulatorTime = currentTime;
        updateSimulator();
    }

    // Sample and transmit at fixed rate
    if (currentTime - lastSampleTime >= SAMPLE_PERIOD_MS) {
        lastSampleTime = currentTime;
        sampleAndTransmit(currentTime);
    }
}

// ============== SIMULATOR ==============
void updateSimulator() {
    // Generate slow ramp up/down to simulate fungal signal variations
    // Full cycle: 0 -> 255 -> 0 takes about 25 seconds at default settings

    pwmValue += pwmDirection * pwmStepSize;

    if (pwmValue >= 255) {
        pwmValue = 255;
        pwmDirection = -1;
    } else if (pwmValue <= 0) {
        pwmValue = 0;
        pwmDirection = 1;
    }

    analogWrite(PWM_PIN, pwmValue);
}

// ============== SAMPLING ==============
void sampleAndTransmit(unsigned long timestamp) {
    // Take multiple readings and average for noise reduction
    const int NUM_SAMPLES = 4;
    long adcSum = 0;

    for (int i = 0; i < NUM_SAMPLES; i++) {
        adcSum += analogRead(ADC_PIN);
        delayMicroseconds(100);  // Small delay between readings
    }

    int adcAvg = adcSum / NUM_SAMPLES;

    // Convert to millivolts using detected ADC range
    float voltage_mV = (adcAvg / (float)ADC_MAX_VALUE) * ADC_VREF_MV;

    // Transmit in CSV format: timestamp_ms,adc_raw,voltage_mV,mister,fan,led
    Serial.print(timestamp);
    Serial.print(",");
    Serial.print(adcAvg);
    Serial.print(",");
    Serial.print(voltage_mV, 3);
    Serial.print(",");
    Serial.print(misterState ? 1 : 0);
    Serial.print(",");
    Serial.print(fanState ? 1 : 0);
    Serial.print(",");
    Serial.println(ledBrightness);
}

// ============== COMMAND HANDLING ==============
void handleSerialCommands() {
    while (Serial.available() > 0) {
        char cmd = Serial.read();

        switch (cmd) {
            // Mister control
            case 'M':
                digitalWrite(MISTER_PIN, HIGH);
                misterState = true;
                Serial.println("# MISTER: ON");
                break;
            case 'm':
                digitalWrite(MISTER_PIN, LOW);
                misterState = false;
                Serial.println("# MISTER: OFF");
                break;

            // Fan control
            case 'F':
                digitalWrite(FAN_PIN, HIGH);
                fanState = true;
                Serial.println("# FAN: ON");
                break;
            case 'f':
                digitalWrite(FAN_PIN, LOW);
                fanState = false;
                Serial.println("# FAN: OFF");
                break;

            // LED control (PWM)
            case 'L':
                ledBrightness = 255;
                analogWrite(LED_PIN, ledBrightness);
                Serial.println("# LED: ON (100%)");
                break;
            case 'l':
                ledBrightness = 0;
                analogWrite(LED_PIN, ledBrightness);
                Serial.println("# LED: OFF");
                break;
            case '1': case '2': case '3': case '4': case '5':
            case '6': case '7': case '8': case '9':
                // Set LED to percentage (1-9 = 10%-90%)
                ledBrightness = map((cmd - '0'), 1, 9, 25, 230);
                analogWrite(LED_PIN, ledBrightness);
                Serial.print("# LED: ");
                Serial.print((cmd - '0') * 10);
                Serial.println("%");
                break;

            // Simulator control
            case 'S':
                simulatorEnabled = true;
                Serial.println("# SIMULATOR: ON");
                break;
            case 's':
                simulatorEnabled = false;
                analogWrite(PWM_PIN, 0);
                Serial.println("# SIMULATOR: OFF");
                break;

            // Status query
            case '?':
                printStatus();
                break;

            // Newline/carriage return - ignore
            case '\n':
            case '\r':
                break;

            default:
                Serial.print("# Unknown command: ");
                Serial.println(cmd);
                break;
        }
    }
}

void printStatus() {
    Serial.println("# --- STATUS ---");
    Serial.print("# Simulator: ");
    Serial.println(simulatorEnabled ? "ON" : "OFF");
    Serial.print("# Mister: ");
    Serial.println(misterState ? "ON" : "OFF");
    Serial.print("# Fan: ");
    Serial.println(fanState ? "ON" : "OFF");
    Serial.print("# LED: ");
    Serial.print(map(ledBrightness, 0, 255, 0, 100));
    Serial.println("%");
    Serial.println("# --------------");
}
