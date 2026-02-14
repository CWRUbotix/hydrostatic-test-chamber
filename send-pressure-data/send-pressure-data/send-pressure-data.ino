// send-pressure-data.ino
// Streams Bar30 (MS5837-30BA) pressure data as ASCII lines over Serial at 115200 baud.
// Output: t_ms,psi,temp_f

#include <Wire.h>
#include "MS5837.h"

MS5837 sensor;

static const uint32_t SERIAL_BAUD = 115200;

// Fixed sample period (microseconds).
// Set this >= the sensor/library's internal conversion time.
// If you see repeat values or sluggish rate, increase this (e.g., 20000 for 50 Hz).
static const uint32_t SAMPLE_PERIOD_US = 10000; // 10 ms target (100 Hz)

// Unit conversion
static const float MBAR_TO_PSI = 0.0145037738f; // 1 mbar = 0.0145037738 psi

static uint32_t next_sample_us = 0;

static float c_to_f(float c) {
  return (c * 9.0f / 5.0f) + 32.0f;
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial) { }

  Wire.begin();
  Wire.setClock(400000); // attempt 400 kHz I2C

  if (!sensor.init()) {
    Serial.println("ERR,MS5837_INIT_FAIL");
    while (true) { delay(1000); }
  }

  // Bar30 is the 30 bar model
  sensor.setModel(MS5837::MS5837_30BA);

  // Density only affects depth calculations; set anyway
  sensor.setFluidDensity(997); // kg/m^3 fresh water

  next_sample_us = micros();

  // Optional header (Python ignores it)
  Serial.println("t_ms,psi,temp_f");
}

void loop() {
  const uint32_t now_us = micros();

  // micros() wrap-safe timing check
  if ((int32_t)(now_us - next_sample_us) >= 0) {
    next_sample_us += SAMPLE_PERIOD_US;

    // BlueRobotics library: read() has no arguments in your installed version
    sensor.read();

    // pressure() returns mbar, temperature() returns C
    const float p_mbar = sensor.pressure();
    const float t_c = sensor.temperature();

    const float p_psi = p_mbar * MBAR_TO_PSI;
    const float t_f = c_to_f(t_c);

    const uint32_t t_ms = millis();

    // ASCII CSV line
    Serial.print(t_ms);
    Serial.print(",");
    Serial.print(p_psi, 3);
    Serial.print(",");
    Serial.println(t_f, 2);
  }
}
