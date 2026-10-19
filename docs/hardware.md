# Hardware (reference build)

This is the hardware of the reference instrument. Other cameras and IMUs can
work: see [Other cameras and IMUs](other-sensors.md).

<!-- PHOTO: the parts, the wiring, the inside of the enclosure -->

| Part | Reference used | Note |
|---|---|---|
| Computer | Raspberry Pi Zero 2 W | 512 MB RAM; any Pi with a CSI camera port should work |
| Camera | Raspberry Pi Camera Module 3, standard lens (Sony IMX708), not the Wide version | focus at infinity; ~68° horizontal field of view in the 2304×1296 mode |
| IMU | Adafruit ISM330DHCX (STEMMA QT) | accelerometer + gyroscope, I²C |
| Clock | DS3231 RTC module with a coin cell | keeps the time without network |
| Enclosure | 3D printed (files in `hardware/`, coming) | see the mounting rules below |
| Power | 5 V USB supply | the Pi warms the enclosure: see [Known limits](limits.md) |

<!-- TODO: 3D printable files in hardware/ -->

## Wiring

The IMU and the RTC share the I²C bus 1 of the Pi:

| Signal | Pi header pin | ISM330DHCX | DS3231 |
|---|---|---|---|
| 3.3 V | 1 | VIN | VCC |
| SDA (GPIO 2) | 3 | SDA | SDA |
| SCL (GPIO 3) | 5 | SCL | SCL |
| GND | 6 | GND | GND |

Expected I²C addresses: ISM330DHCX at `0x6a` (`0x6b` if its ADR jumper is
bridged), DS3231 at `0x68` (and its EEPROM at `0x57`). Check with
`i2cdetect -y 1` (package `i2c-tools`).

The camera connects to the CSI port (the Pi Zero needs the narrow 22-pin
cable).

## Mounting rules

These rules come from real failures. They matter more than any software
setting.

1. **The camera and the IMU must not move relative to each other.** A rotation
   of 0.1° between them gives about 6 nmi of error. Screw both boards to the
   same rigid part. A knock or a drop can shift them; re-zero after any shock.
2. **Fix the enclosure rigidly to the ship**, as low and as close to the
   centre of the ship as the sky view permits. High positions (a mast) add
   more wave acceleration.
3. **Give the camera a clear sky.** The camera must not see the ship's lights,
   masts or funnel smoke. The reference build tilts the camera about 23° from
   the vertical.
4. **Keep the IMU away from heat sources** if you can. Its offset changes with
   temperature (about 1 nmi per °C after the temperature calibration).
