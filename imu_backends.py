#!/usr/bin/env python3
"""IMU driver for the capture scripts: Adafruit ISM330DHCX over STEMMA QT / I2C.

A raw 6-DoF sensor (accelerometer + gyroscope), no on-chip fusion and no
on-chip calibration: what it reports never changes behind our back, and all of
its calibration lives in the external files (M_CASE + the affine model).

  - "gravity" is the raw acceleration: equal to gravity while the rig is
    still; the fix path medians over the exposure (and averages fixes over
    60 s), so linear/wave acceleration averages out.
  - gyro is in rad/s. Its zero-rate offset is NOT constant (it drifts with
    temperature and warm-up, ~0.1-0.4 deg/s), which is why the gyro-based
    wave compensation is off by default.
  - temperature is a real die temperature in degrees C.

Driver on the Pi: pip install adafruit-circuitpython-lsm6ds
Wiring check: the sensor answers on I2C bus 1 at 0x6a (0x6b if the ADR jumper
is bridged), WHO_AM_I register 0x0f reads 0x6b.
"""

import time

import board
import busio

NAME = "ism330dhcx"


def make_imu(retries=5):
    """Connect to the ISM330DHCX and return the driver object."""
    from adafruit_lsm6ds.ism330dhcx import ISM330DHCX
    i2c = busio.I2C(board.SCL, board.SDA)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            sensor = ISM330DHCX(i2c)
            if attempt > 1:
                print(f"IMU up on attempt {attempt}.")
            return sensor
        except Exception as e:   # OSError / ValueError on a bad bus
            last_error = e
            print(f"IMU init failed ({e}); retrying, attempt {attempt}/{retries} ...")
            time.sleep(1.0)
    raise RuntimeError(
        f"ISM330DHCX would not start after {retries} attempts ({last_error}). "
        "Check the STEMMA QT cable at both ends (SDA/SCL), the I2C address "
        "(0x6a, or 0x6b with the ADR jumper bridged), and that "
        "adafruit-circuitpython-lsm6ds is installed.")
