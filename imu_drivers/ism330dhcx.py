"""ISM330DHCX (Adafruit STEMMA QT board) over I2C -- the reference build's IMU.

A raw 6-DoF sensor (accelerometer + gyroscope), no on-chip fusion and no
on-chip calibration: what it reports never changes behind our back, and all
of its calibration lives in configs/m_case.json and configs/accel_cal.json.
The driver library objects already have the three attributes the capture
needs (acceleration in m/s^2, gyro in rad/s, temperature in deg C).

Needs: pip install adafruit-blinka adafruit-circuitpython-lsm6ds
Address: 0x6a (0x6b if the ADR jumper is bridged). Only I2C bus 1 (the
header pins 3/5) is supported here.

instrument.json:  "imu": {"driver": "ism330dhcx", "address": "0x6a"}
"""

import time


def open_imu(cfg, retries=5):
    import board
    import busio
    from adafruit_lsm6ds.ism330dhcx import ISM330DHCX

    address = int(str(cfg.get("address", "0x6a")), 0)
    i2c = busio.I2C(board.SCL, board.SDA)
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            sensor = ISM330DHCX(i2c, address=address)
            if attempt > 1:
                print(f"IMU up on attempt {attempt}.")
            return sensor
        except Exception as e:   # OSError / ValueError on a bad bus
            last_error = e
            time.sleep(0.5)
    raise RuntimeError(
        f"ISM330DHCX would not start after {retries} attempts ({last_error}). "
        f"Check the cable (SDA/SCL), the I2C address ({hex(address)}) and that "
        "adafruit-circuitpython-lsm6ds is installed.")
