"""Loads the IMU driver named in configs/instrument.json.

The drivers live in imu_drivers/ (one file per sensor, see
imu_drivers/template.py). The capture script only needs make_imu() and
NAME; everything sensor-specific stays in the driver file.
"""

import importlib


def make_imu(imu_cfg):
    """Open the IMU described by the "imu" section of instrument.json."""
    name = imu_cfg.get("driver")
    if not name:
        raise SystemExit('configs/instrument.json: "imu" needs a "driver" '
                         "(a module name in imu_drivers/)")
    module = importlib.import_module(f"imu_drivers.{name}")
    return module.open_imu(imu_cfg)
