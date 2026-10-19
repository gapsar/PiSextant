# configs/ -- the files of YOUR instrument

Git ignores every file here except this one: they describe your own hardware
and calibration. The reference build's files are in
`examples/reference-build/`; copy them here only if your hardware and
mounting are the same, and then re-zero (docs/calibration.md).

| File | Content | Made by |
|---|---|---|
| `instrument.json` | camera settings and IMU driver | you (see below) |
| `solver.json` | solver settings and the star database path | you, from `examples/reference-build/solver.json` |
| `m_case.json` | rotation from the IMU frame to the camera frame | `fit_mcase.py` |
| `accel_cal.json` | accelerometer model: `A`, `b`, temperature term `c` | `calibrate_affine.py`, then `rezero.py` |

## instrument.json

```json
{
  "camera": {
    "name": "free text",
    "sensor_mode": 1,
    "exposure_us": 300000,
    "analogue_gain": 12.0,
    "lens_position": 0.0
  },
  "imu": {
    "driver": "ism330dhcx",
    "address": "0x6a"
  }
}
```

- `sensor_mode`: index in picamera2's `sensor_modes` list (print it with
  `python -c "from picamera2 import Picamera2; print(Picamera2().sensor_modes)"`).
  A binned mode gives brighter stars and smaller images.
- `exposure_us`, `analogue_gain`: long enough to see stars of magnitude ~5-6,
  short enough to keep the trails small at sea (the reference uses 300 ms).
- `lens_position`: only for autofocus cameras (dioptres, 0.0 = infinity);
  leave it out for a fixed-focus lens.
- `imu.driver`: a module name in `imu_drivers/`; the other keys of `imu` go to
  that driver (here the I2C address).

## solver.json

The `database.path` (relative to the repo folder) must point to a database
built for your camera's field of view (`build_database.py --max-fov ...`),
and `solve.fov_estimate` must be close to that field of view.
