# Other cameras and IMUs

Nothing in the code is tied to the reference sensors. Your hardware is
described in `configs/` (see [configs/README.md](../configs/README.md)); the
reference build's files are in `examples/reference-build/`.

## IMU

1. Write a driver in `imu_drivers/<your_sensor>.py`, from
   `imu_drivers/template.py`. `open_imu(cfg)` returns an object with
   `.acceleration` (m/s², raw, 3 values), `.gyro` (rad/s) and `.temperature`
   (°C, or `None`). `imu_drivers/ism330dhcx.py` is a complete example.
2. Select it in `configs/instrument.json`:
   `"imu": {"driver": "<your_sensor>", ...}`. The other keys of `imu` (I²C
   address, bus...) go to your driver.
3. Install the sensor's Python library in the capture venv.
4. Calibrate (all three steps of [Calibration](calibration.md)): the
   calibration belongs to one sensor and one mounting.

The calibration tools (`fit_mcase.py`, `calibrate_affine.py`, `rezero.py`)
work for any IMU: they use only the logged gravity vector and the star
solutions.

## Camera

Any camera that picamera2 supports can work. Set in `configs/instrument.json`:

- `sensor_mode`: the index in picamera2's `sensor_modes` list;
- `exposure_us` and `analogue_gain`: enough to see stars of magnitude ~5-6
  (the reference uses 300 ms and gain 12);
- `lens_position`: only for an autofocus camera (0.0 = infinity).

Then build a star database for the camera's horizontal field of view and
point `configs/solver.json` to it:

```bash
python build_database.py --catalog hip_main.dat --max-fov 50.0     # -> data/database.npz
```

The field of view is `2 * atan(sensor width in mm / (2 * focal length in mm))`.
For the reference camera: 4608 px × 1.4 µm = 6.45 mm, focal length 4.74 mm,
so 68.5° (the solver measures 68.2°). `build_database.py` needs the Hipparcos
main catalog file `hip_main.dat` (~51 MB, from CDS:
<https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat>) and cedar-solve. In
`configs/solver.json`, set `database.path` to the new file and
`solve.fov_estimate` to the field of view.
