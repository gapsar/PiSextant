# Other cameras and IMUs

Current state:

- **Calibration is generic.** `fit_mcase.py`, `calibrate_affine.py` and
  `rezero.py` work for any camera and any accelerometer: they only use the
  logged gravity vector and the star solutions.
- **The IMU driver is one small file**, `imu_backends.py`: `make_imu()` must
  return an object with `.acceleration` (m/s², 3 values), `.gyro` (rad/s) and
  `.temperature` (°C). Another I²C IMU needs a new version of this function and
  a new calibration. The calibration file name `r_cal_ism330dhcx.json` is
  still fixed in the capture script.
- **The camera settings are fixed** in `capture_run_wave.py` (sensor mode,
  300 ms exposure, gain 12, focus at infinity) for the IMX708. Another camera
  needs these settings, a star database for its field of view
  (`build_database.py`, needs the Hipparcos catalog file `hip_main.dat`), and
  the field of view in `configs/solver.json` (`fov_estimate`).

A cleaner way to select the sensors is planned.
