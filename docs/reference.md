# Files and folders

## Scripts

| Script | Runs on | Role |
|---|---|---|
| `capture_run_wave.py` | Pi | camera + IMU + fixes, writes the run folder |
| `solve_worker.py` | Pi | star detection and solving, serves the capture over a Unix socket |
| `imu_backends.py` | Pi | loads the IMU driver named in `configs/instrument.json` |
| `imu_drivers/` | Pi | one driver per IMU (`ism330dhcx.py`, `template.py`) |
| `webui/server.py` | Pi | web interface and process control |
| `set_clock_from_gps.py` | Pi | sets the system clock and the RTC from a serial GPS |
| `fit_mcase.py` | laptop | calibration step 1 (mounting rotation) |
| `calibrate_affine.py` | laptop | calibration step 2 (accelerometer model) |
| `rezero.py` | laptop | calibration step 3 (re-zero) |
| `calibrate_rcal.py` | laptop | shared calibration helpers (zenith model, rotation fit) |
| `multiheading_fix.py` | laptop | multi-heading correction |
| `sea_eval.py` | laptop | fix error of a run against `gps.jsonl` |
| `drift_check.py` | laptop | drift with temperature and time |
| `replay_solve.py` | laptop | re-solves the saved images of a run with another solver config |
| `build_database.py` | laptop | builds a star pattern database for another field of view |
| `nmea_logger.py` | laptop | logs a serial GPS as `gps.jsonl` |

## A run folder

```
runs/<YYYY-MM-DD_HH-MM-SS>/        (Pi local time)
  run_meta.json          camera, IMU, calibration, solver config, clock state, end-of-run totals
  capture.jsonl          per frame: exposure times, camera metadata, IMU window
  imu.jsonl              every IMU sample (~80 Hz): t, gravity (m/s^2), gyro (rad/s)
  temp.jsonl             IMU temperature every 5 s
  solves/live-<config>.jsonl   per frame: star spots, solution, gravity used, fix, 60 s fix
  images/frame_NNNNN.png       saved frames (1 in N)
```

Everything needed to recompute or recalibrate is in this folder. A 3-hour run
writes about 0.5 GB of logs (the solve log keeps every detected star spot).
