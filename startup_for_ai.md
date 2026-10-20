# PiSextant -- guide for AI assistants

You are helping a person build, install, calibrate or run PiSextant. Read
this file first. It gives the facts and the rules you need; the human
documentation is in `README.md` and `docs/`.

## What PiSextant is

A Raspberry Pi with a camera and an IMU computes its latitude and longitude
from star images, with no GPS:

- the star solver gives the camera attitude (RA, Dec, Roll);
- the accelerometer gives gravity; the calibration turns it into the zenith
  in the camera frame;
- attitude + zenith + exact time (UT1, precession-nutation) give the position.

One frame gives one fix; the 60-second average of the fixes (`fix_smoothed`)
is the navigation output. Scale: 1 arc minute of error in the vertical =
1 nautical mile (nmi) of position error; 1 s of clock error = 0.25 nmi of
longitude.

## Repository layout

| Path | Role |
|---|---|
| `capture_run_wave.py` | Pi, capture venv: camera + IMU + fixes, writes `runs/<id>/` |
| `wave_learner.py` | online wave filter used by the capture (logged only: `fix_wave`) |
| `solve_worker.py` | Pi, solver venv: star detection + solving, Unix socket `/tmp/pisextant_solver.sock` |
| `imu_backends.py`, `imu_drivers/` | IMU driver loader and one driver per sensor |
| `webui/server.py` | Pi, system python3 (stdlib only): web page on port 8080, starts/stops the two processes |
| `configs/` | the user's instrument files (git-ignored), see `configs/README.md` |
| `examples/reference-build/` | the reference instrument's configs, calibration and star database |
| `fit_mcase.py`, `calibrate_affine.py`, `rezero.py` | calibration (laptop) |
| `calibrate_rcal.py` | shared helpers (zenith model, solve-log reading, rotation fit) |
| `sea_eval.py`, `drift_check.py`, `multiheading_fix.py` | evaluation (laptop) |
| `wave_learner_replay.py`, `lever_arm_test.py` | wave filter replay on logged runs; lever-arm test (laptop) |
| `replay_solve.py`, `build_database.py` | re-solve saved images; build a star database |
| `nmea_logger.py`, `set_clock_from_gps.py` | GPS truth logging; clock setting |
| `patches/` | two patches for cedar-solve (apply both) |
| `requirements/` | pinned versions for the two Pi venvs and the laptop |

## Configuration contract (configs/)

- `instrument.json` (required): `"camera"` (`sensor_mode`, `exposure_us`,
  `analogue_gain`, optional `lens_position` for autofocus cameras only) and
  `"imu"` (`driver` = module name in `imu_drivers/`, other keys go to the
  driver).
- `solver.json`: `database.path` (relative to the repo root) must match the
  camera's horizontal field of view; `solve.fov_estimate` close to it.
- `m_case.json`: `{"M_case": 3x3}` rotation IMU frame -> camera frame (rows:
  camera boresight, left, up). Written by `fit_mcase.py`. Missing = identity
  = wrong fixes.
- `accel_cal.json`: `{"A": 3x3, "b": [3], "c": [3] or null, "temp_ref_c": T0}`.
  Model: `zenith_cam = normalize(A @ (M_case @ g) + b + c * (T - T0))`.
  Written by `calibrate_affine.py` / `rezero.py`.

The capture reads these files only at its start. `M_case` and the
calibration were fitted together: never change one without refitting the
other.

## IMU driver contract

`imu_drivers/<name>.py` exposes `open_imu(cfg)` returning an object with:
`acceleration` (raw, m/s², 3 values, |a| ~ 9.8 at rest), `gyro` (rad/s),
`temperature` (°C or `None`). Template: `imu_drivers/template.py`; working
example: `imu_drivers/ism330dhcx.py`. Do not swap or flip axes inside a
driver after calibration.

## Typical tasks

**Install on a Pi:** follow `README.md` "Install" in order. Facts:
- capture venv = system Python 3.13 with `--system-site-packages`
  (picamera2 from apt);
- solver venv = Python 3.11 via `uv` (cedar-solve needs numpy < 2);
- cedar-solve at commit `191b8f6`, both patches applied, then
  `cedar-detect-server` (Rust) built for aarch64 and copied to cedar-solve's
  `tetra3/bin/` (or `CEDAR_DETECT_BINARY`);
- RTC: `dtoverlay=i2c-rtc,ds3231` (or the overlay of the user's RTC chip).

**New camera:** horizontal FOV = `2*atan(sensor_width_mm / (2*focal_mm))`;
`python build_database.py --catalog hip_main.dat --max-fov <fov>` (needs the
Hipparcos `hip_main.dat` file and cedar-solve on the computer that builds);
set `configs/solver.json`; pick `sensor_mode` and exposure in
`instrument.json`.

**New IMU:** write the driver from the template, set `imu.driver`, install
its library in the capture venv, test it, then do all calibration steps.

**Calibrate** (`docs/calibration.md`), all at a known position (±200 m,
`--position=LAT,LON`) or with `gps.jsonl` truth in each run folder:
1. `fit_mcase.py` on two runs: flat, and tilted 20-30°;
2. `calibrate_affine.py --temp` on 6-9 orientations (needed when the
   instrument is used at several orientations; for a fixed, near-level
   mount, step 3 alone on top of a rough step 2 can be enough);
3. `rezero.py` after any knock or a new mount (one or two nights with a GPS
   track, without `--position`).
Calibration runs must use `--wave-s 0` (the default).

**Evaluate a run:** write `gps.jsonl` (`{"type": "fix", "t_gps": <unix UTC>,
"lat": .., "lon": ..}` per line; linear interpolation between points), then
`python sea_eval.py runs/<id>`.

## Rules

1. **Prepare offline use.** At sea the instrument usually has no internet:
   before the user leaves, make sure that the packages, the IERS table and the
   map tiles are on the Pi.
2. **Never delete runs, images or calibration files** without an explicit
   yes. Back up a config before you replace it (`<file>.bak-<date>`).
3. **Do not change the fix model** (`calculate_fix` in the capture and the
   helpers in `calibrate_rcal.py`) unless asked; the calibration depends on
   it. Validate any change by replaying logged runs: the new fixes must equal
   the logged ones.
4. **Do not interpret an event** (a jump in the fix, a temperature change)
   as a cause before the user confirms what happened physically.
5. The Pi Zero 2 W has 512 MB of RAM: never load a whole run log into
   memory (a 3 h solve log is ~0.5 GB); stream it line by line.
6. Clock: without NTP the RTC is the time source (fine). After the Pi had
   NTP, `sudo hwclock -w` writes the RTC.
7. Keep the IERS table fresh (`docs/installation.md`): a stale table gives up
   to 0.25 nmi of longitude error, and the laptop and the Pi must agree.

## Troubleshooting

| Symptom | Likely cause | Check |
|---|---|---|
| fixes tens or hundreds of nmi off | no or wrong `m_case.json` / `accel_cal.json`, or the boards moved | capture start lines `M_CASE:` and `Cal:`; re-zero |
| offset grows by several nmi after handling | camera-IMU geometry shifted (0.1° = 6 nmi) | `rezero.py` |
| offset changes with time in the first 20 min | warm-up (IMU temperature) | `temp.jsonl`; wait or start earlier |
| solver never solves | wrong database FOV, bad focus, clouds, lights | `solves/live-*.jsonl` statuses and `n_centroids` |
| webui page dies at boot | out of memory (Pi Zero) | `journalctl -u pisextant-webui`, `dmesg` for OOM |
| `rtc-ds1307 ... SET TIME!` at boot | RTC oscillator-stop flag | `sudo hwclock -w` when the time is good |
| longitude bias ~0.1-0.25 nmi between laptop and Pi | different IERS tables | refresh both |
| capture: `instrument.json not found` | configs not written yet | `configs/README.md` |

## Data formats

- `runs/<id>/imu.jsonl`: `{"t", "gravity": [3], "gyro": [3]}` per sample.
- `runs/<id>/temp.jsonl`: `{"t", "temp_raw"}` in °C every 5 s.
- `runs/<id>/capture.jsonl`: per frame, exposure start/mid/end (unix), IMU window.
- `runs/<id>/solves/live-<config>.jsonl`: first line `type: meta`, then per
  frame `status`, `centroids`, `solution` (RA, Dec, Roll, FOV, Matches,
  RMSE), `gravity_used`, `imu_temp_c`, `fix`, `fix_smoothed`
  (`n_fixes`, `n_rejected`); since v2.2 also `fix_wave`, `fix_wave_smoothed`
  and `wave_learner` (`active`, `models`, `n_train`, `error`) -- logged
  only, the fix does not use them.
- `runs/<id>/run_meta.json`: camera, IMU, calibration, solver config, clock
  state, end-of-run totals.
