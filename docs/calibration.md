# Calibration

The calibration lives in two files in `configs/`. They belong to your
instrument, so git ignores them.

| File | Content | Made by |
|---|---|---|
| `configs/m_case.json` | rotation from the IMU frame to the camera frame | `fit_mcase.py` |
| `configs/accel_cal.json` | accelerometer model: matrix `A`, offset `b`, temperature term `c` | `calibrate_affine.py`, then `rezero.py` |

The model is

```
zenith_in_camera = normalize( A @ (M_case @ g) + b + c * (T - T0) )
```

where `g` is the measured acceleration and `T` the IMU temperature.

All calibration steps need a **clear sky** and a **known position**: either a
fixed place known to ~200 m (pass `--position=LAT,LON`), or a GPS track in
each run folder as `gps.jsonl` (see
[Checking a run](checking-a-run.md)). The fitting scripts
run on a laptop: copy the run folders from the Pi (`runs/<run>/`) first.

## Step 1: mounting rotation (M_CASE)

Without `configs/m_case.json`, the capture uses the identity and the fixes are
wrong. If you built the reference enclosure, you can start from
`examples/reference-build/m_case.json`; otherwise measure it:

1. Record two runs of about 10 minutes at the known position: one with the
   enclosure flat, one with one side lifted by 20-30°.
2. On the laptop:

   ```bash
   python fit_mcase.py runs/<flat> runs/<tilted> --lat 48.8566 --lon 2.3522
   ```

   It writes `configs/m_case.json`. Copy it to the Pi.

## Step 2: accelerometer calibration

The accelerometer has gains, cross-axis errors and an offset. They change the
fix by tens of nmi and depend on the orientation, so this step needs several
orientations.

1. In one session, at the known position, record **6 to 9 runs of 5-10
   minutes** with the enclosure in clearly different orientations: flat,
   tilted 20-50° toward different sides, and rolled about the camera axis.
   Change the orientation by moving the whole enclosure, never by pressing
   the camera or the IMU.
2. For the temperature term, start with a cold instrument: the capture load
   warms the Pi by a few °C in the first 20 minutes.
3. On the laptop:

   ```bash
   python calibrate_affine.py runs/<r1> runs/<r2> ... --position=48.8566,2.3522 \
       --temp --out configs/accel_cal.json
   ```

   It prints the error per run and a "leave one orientation out" test, which
   is the honest number for a new orientation. Copy the file to the Pi.

The capture reads both files at its start. A run without
`accel_cal.json` is uncalibrated, and the webui checklist shows it.

## Step 3: re-zero (after a knock, or once installed on the ship)

A knock or a new mount can turn the camera relative to the IMU by a small
angle. The gains and the offset stay the same, so you only need to measure this
rotation:

- **At a known fixed position:** one flat run and one tilted run (~30°):

  ```bash
  python rezero.py runs/<flat> runs/<tilted> --cal configs/accel_cal.json \
      --position=48.8566,2.3522 --out configs/accel_cal.json.new
  ```

- **At sea, with the ship's GPS track** in each run's `gps.jsonl`: one or more
  runs of 1-4 hours, without `--position`. Runs on two nights, with the ship on
  different headings, give a better result than one run.

  ```bash
  python rezero.py runs/<night1> runs/<night2> --cal configs/accel_cal.json \
      --out configs/accel_cal.json.new
  ```

Then replace the file on the Pi (keep a copy of the old one).

## Optional: multi-heading correction

An error that is fixed in the instrument (a calibration residual, a small
shift) moves the fix in a direction that turns with the instrument. With runs
at several headings (at least 60° apart, on both sides),
`multiheading_fix.py` separates this error from the position. At a berth, with
5 headings, it gave 0.42 nmi where single runs gave 3.5-4.9 nmi.

```bash
python multiheading_fix.py runs/<r1> runs/<r2> runs/<r3> [--dr-from-gps]
```
