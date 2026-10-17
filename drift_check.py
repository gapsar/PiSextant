#!/usr/bin/env python3
"""Does the fix drift with TEMPERATURE or with TIME? -- runs on the laptop.

For static runs at a known position (no GPS needed). For every solved frame:

  - star-side up:  the true "up" in the camera frame, from the star solve +
                   the known position + time (no IMU involved);
  - IMU-side up:   the accelerometer's "up", through M_CASE and the run's
                   calibration, in the same camera frame;
  - fix error:     the difference between the two (what moves the fix).

A real change of the ship's list moves both sides together (no fix error).
A thermal effect in the IMU moves only the IMU side; a solver / sky-rotation
effect moves only the star side.

Then, over ALL the runs given, it fits the fix error two ways --
  error ~ temperature   and   error ~ time
-- and both together. With temperature going up AND down (heat, then cool),
the two are no longer confounded: the one that explains the error is the
cause. A single warm-up ramp cannot separate them (both rise together).

Usage:
    python drift_check.py runs/<r1> [runs/<r2> ...] --position=48.8566,2.3522
    ... --bin-min 4        # table bin size (minutes; 0.25 = 15 s, e.g. for a press test)

The runs must be untouched during each run; for the regression, use runs at
the SAME orientation (different orientations add calibration residuals).
"""

import argparse
import glob
import json
import os

import numpy as np
from astropy.time import Time

from fit_mcase import icrs_to_cam, zenith_icrs


def load_run(run, lat, lon):
    meta = json.load(open(os.path.join(run, "run_meta.json")))
    m_case = np.array(meta["M_case"])
    cal_a = np.array(meta.get("R_cal", np.eye(3)))
    cal_b = np.array(meta.get("cal_b", [0.0, 0.0, 0.0]))
    temps = [json.loads(l) for l in open(os.path.join(run, "temp.jsonl"))]
    tt = np.array([x["t"] for x in temps])
    tv = np.array([x["temp_raw"] for x in temps])
    rows = []
    for path in glob.glob(os.path.join(run, "solves", "live-*.jsonl")):
        for line in open(path):
            d = json.loads(line)
            s = d.get("solution") or {}
            if d.get("type") != "frame" or s.get("RA") is None or not d.get("gravity_median"):
                continue
            t = Time(d["time_iso_mid"]).unix
            star = icrs_to_cam(s["RA"], s["Dec"], s["Roll"]) @ zenith_icrs(lat, lon, d["time_iso_mid"])
            v = cal_a @ (m_case @ np.asarray(d["gravity_median"], float)) + cal_b
            rows.append((t, np.interp(t, tt, tv), *star, *(v / np.linalg.norm(v)),
                         np.linalg.norm(d["gravity_median"])))
    return np.array(rows)


def tilt_axes(v0):
    """Two unit vectors perpendicular to v0 (the run's mean up direction)."""
    e1 = np.cross(v0, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(v0, e1)


def r2(x, y):
    a = np.c_[x, np.ones(len(x))]
    res = y - a @ np.linalg.lstsq(a, y, rcond=None)[0]
    return 1 - np.var(res) / np.var(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--position", required=True, metavar="LAT,LON",
                    help="known position; write --position=-21.0,55.3 (with '=') "
                         "when the latitude is negative")
    ap.add_argument("--bin-min", type=float, default=4.0)
    args = ap.parse_args()
    lat, lon = (float(v) for v in args.position.split(","))

    data = [load_run(r, lat, lon) for r in args.runs]
    data = [(r, d) for r, d in zip(args.runs, data) if len(d) >= 10]
    if not data:
        raise SystemExit("no run with enough solved frames")
    allr = np.vstack([d for _, d in data])
    up0 = allr[:, 2:5].mean(0)
    e1, e2 = tilt_axes(up0 / np.linalg.norm(up0))
    t_first = allr[:, 0].min()

    def comps(v):            # deg along the two tilt axes
        return np.degrees(np.c_[v @ e1, v @ e2])

    print("Per run, in bins (deg; nmi = deg x 60). Star and IMU columns are the change "
          "since the first bin of all runs.")
    ref_s = comps(allr[:, 2:5])[: 60].mean(0)
    ref_i = comps(allr[:, 5:8])[: 60].mean(0)
    for run, d in data:
        print(f"\n{os.path.basename(run)}")
        print("  m:ss    temp C   star (ax1, ax2)    IMU (ax1, ax2)    fix error nmi (ax1, ax2)   |g| m/s2")
        t = (d[:, 0] - t_first) / 60.0
        bins = (t // args.bin_min).astype(int)
        S, I = comps(d[:, 2:5]) - ref_s, comps(d[:, 5:8]) - ref_i
        for k in np.unique(bins):
            s = bins == k
            err = (I - S)[s].mean(0) * 60
            t0s = k * args.bin_min * 60
            print(f"  {int(t0s // 60):3d}:{int(t0s % 60):02d}  {d[s, 1].mean():6.2f}   "
                  f"{S[s, 0].mean():+.3f} {S[s, 1].mean():+.3f}     "
                  f"{I[s, 0].mean():+.3f} {I[s, 1].mean():+.3f}     "
                  f"{err[0]:+6.2f} {err[1]:+6.2f}             {d[s, 8].mean():.4f}")

    T = allr[:, 1]
    tm = (allr[:, 0] - t_first) / 60.0
    err = (comps(allr[:, 5:8]) - comps(allr[:, 2:5])) * 60
    print(f"\nAll runs together ({len(allr)} frames, temperature {T.min():.1f}..{T.max():.1f} C, "
          f"{tm.max():.0f} min). Explained variance R^2:")
    print("                       by temperature   by time   by both")
    for name, y in (("fix error, axis 1", err[:, 0]), ("fix error, axis 2", err[:, 1]),
                    ("|g| (IMU only)", allr[:, 8])):
        both = np.c_[T, tm, np.ones(len(T))]
        res = y - both @ np.linalg.lstsq(both, y, rcond=None)[0]
        print(f"  {name:20s}   {r2(T, y):10.2f}   {r2(tm, y):7.2f}   {1 - np.var(res) / np.var(y):7.2f}")
    if np.corrcoef(T, tm)[0, 1] ** 2 > 0.8:
        print("\nWARNING: temperature and time are strongly correlated here (r^2 > 0.8): "
              "the two cannot be told apart. Add a cooling phase (temperature going DOWN).")
    else:
        print("\nTemperature and time are decorrelated enough: the larger R^2 points to the cause.")


if __name__ == "__main__":
    main()
