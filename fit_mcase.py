#!/usr/bin/env python3
"""Measure the camera-to-IMU mounting rotation (M_CASE) from star solves.

Runs on the laptop, on run folders captured at a KNOWN position with the rig
static. For every solved frame the star solve gives the camera attitude, the
known position + time give the zenith, so the true "up" direction is known in
the camera frame; the IMU measured the same "up" in its own frame (median
accel over the exposure). The rotation M with  M @ up_imu ~ up_cam  over all
frames is Wahba's problem, solved by SVD.

One static pose only pins M up to a spin about the vertical; give at least two
runs at clearly different tilts (e.g. flat, then propped up on one side).

Uses exactly the same zenith model as calculate_fix() in the capture scripts
(ERA + longitude, ICRS), so the M it prints is what the pipeline expects. Any
small residual model/mount error is left for calibrate_affine.py to absorb.

    python fit_mcase.py runs/<pose1> runs/<pose2> --lat -21.0 --lon 55.28

It writes configs/m_case.json (change with --out), which the capture script
loads at start. Refit the accelerometer calibration after a new M_CASE.
"""

import argparse
import glob
import json
import os

import erfa
import numpy as np
from astropy.time import Time


def zenith_icrs(lat_deg, lon_deg, t):
    """Zenith unit vector in ICRS, inverse of calculate_fix's lat/lon step:
    built in CIRS (longitude + Earth rotation angle), then rotated to ICRS
    with the precession-nutation matrix (erfa.c2i06a, IAU 2006/2000A)."""
    tm = Time(t, format="iso", scale="utc")
    lst = np.radians(tm.earth_rotation_angle("tio").deg + lon_deg)
    lat = np.radians(lat_deg)
    z_cirs = np.array([np.cos(lat) * np.cos(lst), np.cos(lat) * np.sin(lst),
                       np.sin(lat)])
    tt = tm.tt
    return erfa.c2i06a(tt.jd1, tt.jd2).T @ z_cirs


def icrs_to_cam(ra_deg, dec_deg, roll_deg):
    """Same camera frame as calculate_fix: rows = boresight, 'left', 'up'."""
    ra, dec, roll = np.radians([ra_deg, dec_deg, roll_deg])
    boresight = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                          np.sin(dec)])
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    north = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra),
                      np.cos(dec)])
    return np.array([boresight,
                     np.cos(roll) * east + np.sin(roll) * north,
                     -np.sin(roll) * east + np.cos(roll) * north])


def load_pairs(run_dir, lat, lon):
    """(up_imu, up_cam) unit-vector pairs for every solved frame of a run."""
    pairs = []
    for path in glob.glob(os.path.join(run_dir, "solves", "live-*.jsonl")):
        for line in open(path):
            rec = json.loads(line)
            sol = rec.get("solution") or {}
            g = rec.get("gravity_median") or rec.get("gravity_used")
            if rec.get("type") != "frame" or sol.get("RA") is None or not g:
                continue
            up_imu = np.asarray(g, float) / np.linalg.norm(g)
            up_cam = icrs_to_cam(sol["RA"], sol["Dec"], sol["Roll"]) @ \
                zenith_icrs(lat, lon, rec["time_iso_mid"])
            pairs.append((up_imu, up_cam))
    return pairs


def wahba(a, b):
    """Rotation M minimizing sum |b_i - M a_i|^2 (proper rotation, det +1)."""
    u, _, vt = np.linalg.svd(b.T @ a)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    return u @ d @ vt


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs", "m_case.json"))
    args = parser.parse_args()

    all_a, all_b = [], []
    for run in args.runs:
        pairs = load_pairs(run, args.lat, args.lon)
        if not pairs:
            print(f"{run}: no solved frames with IMU data -- skipped")
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        elev = 90 - np.degrees(np.arccos(np.clip(b[:, 0], -1, 1)))
        print(f"{run}: {len(pairs)} solved frames, camera elevation "
              f"{elev.mean():.2f} deg (spread {elev.std():.3f}), "
              f"IMU up {np.round(a.mean(0), 4)}")
        all_a.append(a)
        all_b.append(b)
    if not all_a:
        raise SystemExit("nothing to fit")
    a, b = np.vstack(all_a), np.vstack(all_b)

    # spread of the "up" directions: < ~5 deg means a single pose, and the
    # spin about the vertical is then NOT determined
    ups = a / np.linalg.norm(a, axis=1, keepdims=True)
    spread = np.degrees(np.arccos(np.clip(ups @ ups.mean(0) /
                                          np.linalg.norm(ups.mean(0)), -1, 1))).max()
    m = wahba(a, b)
    res = np.degrees(np.arccos(np.clip(np.sum((a @ m.T) * b, axis=1), -1, 1)))
    print(f"\nIMU 'up' directions span {spread:.1f} deg "
          f"({'OK, rotation fully determined' if spread > 5 else 'SINGLE POSE: spin about vertical undetermined'})")
    print(f"fit residual: median {np.median(res):.3f} deg, max {res.max():.3f} deg")
    for i, (a_i, b_i) in enumerate(zip(all_a, all_b)):
        r_i = np.degrees(np.arccos(np.clip(np.sum((a_i @ m.T) * b_i, 1), -1, 1)))
        print(f"  run {i + 1}: median residual {np.median(r_i):.3f} deg")
    print("\nM_CASE = np.array([")
    names = ["camera boresight", "camera left     ", "camera up       "]
    for row, name in zip(m, names):
        print(f"    [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}],   # {name}")
    print("])")
    print(f"det = {np.linalg.det(m):+.6f}")
    if spread <= 5:
        print("Not written: add a run at another tilt first.")
        return
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"M_case": m.tolist(),
               "runs": [os.path.basename(r.rstrip("/")) for r in args.runs],
               "position": [args.lat, args.lon],
               "residual_median_deg": float(np.median(res))},
              open(args.out, "w"), indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
