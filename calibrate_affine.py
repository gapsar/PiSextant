#!/usr/bin/env python3
"""Fit the orientation-independent accelerometer calibration (affine model).

Why this exists: a single rotation (calibrate_rcal.py) can only be exact at
the orientation it was fit at, because the accelerometer error is a linear
distortion (gains, cross-axis terms, offset), not a rotation -- the reported
|gravity| itself changes with orientation. The model fit here,

    true_up = normalize(A @ m + b)      m = M_case @ gravity (raw, m/s^2)

with A a full 3x3 (absorbing the mounting rotation AND the per-axis gains /
cross-axis terms) and b a bias, is valid at ANY mount orientation. 12
parameters, so it needs a multi-orientation session to be determined:
6-9 runs of ~5-10 min at well-spread tilts and camera rolls, at a known
position (see the README, "Calibration", for the exact protocol).

Usage:
    python calibrate_affine.py runs/<r1> runs/<r2> ...   # the session's runs
    python calibrate_affine.py runs/                     # every run inside
    ... --position=LAT,LON --temp --out configs/r_cal_ism330dhcx.json

Temperature (--temp): the ISM330DHCX accelerometer offset moves with its
temperature (~1.5 nmi/deg C of fix drift, measured 2026-10-09). --temp adds a
term c * (T - T0) to the model (3 more parameters), T being the sensor
temperature at each frame from temp.jsonl. It needs runs spanning a range of
temperatures (capture warms the Pi: start cold, keep capturing ~20 min).

Each run needs gps.jsonl (nmea_logger.py, or written by hand from known
positions), capture.jsonl, imu.jsonl and a solve log. No GPS? If the whole
session was captured WITHOUT moving (on land or at anchor), pass the known
position once instead: --position=LAT,LON (decimal degrees, accurate to
~200 m; keep the '=' when the latitude is negative).
Prints, per run: the fit residual, the mean-fix error and the
per-fix median error, plus a leave-one-orientation-out validation (runs are
auto-grouped by gravity direction; each group is held out, the model refit
on the rest and scored on it -- the honest transfer number). Writes the
calibration json, which capture_run_wave.py reads as
configs/r_cal_ism330dhcx.json on the Pi.
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_rcal as cr

G0 = 9.80665            # target |A @ m + b|
TRIM_DEG = 1.5          # false-solve residual cut, per run
CLUSTER_DEG = 5.0       # runs whose mean gravity directions are closer than
                        # this are one "orientation" for the hold-out test


def collect_raw(run_dir, name_filter, wave_s, fixed_pos=None, skip_min=0.0):
    """Per-frame (m, true_up) pairs with m kept RAW (unnormalized, m/s^2):
    the magnitude carries calibration information the fit uses.

    fixed_pos=(lat, lon) replaces the gps.jsonl truth with one constant
    position -- only valid if the device did not move during the runs
    (on land or at anchor)."""
    t_mid = {c["frame_id"]: c["t_exposure_mid"]
             for c in cr.load_jsonl(os.path.join(run_dir, "capture.jsonl"))}
    if fixed_pos is not None:
        t_all = [t for t in t_mid.values() if t is not None]
        gt = np.array([min(t_all) - 60.0, max(t_all) + 60.0])
        gla = np.array([fixed_pos[0]] * 2)
        glo = np.array([fixed_pos[1]] * 2)
    else:
        gps = [r for r in cr.load_jsonl(os.path.join(run_dir, "gps.jsonl"))
               if r.get("type") == "fix"]
        gps.sort(key=lambda r: r["t_gps"])
        gt = np.array([r["t_gps"] for r in gps])
        gla = np.array([r["lat"] for r in gps])
        glo = np.array([r["lon"] for r in gps])
    lines = cr.load_jsonl(cr.pick_solve_log(run_dir, name_filter))
    m_case = np.array(lines[0]["M_case"], dtype=float)
    frames = [l for l in lines if l.get("type") == "frame"
              and l.get("status") == "success"
              and l.get("solution", {}).get("RA") is not None
              and l.get("gravity_used") is not None
              and gt[0] <= t_mid.get(l["frame_id"], -1) <= gt[-1]]
    if skip_min > 0 and frames:
        t_run0 = min(t for t in t_mid.values() if t is not None)
        frames = [f for f in frames
                  if t_mid[f["frame_id"]] >= t_run0 + skip_min * 60.0]
    tm = np.array([t_mid[f["frame_id"]] for f in frames])
    from astropy.time import Time
    era = Time(tm, format="unix").earth_rotation_angle("tio").deg
    c2i = cr.c2i_matrices(tm) if len(tm) else []
    wave = cr.load_wave_gravity(run_dir, wave_s) if wave_s > 0 else lambda t: None
    temps = [r for r in cr.load_jsonl(os.path.join(run_dir, "temp.jsonl"))
             if r.get("t") is not None and r.get("temp_raw") is not None]
    tt = np.array([r["t"] for r in temps])
    tv = np.array([r["temp_raw"] for r in temps])
    out = []
    for f, t, e, q in zip(frames, tm, era, c2i):
        g = wave(t)
        if g is None:
            g = np.asarray(f["gravity_used"], dtype=float)
        sol = f["solution"]
        att = cr.attitude_matrix(sol["RA"], sol["Dec"], sol["Roll"])
        lat = float(np.interp(t, gt, gla))
        lon = float(np.interp(t, gt, glo))
        out.append({"m": m_case @ g,
                    "temp": float(np.interp(t, tt, tv)) if len(tt) else None,
                    "true": att @ cr.zenith_icrs(lat, lon, e, q), "c2i": q,
                    "att": att, "era": float(e), "gps": (lat, lon)})
    return out


def trim(pairs):
    """Drop false solves: per-run rotation fit, cut residuals > TRIM_DEG."""
    if len(pairs) < 5:
        return pairs
    t = np.array([p["true"] for p in pairs])
    m = np.array([p["m"] / np.linalg.norm(p["m"]) for p in pairs])
    R, _, _ = cr.fit_rotation(t, m)
    res = np.degrees(np.arccos(np.clip(((R @ m.T).T * t).sum(1), -1, 1)))
    return [p for p, r in zip(pairs, res) if r < TRIM_DEG]


def model_vec(pairs, A, b, c=None, t0=0.0):
    """A @ m + b (+ c * (T - t0) with the temperature term)."""
    m = np.array([p["m"] for p in pairs])
    v = (A @ m.T).T + b
    if c is not None:
        dt = np.array([p["temp"] - t0 for p in pairs])
        v = v + np.outer(dt, c)
    return v


def fit_affine(pairs, sigma_dir=0.003, sigma_mag=0.01, temp=False, t0=25.0):
    """Least squares on direction (unit-vector difference / sigma_dir) and
    magnitude (|A@m+b| - G0) / sigma_mag, soft_l1 against leftover false
    solves. Seeded with the pooled rotation fit. temp=True also fits c, the
    offset change per deg C around t0. Returns (A, b, c) -- c is None
    without temp."""
    m = np.array([p["m"] for p in pairs])
    t = np.array([p["true"] for p in pairs])
    R0, _, _ = cr.fit_rotation(t, m / np.linalg.norm(m, axis=1, keepdims=True))

    def unpack(x):
        return x[:9].reshape(3, 3), x[9:12], (x[12:15] if temp else None)

    def resid(x):
        v = model_vec(pairs, *unpack(x), t0=t0)
        n = np.linalg.norm(v, axis=1)
        return np.concatenate([((v / n[:, None] - t) / sigma_dir).ravel(),
                               (n - G0) / sigma_mag])

    scale0 = G0 / np.mean(np.linalg.norm(m, axis=1))
    x0 = np.concatenate([(R0 * scale0).ravel(), np.zeros(6 if temp else 3)])
    fit = least_squares(resid, x0, loss="soft_l1", f_scale=3.0)
    return unpack(fit.x)


def affine_pairs(pairs, A, b, c=None, t0=25.0):
    v = model_vec(pairs, A, b, c, t0)
    out = []
    for p, vi in zip(pairs, v):
        q = dict(p)
        q["meas"] = vi / np.linalg.norm(vi)
        out.append(q)
    return out


def rot_pairs(pairs):
    out = []
    for p in pairs:
        q = dict(p)
        q["meas"] = p["m"] / np.linalg.norm(p["m"])
        out.append(q)
    return out


def mean_dir(pairs):
    d = np.array([p["m"] / np.linalg.norm(p["m"]) for p in pairs]).mean(0)
    return d / np.linalg.norm(d)


def cluster_runs(data):
    """Group runs by mean gravity direction (within CLUSTER_DEG)."""
    clusters = []          # list of (direction, [labels])
    for label, pairs in data.items():
        d = mean_dir(pairs)
        for cd, labels in clusters:
            if np.degrees(np.arccos(np.clip(d @ cd, -1, 1))) < CLUSTER_DEG:
                labels.append(label)
                break
        else:
            clusters.append((d, [label]))
    return clusters


def score(pairs, A, b, c=None, t0=25.0):
    ap = affine_pairs(pairs, A, b, c, t0)
    return (cr.mean_fix_error_nmi(ap, np.eye(3)),
            float(np.median(cr.fix_errors_nmi(ap, np.eye(3)))))


def main():
    ap = argparse.ArgumentParser(
        description="Fit the orientation-independent affine accel calibration")
    ap.add_argument("runs", nargs="+", help="run folders, or a directory of them")
    ap.add_argument("--solve-log", default=None,
                    help="substring picking a solve log (default: the live one)")
    ap.add_argument("--wave-s", type=float, default=0.0,
                    help="wave gravity window in seconds (default 0 = off, "
                         "the same as the live capture)")
    ap.add_argument("--position", default=None, metavar="LAT,LON",
                    help="fixed truth position in decimal degrees, e.g. "
                         "'28.0151,128.6884' -- replaces gps.jsonl for "
                         "sessions captured while NOT moving (land/anchor); "
                         "know it to ~0.1 nmi (~200 m) or better")
    ap.add_argument("--skip-min", type=float, default=0.0,
                    help="minutes to drop at the start of each run (thermal "
                         "warm-up ~10-15; unfrozen settle ~45)")
    ap.add_argument("--temp", action="store_true",
                    help="add the temperature term c * (T - T0) (needs runs "
                         "spanning a temperature range)")
    ap.add_argument("--temp-ref", type=float, default=25.0,
                    help="T0, the reference temperature in deg C (default 25)")
    ap.add_argument("--out", default="data/affine_cal.json",
                    help="output calibration file (default data/affine_cal.json)")
    args = ap.parse_args()

    fixed_pos = None
    if args.position is not None:
        try:
            fixed_pos = tuple(float(v) for v in args.position.split(","))
        except ValueError:
            sys.exit("--position must be 'LAT,LON' in decimal degrees")
        print(f"Truth: fixed position ({fixed_pos[0]:+.5f}, {fixed_pos[1]:+.5f})"
              f" -- assumes the device did not move during the session.")

    run_dirs = cr.expand_run_dirs(args.runs)
    if len(run_dirs) < 2:
        sys.exit("Need at least 2 runs (ideally 8-10 at spread orientations).")

    data = {}
    print(f"{'run':24s} {'pairs':>5s} {'|g| m/s^2':>9s} {'tilt':>6s} {'temp C':>11s}")
    for d in run_dirs:
        label = os.path.basename(os.path.normpath(d))
        pairs = trim(collect_raw(d, args.solve_log, args.wave_s, fixed_pos,
                                 args.skip_min))
        if len(pairs) < 5:
            print(f"{label:24s} {len(pairs):5d}  -- too few pairs, skipped")
            continue
        if args.temp and any(p["temp"] is None for p in pairs):
            print(f"{label:24s} {len(pairs):5d}  -- no temp.jsonl, skipped (--temp)")
            continue
        data[label] = pairs
        gmag = np.mean([np.linalg.norm(p["m"]) for p in pairs])
        tilt = np.degrees(np.arccos(abs(mean_dir(pairs)[0])))
        tp = [p["temp"] for p in pairs if p["temp"] is not None]
        trange = f"{min(tp):5.1f}-{max(tp):4.1f}" if tp else "    -"
        print(f"{label:24s} {len(pairs):5d} {gmag:9.4f} {tilt:5.1f}d {trange}")

    clusters = cluster_runs(data)
    print(f"\n{len(data)} runs, {len(clusters)} distinct orientations "
          f"(need >= 5-6 for a determined fit; the more the better)")
    if len(clusters) < 4:
        print("WARNING: too few orientations -- the fit below is likely "
              "underdetermined and will NOT transfer. Capture more rolls/tilts.")

    pool = [p for pp in data.values() for p in pp]
    T0 = args.temp_ref
    A, b, c = fit_affine(pool, temp=args.temp, t0=T0)
    s = np.linalg.svd(A, compute_uv=False)
    print(f"\nFit: gain spread {(s.max()/s.min()-1)*100:.2f}% "
          f"(healthy: a few %), |b| = {np.linalg.norm(b):.4f} m/s^2 "
          f"(healthy: ~0.1; >0.3 suggests underdetermination)")
    if c is not None:
        tp = [p["temp"] for p in pool]
        print(f"Temperature term: |c| = {np.linalg.norm(c) * 1000:.2f} mm/s^2 per C "
              f"({np.linalg.norm(c) / G0 * 1000:.3f} mg/C), fitted over "
              f"{min(tp):.1f}..{max(tp):.1f} C (T0 = {T0:g} C)")

    print(f"\nIn-sample per run (mean-fix / per-fix-median, nmi):")
    for label, pairs in data.items():
        mf, med = score(pairs, A, b, c, T0)
        print(f"  {label:24s} {mf:7.2f} / {med:6.2f}")

    print(f"\nLeave-one-orientation-out (the number that predicts transfer):")
    for cd, labels in clusters:
        train = [p for l, pp in data.items() if l not in labels for p in pp]
        test = [p for l in labels for p in data[l]]
        if not train or not test:
            continue
        Ai, bi, ci = fit_affine(train, temp=args.temp, t0=T0)
        mf, med = score(test, Ai, bi, ci, T0)
        print(f"  hold out {'+'.join(labels):40s} {mf:7.2f} / {med:6.2f}")

    out = {"A": A.tolist(), "b": b.tolist(),
           "model": "true_up = normalize(A @ (M_case @ gravity) + b)",
           "n_pairs": len(pool), "n_orientations": len(clusters),
           "runs": sorted(data.keys()), "wave_s": args.wave_s,
           "truth": (f"fixed position {fixed_pos}" if fixed_pos
                     else "gps.jsonl per run")}
    if c is not None:
        out.update({"c": c.tolist(), "temp_ref_c": T0,
                    "model": "true_up = normalize(A @ (M_case @ gravity) + b "
                             "+ c * (T - temp_ref_c)), T = IMU temperature in C"})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out} -- copy it to the Pi as "
          f"configs/r_cal_ism330dhcx.json")


if __name__ == "__main__":
    main()
