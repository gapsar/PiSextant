#!/usr/bin/env python3
"""Lever-arm test -- runs on the laptop, reads sea runs, writes nothing.

The accelerometer measures gravity plus the acceleration of the instrument.
For a rigid ship, that acceleration is a_O + dw/dt x r + w x (w x r): a_O is
the acceleration of some point O of the ship, r the arm from O to the
instrument, w the angular rate (gyro). The two lever-arm terms are known from
the gyro once r is known, and r is the same for every frame of a run.

Per solved frame, the true "up" in the camera frame comes from the star
attitude and the truth track (gps.jsonl). The horizontal part of the measured
gravity minus true up is the false tilt. This script fits r (3 numbers) plus
a constant offset (3 numbers) to that false tilt, by linear least squares, and
reports the fix error before and after the correction:
  - one-frame scatter and 60 s smoothed fix, on the whole run (in-sample);
  - the same with r fitted on one half of the run and applied to the other
    (out-of-sample);
  - the vertical part, which is NOT fitted: if the model is right, the
    measured |g| changes must follow the predicted vertical lever-arm term.

    python lever_arm_test.py runs/<run> [runs/<run> ...] [--skip-min 0]
"""

import argparse
import json
import math
import os

import numpy as np
from scipy.signal import butter, filtfilt

import calibrate_rcal as cr

GRID_HZ = 50.0        # IMU resampled onto a uniform grid (logged ~79 Hz, jittery)
GYRO_LP_HZ = 2.0      # gyro low-pass before the derivative (waves are < 0.5 Hz)
BIAS_WINDOW_S = 300.0 # running median of the gyro = its zero-rate offset
GATE_DEG = 2.0        # frames with a larger false tilt are false solves
SMOOTH_S = 60.0
RAD_TO_NMI = 180 / math.pi * 60


def jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def skew(v):
    """[v]x for an array of vectors (..., 3) -> (..., 3, 3)."""
    z = np.zeros(v.shape[:-1])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([np.stack([z, -w, y], -1),
                     np.stack([w, z, -x], -1),
                     np.stack([-y, x, z], -1)], -2)


def running_median(x, n):
    """Running median along axis 0 over n samples, computed on a coarse grid."""
    step = max(1, n // 20)
    idx = np.arange(0, len(x), step)
    med = np.array([np.median(x[max(0, i - n // 2):i + n // 2 + 1], axis=0) for i in idx])
    return np.stack([np.interp(np.arange(len(x)), idx, med[:, k]) for k in range(3)], -1)


def load_run(R, skip_min):
    log = cr.pick_solve_log(R, None)
    meta, frames = None, []
    for d in jsonl(log):
        if d.get("type") == "meta":
            meta = d
        elif d.get("type") == "frame" and d.get("fix") and d.get("solution", {}).get("RA") is not None:
            s = d["solution"]
            frames.append((d["frame_id"], s["RA"], s["Dec"], s["Roll"],
                           d["fix"]["era_deg"], d.get("imu_temp_c")))
    cap = {d["frame_id"]: (d["t_exposure_start"], d["t_exposure_end"])
           for d in jsonl(os.path.join(R, "capture.jsonl")) if "frame_id" in d}
    imu = [(d["t"], d["gravity"], d["gyro"]) for d in jsonl(os.path.join(R, "imu.jsonl"))
           if d.get("gravity") and d.get("gyro")]
    gps = sorted((g for g in jsonl(os.path.join(R, "gps.jsonl")) if "lat" in g),
                 key=lambda g: g["t_gps"])
    cal = dict(M=np.array(meta["M_case"]), A=np.array(meta["R_cal"]),
               b=np.array(meta["cal_b"]),
               c=np.array(meta["cal_c"]) if meta.get("cal_c") else None,
               T0=meta.get("cal_temp_ref_c", 25.0))
    ti = np.array([x[0] for x in imu])
    acc = np.array([x[1] for x in imu])
    gyr = np.array([x[2] for x in imu])
    t_first = min(cap[f[0]][0] for f in frames if f[0] in cap)
    frames = [f for f in frames if f[0] in cap and cap[f[0]][0] >= t_first + 60 * skip_min]
    return frames, cap, ti, acc, gyr, gps, cal


def prepare(R, skip_min):
    frames, cap, ti, acc, gyr, gps, cal = load_run(R, skip_min)
    M = cal["M"]
    # uniform grid; gyro and accel in the camera frame
    tg = np.arange(ti[0], ti[-1], 1 / GRID_HZ)
    gap = np.interp(tg, ti[1:], np.diff(ti)) > 0.2
    a_g = np.stack([np.interp(tg, ti, acc[:, k]) for k in range(3)], -1)
    w_raw = np.stack([np.interp(tg, ti, gyr[:, k]) for k in range(3)], -1) @ M.T
    bb, aa = butter(4, GYRO_LP_HZ / (GRID_HZ / 2))
    w_lp = filtfilt(bb, aa, w_raw, axis=0)
    w = w_lp - running_median(w_lp, int(BIAS_WINDOW_S * GRID_HZ))
    wdot = np.gradient(w_lp, 1 / GRID_HZ, axis=0)          # offset-free
    Mlev = skew(wdot) + skew(w) @ skew(w)                     # a_lever = Mlev @ r

    gt = np.array([g["t_gps"] for g in gps])
    gla = np.array([g["lat"] for g in gps])
    glo = np.array([g["lon"] for g in gps])

    rows = []
    for fid, ra, dec, roll, era, temp in frames:
        t0, t1 = cap[fid]
        k = (tg >= t0) & (tg <= t1)
        if k.sum() < 5 or gap[k].any():
            continue
        v = cal["A"] @ (M @ a_g[k].mean(0)) + cal["b"]
        if cal["c"] is not None and temp is not None:
            v = v + cal["c"] * (temp - cal["T0"])
        tm = 0.5 * (t0 + t1)
        c2i = cr.c2i_matrices(tm)[0]
        Ratt = cr.attitude_matrix(ra, dec, roll)
        u = Ratt @ cr.zenith_icrs(np.interp(tm, gt, gla), np.interp(tm, gt, glo), era, c2i)
        rows.append(dict(t=tm, v=v, u=u, Ml=Mlev[k].mean(0), w=w[k].mean(0), R=Ratt, era=era, c2i=c2i,
                         lat=np.interp(tm, gt, gla), lon=np.interp(tm, gt, glo)))
    # false solves out
    tilt = np.array([math.degrees(np.linalg.norm(np.cross(r["v"] / np.linalg.norm(r["v"]), r["u"])))
                     for r in rows])
    keep = tilt < GATE_DEG
    return [r for r, k in zip(rows, keep) if k], int((~keep).sum())


def timing_term(r):
    """Change of the measured gravity if the IMU clock is late by 1 s: the
    up vector turns at -w x u in the camera frame."""
    return -np.linalg.norm(r["v"]) * np.cross(r["w"], r["u"])


def fit(rows, lever=True, timing=True):
    """Least squares: P (v - Mlev r - tau * timing_term - c) = 0,
    P = horizontal projector. Returns (r, tau, c)."""
    X, y = [], []
    for r in rows:
        P = np.eye(3) - np.outer(r["u"], r["u"])
        cols = [P @ r["Ml"] if lever else np.zeros((3, 3)),
                (P @ timing_term(r))[:, None] if timing else np.zeros((3, 1)), P]
        X.append(np.hstack(cols))
        y.append(P @ r["v"])
    p = np.linalg.lstsq(np.vstack(X), np.concatenate(y), rcond=None)[0]
    return p[:3], p[3], p[4:]


def errors(rows, r_arm, tau=0.0):
    """Per-frame N/E fix error (nmi) after removing Mlev @ r_arm and the
    timing term, and the trailing 60 s mean fix error."""
    t, eN, eE = [], [], []
    for r in rows:
        v = r["v"] - r["Ml"] @ r_arm - tau * timing_term(r)
        lat, lon = cr.fix_latlon(r["R"], v / np.linalg.norm(v), r["era"], r["c2i"])
        t.append(r["t"])
        eN.append((lat - r["lat"]) * 60)
        eE.append((lon - r["lon"]) * 60 * math.cos(math.radians(r["lat"])))
    t, eN, eE = map(np.array, (t, eN, eE))
    sm = np.empty(len(t))
    j = 0
    for i in range(len(t)):
        while t[j] < t[i] - SMOOTH_S:
            j += 1
        sm[i] = np.hypot(eN[j:i + 1].mean(), eE[j:i + 1].mean())
    k = t > t[0] + SMOOTH_S
    return eN, eE, sm[k]


def summary(rows, r_arm, tau=0.0):
    eN, eE, sm = errors(rows, r_arm, tau)
    sc = np.hypot(eN - eN.mean(), eE - eE.mean())
    return (f"scatter rms {np.sqrt(np.mean(sc ** 2)):5.1f}  "
            f"60 s fix median {np.median(sm):5.2f} p95 {np.percentile(sm, 95):5.2f} nmi")


def vertical_check(rows, r_arm):
    """Correlation of measured |g| variations with the predicted vertical
    lever-arm acceleration (not fitted)."""
    mv = np.array([r["u"] @ r["v"] for r in rows])
    pv = np.array([r["u"] @ (r["Ml"] @ r_arm) for r in rows])
    mv = mv - mv.mean()
    pv = pv - pv.mean()
    slope = (mv @ pv) / (pv @ pv)
    return np.corrcoef(mv, pv)[0, 1], slope, pv.std(), mv.std()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--skip-min", type=float, default=0.0)
    args = ap.parse_args()
    zero = np.zeros(3)
    for R in args.runs:
        rows, n_gate = prepare(R, args.skip_min)
        print(f"\n=== {os.path.basename(R.rstrip('/'))}: {len(rows)} frames "
              f"({n_gate} gated > {GATE_DEG} deg)")
        if len(rows) < 200:
            print("  too few frames")
            continue
        up = np.mean([r["u"] for r in rows], axis=0)
        up /= np.linalg.norm(up)
        print(f"  before:                {summary(rows, zero)}")
        half = len(rows) // 2
        A, B = rows[:half], rows[half:]
        for name, lev, tim in (("lever arm only", True, False), ("timing only", False, True),
                               ("lever arm + timing", True, True)):
            r_arm, tau, c = fit(rows, lev, tim)
            h = r_arm @ up
            print(f"  -- {name}: r [{r_arm[0]:+6.1f} {r_arm[1]:+6.1f} {r_arm[2]:+6.1f}] m "
                  f"(height above O {h:+5.1f}, horizontal {np.linalg.norm(r_arm - h * up):5.1f}), "
                  f"tau {tau * 1000:+6.0f} ms")
            print(f"     in-sample:          {summary(rows, r_arm, tau)}")
            rA, tA, _ = fit(A, lev, tim)
            rB, tB, _ = fit(B, lev, tim)
            print(f"     halves: tau {tA * 1000:+.0f} / {tB * 1000:+.0f} ms, "
                  f"r_A [{rA[0]:+.1f} {rA[1]:+.1f} {rA[2]:+.1f}] r_B [{rB[0]:+.1f} {rB[1]:+.1f} {rB[2]:+.1f}]")
            print(f"     B before / with A:  {summary(B, zero)}  |  {summary(B, rA, tA)}")
            print(f"     A before / with B:  {summary(A, zero)}  |  {summary(A, rB, tB)}")
        cc, slope, sp, sm_ = vertical_check(rows, r_arm)
        print(f"  vertical (not fitted): corr {cc:+.2f}, slope {slope:+.2f} (1 = model), "
              f"predicted std {sp:.3f} vs measured std {sm_:.3f} m/s2")


if __name__ == "__main__":
    main()
