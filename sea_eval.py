#!/usr/bin/env python3
"""Fix error of one run against its GPS truth -- runs on the laptop.

Reads the fixes of the live solve log (or the one picked with --solve-log,
e.g. a replay) and the truth track in gps.jsonl (linear interpolation between the points; write extra points to
extrapolate). Prints the overall offset, the 60 s fix error, a table per
time bin (error, solve rate, temperature, camera azimuth) and the drift
with temperature and with time.

    python sea_eval.py runs/<run> [--bin-min 10] [--gps other_gps.jsonl]
    python sea_eval.py runs/<run> --solve-log replay-     # a replayed log
"""

import argparse
import json
import os

import numpy as np

import calibrate_rcal as cr
import multiheading_fix as mh


def jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--bin-min", type=float, default=10.0)
    ap.add_argument("--gps", help="truth file (default: <run>/gps.jsonl)")
    ap.add_argument("--solve-log", default=None,
                    help="substring picking a solve log (default: the live one)")
    args = ap.parse_args()
    R = args.run

    gps = [g for g in jsonl(args.gps or os.path.join(R, "gps.jsonl")) if "lat" in g]
    gps.sort(key=lambda g: g["t_gps"])
    gt = np.array([g["t_gps"] for g in gps], float)
    gla = np.array([g["lat"] for g in gps])
    glo = np.array([g["lon"] for g in gps])

    t_mid = {d["frame_id"]: d.get("t_exposure_mid")
             for d in jsonl(os.path.join(R, "capture.jsonl")) if "frame_id" in d}
    rows, t_all = [], []
    log = cr.pick_solve_log(R, args.solve_log)
    if log is None:
        raise SystemExit(f"no solve log in {R}/solves")
    print(f"solve log: {os.path.basename(log)}")
    for d in jsonl(log):
        if d.get("type") != "frame":
            continue
        t = t_mid.get(d["frame_id"])
        if t is None:
            continue
        t_all.append(t)
        if not d.get("fix"):
            continue
        wv = d.get("fix_wave_smoothed") or {}     # wave learner (v2.2+), logged only
        rows.append((t, d["fix"]["latitude"], d["fix"]["longitude"],
                     d["fix_smoothed"]["latitude"], d["fix_smoothed"]["longitude"],
                     d.get("imu_temp_c") or np.nan,
                     wv.get("latitude", np.nan), wv.get("longitude", np.nan)))
    a = np.array(rows)
    t_all = np.array(sorted(t_all))
    t = a[:, 0]
    if t.min() < gt[0] or t.max() > gt[-1]:
        print(f"NOTE: truth covers only {(gt[-1] - gt[0]) / 60:.0f} min; frames outside are "
              "held at the end positions -- add extrapolated points to gps.jsonl")
    tl, tn = np.interp(t, gt, gla), np.interp(t, gt, glo)
    c = np.cos(np.radians(tl))
    eN, eE = (a[:, 1] - tl) * 60, (a[:, 2] - tn) * 60 * c
    sN, sE = (a[:, 3] - tl) * 60, (a[:, 4] - tn) * 60 * c
    T = a[:, 5]
    m = (t - t_all[0]) / 60
    ma = (t_all - t_all[0]) / 60

    frames = mh.load_frames(R)
    ht = np.array([f["t"] for f in frames])
    hv = np.array([mh.heading_deg(f) for f in frames])

    k = m > 1
    e = np.hypot(sN, sE)[k]
    print(f"frames solved {len(a)}/{len(t_all)} ({len(a) / len(t_all):.0%}), {ma[-1]:.0f} min, "
          f"IMU temp {np.nanmin(T):.1f}..{np.nanmax(T):.1f} C")
    print(f"offset (median of frames): N {np.median(eN):+.2f}  E {np.median(eE):+.2f}  "
          f"-> {np.hypot(np.median(eN), np.median(eE)):.2f} nmi")
    print(f"one frame scatter (std): N {eN.std():.1f}  E {eE.std():.1f} nmi")
    print(f"60 s fix error: median {np.median(e):.2f}  p95 {np.percentile(e, 95):.2f}  "
          f"max {e.max():.2f} nmi")
    wN, wE = (a[:, 6] - tl) * 60, (a[:, 7] - tn) * 60 * c
    kw = k & ~np.isnan(wN)
    if kw.sum() > 10:
        es, ew = np.hypot(sN, sE)[kw], np.hypot(wN, wE)[kw]
        print(f"wave learner (logged only), same {kw.sum()} frames: 60 s fix median "
              f"{np.median(es):.2f} -> {np.median(ew):.2f}  p95 {np.percentile(es, 95):.2f} -> "
              f"{np.percentile(ew, 95):.2f} nmi")

    B = args.bin_min
    print(f"\n  min   solved     N_err  E_err  |err|   temp  cam_az")
    for b0 in np.arange(0, ma[-1], B):
        kb = (m >= b0) & (m < b0 + B)
        n_all = ((ma >= b0) & (ma < b0 + B)).sum()
        kh = (ht >= t_all[0] + b0 * 60) & (ht < t_all[0] + (b0 + B) * 60)
        az = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(hv[kh]))))) % 360 if kh.any() else np.nan
        if kb.sum() < 5:
            print(f" {b0:4.0f}  {kb.sum():4d}/{n_all:4d}")
            continue
        n_, e_ = np.median(eN[kb]), np.median(eE[kb])
        print(f" {b0:4.0f}  {kb.sum():4d}/{n_all:4d}   {n_:+6.2f} {e_:+6.2f} {np.hypot(n_, e_):6.2f} "
              f"{np.nanmean(T[kb]):6.2f}  {az:6.1f}")

    ok = ~np.isnan(T)
    for lo in (0, 20):
        kk = ok & (m >= lo)
        if kk.sum() < 50:
            continue
        X = np.c_[np.ones(kk.sum()), T[kk] - T[kk].mean(), m[kk] / 60]
        for name, ee in (("N", eN), ("E", eE)):
            cf = np.linalg.lstsq(X, ee[kk], rcond=None)[0]
            print(f" from {lo:2d} min, {name}: {cf[1]:+.2f} nmi/C and {cf[2]:+.2f} nmi/h (joint fit)")


if __name__ == "__main__":
    main()
