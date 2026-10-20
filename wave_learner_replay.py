#!/usr/bin/env python3
"""Replay of the online wave filter (wave_learner.py) on a logged run --
runs on the laptop, writes nothing.

Feeds the run's IMU samples and solved frames to the filter in time order,
exactly as capture_run_wave.py does live (the filter never sees the truth),
then scores both fixes against the truth track in gps.jsonl:
  - "before" = the logged live fix (recomputed here; the max difference to
    the logged fix is printed as a check),
  - "after"  = the fix from the filter's corrected up vector.
Only frames where the filter is active are scored (it needs ~5 min of gyro
for its offset and ~20 min of frames before its first model).

    python wave_learner_replay.py runs/<run> [runs/<run> ...] [--end-min 75]
"""

import argparse
import json
import math
import os

import numpy as np

import calibrate_rcal as cr
import wave_learner as wl

SMOOTH_S = 60.0
SMOOTH_GATE_DEG = 5.0
FALSE_DEG = 2.0         # scoring only: live fixes this far from the truth are false solves


def jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def smoothed(t, lat, lon):
    """Trailing SMOOTH_S mean with the live false-solve gate."""
    out, j = [], 0
    for i in range(len(t)):
        while t[j] < t[i] - SMOOTH_S:
            j += 1
        la, lo = lat[j:i + 1], lon[j:i + 1]
        keep = np.hypot(la - np.median(la), (lo - np.median(lo)) *
                        math.cos(math.radians(la[-1]))) <= SMOOTH_GATE_DEG
        out.append((la[keep].mean(), lo[keep].mean()))
    return np.array(out)


def score(t, lat, lon, tl, tn):
    eN = (lat - tl) * 60
    eE = (lon - tn) * 60 * np.cos(np.radians(tl))
    s = smoothed(t, lat, lon)
    sN, sE = (s[:, 0] - tl) * 60, (s[:, 1] - tn) * 60 * np.cos(np.radians(tl))
    k = t > t[0] + SMOOTH_S
    rms = np.sqrt(np.mean((eN - eN.mean()) ** 2 + (eE - eE.mean()) ** 2))
    e60 = np.hypot(sN, sE)[k]
    wave60 = np.sqrt(np.mean((sN[k] - sN[k].mean()) ** 2 + (sE[k] - sE[k].mean()) ** 2))
    return rms, wave60, np.median(e60), np.percentile(e60, 95)


def replay(R, end_min):
    log = cr.pick_solve_log(R, None)
    meta, frames = None, []
    for d in jsonl(log):
        if d.get("type") == "meta":
            meta = d
        elif (d.get("type") == "frame" and d.get("fix") and d.get("gravity_used")
              and (d.get("solution") or {}).get("RA") is not None):
            frames.append(d)
    t_mid = {d["frame_id"]: d["t_exposure_mid"]
             for d in jsonl(os.path.join(R, "capture.jsonl")) if "frame_id" in d}
    frames = [f for f in frames if f["frame_id"] in t_mid]
    frames.sort(key=lambda f: t_mid[f["frame_id"]])
    if end_min:
        t_end = t_mid[frames[0]["frame_id"]] + 60 * end_min
        frames = [f for f in frames if t_mid[f["frame_id"]] < t_end]
    M = np.array(meta["M_case"])
    A, b = np.array(meta["R_cal"]), np.array(meta["cal_b"])
    c = np.array(meta["cal_c"]) if meta.get("cal_c") else None
    T0 = meta.get("cal_temp_ref_c", 25.0)

    imu = [(d["t"], d["gyro"]) for d in jsonl(os.path.join(R, "imu.jsonl")) if d.get("gyro")]
    ft = np.array([t_mid[f["frame_id"]] for f in frames])
    c2i = cr.c2i_matrices(ft)

    feats, filt = wl.GyroFeatures(), wl.OnlineTiltFilter()
    rows, j, check = [], 0, []
    for i, f in enumerate(frames):
        while j < len(imu) and imu[j][0] <= ft[i]:
            feats.add(imu[j][0], M @ np.asarray(imu[j][1]))
            j += 1
        s = f["solution"]
        v = A @ (M @ np.asarray(f["gravity_used"])) + b
        if c is not None and f.get("imu_temp_c") is not None:
            v = v + c * (f["imu_temp_c"] - T0)
        up = v / np.linalg.norm(v)
        Ratt = cr.attitude_matrix(s["RA"], s["Dec"], s["Roll"])
        era = f["fix"]["era_deg"]
        up_corr = filt.add_frame(ft[i], feats.features(ft[i]), up,
                                 Ratt @ c2i[i].T @ rot_z(era))
        lat0, lon0 = cr.fix_latlon(Ratt, up, era, c2i[i])
        check.append(max(abs(lat0 - f["fix"]["latitude"]),
                         abs(lon0 - f["fix"]["longitude"])) * 3600)
        if up_corr is not None:
            lat1, lon1 = cr.fix_latlon(Ratt, up_corr, era, c2i[i])
            rows.append((ft[i], lat0, lon0, lat1, lon1))
    return np.array(rows), np.array(check), filt, len(frames)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--end-min", type=float, default=None,
                    help="ignore frames after this many minutes (e.g. instrument handled)")
    args = ap.parse_args()
    print("                        frame rms   60 s wave part   60 s median   60 s p95   (nmi)")
    for R in args.runs:
        name = os.path.basename(R.rstrip("/"))
        rows, chk, filt, n = replay(R, args.end_min)
        if len(rows) < 100:
            print(f"{name}: filter active on {len(rows)} frames only -- run too short")
            continue
        gps = sorted((g for g in jsonl(os.path.join(R, "gps.jsonl")) if "lat" in g),
                     key=lambda g: g["t_gps"])
        gt = np.array([g["t_gps"] for g in gps])
        tl = np.interp(rows[:, 0], gt, [g["lat"] for g in gps])
        tn = np.interp(rows[:, 0], gt, [g["lon"] for g in gps])
        far = np.hypot(rows[:, 1] - tl, (rows[:, 2] - tn) * np.cos(np.radians(tl))) > FALSE_DEG
        rows, tl, tn = rows[~far], tl[~far], tn[~far]          # false solves out of both scores
        b = score(rows[:, 0], rows[:, 1], rows[:, 2], tl, tn)
        a = score(rows[:, 0], rows[:, 3], rows[:, 4], tl, tn)
        print(f"{name}  before  {b[0]:6.1f}       {b[1]:6.2f}        {b[2]:6.2f}     {b[3]:6.2f}")
        print(f"{'':19s}  after   {a[0]:6.1f}       {a[1]:6.2f}        {a[2]:6.2f}     {a[3]:6.2f}"
              f"   ({len(rows)}/{n} frames active, {filt.n_models} models, "
              f"check vs logged fix: median {np.median(chk):.2g} arcsec, "
              f"{(chk > 1).sum()} frames > 1 arcsec; {far.sum()} false solves not scored)")


if __name__ == "__main__":
    main()
