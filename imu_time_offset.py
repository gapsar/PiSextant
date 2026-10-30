#!/usr/bin/env python3
"""IMU-camera time offset from the logs -- runs on the laptop, writes nothing.

Two consecutive solved frames give the camera rotation between them from
the stars alone (attitude precision ~0.03 deg). The gyro gives the same
rotation when it is integrated over the same interval -- but only if the
IMU and the frame timestamps agree. This script searches the time shift tau
that makes them agree best:

    star rotation(t_i -> t_j) = s * integral of gyro over [t_i + tau, t_j + tau] - b * (t_j - t_i)

with s a gyro scale (its sign also gives the rotation convention) and b
one gyro zero-rate offset per BLOCK_S. t_i, t_j are the logged exposure
mid-times (capture.jsonl), so tau > 0 means the true instant of an image is
later on the IMU clock than logged. Expected: a few ms + half the rolling
shutter readout if the SensorTimestamp convention in capture_run_wave.py
(start of readout = end of exposure) is right; ~ +300 ms if
SensorTimestamp is in fact the start of the exposure.

    python imu_time_offset.py runs/<run> [runs/<run> ...] [--end-min 75]
"""

import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation

import calibrate_rcal as cr

GRID_HZ = 200.0
MAX_DT = 1.5           # frame pairs further apart are not used
BLOCK_S = 600.0        # one gyro offset per block (it drifts with temperature)
GUARD_DEG_S = 10.0     # instrument handled by hand
TAUS = np.arange(-0.6, 0.6001, 0.005)


def jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load(R, end_min):
    log = cr.pick_solve_log(R, None)
    meta, att = None, {}
    for d in jsonl(log):
        if d.get("type") == "meta":
            meta = d
        elif d.get("type") == "frame" and (d.get("solution") or {}).get("RA") is not None:
            s = d["solution"]
            att[d["frame_id"]] = cr.attitude_matrix(s["RA"], s["Dec"], s["Roll"])
    t_mid = {d["frame_id"]: d["t_exposure_mid"]
             for d in jsonl(os.path.join(R, "capture.jsonl")) if "frame_id" in d}
    imu = [(d["t"], d["gyro"]) for d in jsonl(os.path.join(R, "imu.jsonl")) if d.get("gyro")]
    ti = np.array([x[0] for x in imu])
    w = np.array([x[1] for x in imu]) @ np.array(meta["M_case"]).T     # camera frame
    ids = sorted(f for f in att if f in t_mid)
    t0 = t_mid[ids[0]]
    if end_min:
        ids = [f for f in ids if t_mid[f] < t0 + 60 * end_min]
    return ids, att, t_mid, ti, w


def pairs(ids, att, t_mid, ti, w):
    """Star rotation vectors (camera frame, rad) of consecutive frame pairs."""
    P = []
    for a, b in zip(ids[:-1], ids[1:]):
        dt = t_mid[b] - t_mid[a]
        if not 0.2 < dt < MAX_DT:
            continue
        k = (ti > t_mid[a] - 1) & (ti < t_mid[b] + 1)
        if not k.any() or np.degrees(np.abs(w[k]).max()) > GUARD_DEG_S:
            continue
        rv = Rotation.from_matrix(att[b] @ att[a].T).as_rotvec()
        if np.degrees(np.linalg.norm(rv)) > 3:          # a false solve in the pair
            continue
        P.append((t_mid[a], t_mid[b], rv))
    return P


def integral(ti, w):
    """Cumulative gyro integral on a uniform grid (camera frame, rad)."""
    tg = np.arange(ti[0], ti[-1], 1 / GRID_HZ)
    wg = np.stack([np.interp(tg, ti, w[:, k]) for k in range(3)], -1)
    C = np.vstack([np.zeros((1, 3)), np.cumsum(0.5 * (wg[1:] + wg[:-1]) / GRID_HZ, 0)])
    return tg, C


def fit(P, tg, C, tau, keep=None):
    """Least squares for scale s and per-block offsets b at shift tau.
    Returns rms residual (deg), s, residual norms per pair (deg)."""
    ta = np.array([p[0] for p in P])
    tb = np.array([p[1] for p in P])
    y = np.array([p[2] for p in P])
    G = (np.stack([np.interp(tb + tau, tg, C[:, k]) for k in range(3)], -1)
         - np.stack([np.interp(ta + tau, tg, C[:, k]) for k in range(3)], -1))
    blk = ((ta - ta[0]) // BLOCK_S).astype(int)
    nb = blk.max() + 1
    if keep is None:
        keep = np.ones(len(P), bool)
    # unknowns: s, then 3 offsets per block; rows: 3 per pair
    X = np.zeros((len(P), 3, 1 + 3 * nb))
    X[:, :, 0] = G
    for k in range(3):
        X[np.arange(len(P)), k, 1 + 3 * blk + k] = -(tb - ta)
    p = np.linalg.lstsq(X[keep].reshape(-1, X.shape[2]), y[keep].ravel(), rcond=None)[0]
    res = np.linalg.norm(y - (X @ p), axis=1)
    return np.degrees(np.sqrt(np.mean(res[keep] ** 2))), p[0], np.degrees(res)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--end-min", type=float, default=None)
    args = ap.parse_args()
    for R in args.runs:
        ids, att, t_mid, ti, w = load(R, args.end_min)
        P = pairs(ids, att, t_mid, ti, w)
        tg, C = integral(ti, w)
        name = os.path.basename(R.rstrip("/"))
        if len(P) < 100:
            print(f"{name}: only {len(P)} frame pairs")
            continue
        # trim bad pairs (false solves) at a first guess, then scan tau
        keep = np.ones(len(P), bool)
        for _ in range(3):
            _, _, res = fit(P, tg, C, 0.0, keep)
            keep = res < 3 * np.median(res[keep])
        scan = np.array([fit(P, tg, C, tau, keep)[0] for tau in TAUS])
        i = int(np.argmin(scan))
        # parabola through the minimum for sub-step precision
        if 0 < i < len(TAUS) - 1:
            a, b, c = scan[i - 1:i + 2]
            tau = TAUS[i] + 0.5 * (a - c) / (a - 2 * b + c) * (TAUS[1] - TAUS[0])
        else:
            tau = TAUS[i]
        rms, s, res = fit(P, tg, C, tau, keep)
        # precision: split into 4 time quarters
        q = np.array_split(np.arange(len(P)), 4)
        taus_q = []
        for idx in q:
            kq = np.zeros(len(P), bool)
            kq[idx] = True
            kq &= keep
            sc = [fit(P, tg, C, t_, kq)[0] for t_ in TAUS]
            taus_q.append(TAUS[int(np.argmin(sc))])
        rate = np.degrees(np.median([np.linalg.norm(p[2]) / (p[1] - p[0]) for p in P]))
        print(f"{name}: {keep.sum()}/{len(P)} pairs, median rate {rate:.2f} deg/s, gyro scale {s:+.4f}")
        print(f"   tau = {tau * 1000:+6.1f} ms   (quarters: "
              f"{', '.join(f'{x * 1000:+.0f}' for x in taus_q)} ms)   "
              f"rms residual {rms:.4f} deg")
        print("   rms residual (deg) vs tau: " + "  ".join(
            f"{t_ * 1000:+.0f} ms {fit(P, tg, C, t_, keep)[0]:.4f}"
            for t_ in (-0.3, -0.1, -0.03, 0.0, 0.03, 0.1, 0.3)))


if __name__ == "__main__":
    main()
