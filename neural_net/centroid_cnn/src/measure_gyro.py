"""Step 4 part 2: real exposure-window motion, gyro noise, gyro sampling.

Written by Claude, 27 Oct 2026. Run from the centroid_cnn folder:
    python src/measure_gyro.py

Development runs 13/15/16 Oct, plus 20 Oct before minute 75 (rough sea,
guide section 3). Every exposure in capture.jsonl (saved image or not).

Per exposure window (offset = mean over +-150 s, camera frame):
  - max and mean rate, deg/s;
  - trail path length of a star at the frame centre, px
    (f * integral of the rate perpendicular to the boresight);
  - extra path at r = 1100 px from the boresight (roll) rate;
  - curvature: max distance of the centre path from its chord, px;
  - speed ratio: max / min speed over the 300 ms (uneven speed).
Gyro noise (guide 4.2.5): in calm windows (max rate < 0.2 deg/s), the
scatter of w_i - (w_(i-1) + w_(i+1)) / 2, times sqrt(2/3) (white noise
std per sample; a linear slow motion cancels).
Sampling (guide 4.2.6): interval quantiles and the gap fraction.

PREDICTIONS (before the run):
  G1 centre trail length: median 1-3 px on 13/15/16 Oct, p99 10-25 px;
     20 Oct longer (rough sea).
  G2 most trails are nearly straight: curvature p90 < 0.5 px; long trails
     (> 10 px) more curved in absolute px.
  G3 gyro white noise 0.005-0.02 deg/s per sample per axis (ISM330DHCX
     datasheet density ~ 0.005 deg/s/sqrt(Hz) times the filter bandwidth;
     a rough guess).
  G4 interval median 12.6-12.7 ms, p99 ~35 ms, gaps > 50 ms < 0.5%.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE.parents[1] / "runs"
OUT = HERE / "reports" / "step4"
NIGHTS = {"2026-10-13_16-14-01": None, "2026-10-15_16-00-55": None,
          "2026-10-16_16-05-50": None, "2026-10-20_15-36-33": 75}
F_PX = 1701.0
R_EDGE = 1100.0
DT = 0.001


def jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def one_night(name, end_min):
    run = RUNS / name
    log = sorted((run / "solves").glob("*.jsonl"))[0]
    with open(log) as fh:                        # first line = meta with M_case
        m_case = np.array(json.loads(fh.readline())["M_case"])
    t, g = [], []
    with open(run / "imu.jsonl") as fh:
        for line in fh:
            if '"gyro"' in line:
                d = json.loads(line)
                if d.get("gyro"):
                    t.append(d["t"])
                    g.append(d["gyro"])
    t = np.array(t)
    o = np.argsort(t)
    t, w = t[o], (np.array(g) @ m_case.T)[o]
    cs = np.vstack([np.zeros(3), np.cumsum(w, 0)])
    cap = jsonl(run / "capture.jsonl")
    t0 = cap[0]["t_exposure_mid"]
    rows, noise = [], []
    for c in cap:
        tm = c.get("t_exposure_mid")
        if tm is None or (end_min and tm > t0 + 60 * end_min):
            continue
        a, b = np.searchsorted(t, [tm - 150, tm + 150])
        i0, i1 = np.searchsorted(t, [tm - 0.15, tm + 0.15])
        if b - a < 1000 or i1 - i0 < 15:
            continue
        off = (cs[b] - cs[a]) / (b - a)
        k0, k1 = max(i0 - 2, 0), i1 + 2
        tt, ww = t[k0:k1], w[k0:k1] - off
        rate = np.degrees(np.linalg.norm(w[i0:i1] - off, axis=1))
        if rate.max() > 10:                      # handled by hand: exclude
            continue
        tg = np.arange(tm - 0.15, tm + 0.15, DT)
        wg = np.stack([np.interp(tg, tt, ww[:, j]) for j in range(3)], -1)
        th = np.cumsum(wg * DT, 0)
        p = F_PX * th[:, 1:]                     # centre star, px (sign irrelevant here)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        length = seg.sum()
        roll = R_EDGE * np.abs(wg[:, 0]).sum() * DT
        u = (p[-1] - p[0]) / max(np.linalg.norm(p[-1] - p[0]), 1e-9)
        curve = np.abs((p[:, 0] - p[0, 0]) * u[1] - (p[:, 1] - p[0, 1]) * u[0]).max()
        sp = np.convolve(seg, np.ones(20) / 20, "valid")
        rows.append([rate.max(), rate.mean(), length, roll, curve,
                     sp.max() / max(sp.min(), 1e-6)])
        if rate.max() < 0.2:
            x = w[i0:i1]
            noise.append((x[1:-1] - 0.5 * (x[:-2] + x[2:])) * np.sqrt(2 / 3))
    rows = np.array(rows)
    dtt = np.diff(t)
    noise = np.degrees(np.concatenate(noise)) if noise else np.zeros((0, 3))
    q = lambda x, p: np.percentile(x, p).round(3).tolist()
    res = {
        "windows": len(rows),
        "rate_max_deg_s_p10_p50_p90_p99": q(rows[:, 0], [10, 50, 90, 99]),
        "centre_length_px_p10_p50_p90_p99": q(rows[:, 2], [10, 50, 90, 99]),
        "roll_extra_at_r1100_px_p50_p90_p99": q(rows[:, 3], [50, 90, 99]),
        "curvature_px_p50_p90_p99": q(rows[:, 4], [50, 90, 99]),
        "curvature_px_p50_p90_long_gt_10px": q(rows[rows[:, 2] > 10, 4], [50, 90]) if (rows[:, 2] > 10).sum() > 5 else None,
        "speed_ratio_p50_p90_long_gt_5px": q(rows[rows[:, 2] > 5, 5], [50, 90]) if (rows[:, 2] > 5).sum() > 5 else None,
        "frac_length_gt": {str(x): float(np.mean(rows[:, 2] > x)) for x in [1, 3, 5, 10, 20, 30]},
        "gyro_white_noise_deg_s_per_axis": np.std(noise, axis=0).round(4).tolist() if len(noise) else None,
        "noise_samples": int(len(noise)),
        "interval_ms_p1_p50_p99": (np.percentile(dtt, [1, 50, 99]) * 1e3).round(2).tolist(),
        "gap_frac_gt_50ms": float(np.mean(dtt > 0.05)),
        "gap_frac_gt_100ms": float(np.mean(dtt > 0.1)),
    }
    return res, rows[:, 2], dtt


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out, lengths = {}, {}
    for name, end in NIGHTS.items():
        res, L, dtt = one_night(name, end)
        out[name] = res
        lengths[name] = L
        print(name, json.dumps(res))
    json.dump(out, open(OUT / "gyro.json", "w"), indent=1)
    np.savez(OUT / "window_lengths.npz", **{k[:10]: v for k, v in lengths.items()})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-1, np.log10(60), 50)
    for k, v in lengths.items():
        ax.hist(v, bins, histtype="step", density=True, label=k[:10])
    ax.set_xscale("log")
    ax.set_xlabel("centre trail length during the exposure, px")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "trail_lengths.png", dpi=90)


if __name__ == "__main__":
    main()
