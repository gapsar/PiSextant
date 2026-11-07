"""Step 6: strong classical baselines on the synthetic TEST split (guide step 6).

Written by Claude, 29 Oct 2026. Run from the centroid_cnn folder (needs
tetra3 on the path: `source ../../dev_env.sh` or the installed package):
    python src/baselines.py --name v2 [--n 10000]
Results: reports/step6/baselines.txt (table, src/metrics.py format),
baselines.json, estimates.npz (per-sample estimates, for step 8).

Every method sees only what the CNN sees: channel 1 (noise units, background
removed) and, for the gyro methods, channels 2 and 3 (MEASURED path). The
label is the mid-time position (guide section 2), so a light-centre method is
biased on uneven trails by design; the "+gyro" lines remove the measured
light-centre offset (channel-2 first moment) from the estimate.

  com           centre of mass of the blob above 3 sigma that holds the most
                light within 6 px of the centre (fallback: 7 x 7 window)
  com+gyro      com - (channel-2 first moment - 16.5)
  centre        the patch centre itself (reference: what the patch position
                alone tells; added after the first run)
  gauss         elliptical Gaussian + background, least squares, pixel centres,
                inside the window (COM blob extent + 4 px) around the COM
                (window added after the first run: neighbours took the fit)
  gauss+gyro    gauss - (channel-2 first moment - 16.5)
  moffat        elliptical Moffat (beta free 1-10) + background
  tetra3        get_centroids_from_image(ch1, bg_sub_mode=None, image_th=3,
                min_area=3, max_area=1000), centroid nearest to the centre
  gyro_fit      B + A * sum_k w_k G_sigma(x - pos - p_k): Gaussian PSF (sigma
                free) convolved with the channel-2 dwell points p_k (mid-time
                point at the patch centre); pos estimates the label directly;
                window 6 px around the measured path
A method that finds nothing returns the patch centre (counted as an error).

PREDICTIONS (written before the first run):
  P1 still stars (trail < 1 px), SNR >= 20: gauss and moffat beat com and
     tetra3 in rms by >= 20%; gyro_fit equals gauss within 10%.
  P2 trails > 3 px: com, gauss, tetra3 have a bias ALONG the trail (light
     centre vs mid-time) and a larger rms; com+gyro and gauss+gyro remove
     most of the bias; gyro_fit has the lowest rms of all.
  P3 SNR < 5: every method fails (|e| > 2 px) in >= 20% of the samples.
  P4 gyro_fit degrades with the gyro error: rms in the bin
     meas_err > 1 px at least 2x the bin < 0.1 px.
  P5 neighbours < 8 px: all methods several times worse than without.
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import ndimage, optimize

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import metrics  # noqa: E402

OUT = HERE / "reports" / "step6"
RR, CC = np.mgrid[0:32, 0:32] + 0.5          # pixel centres, corner coords
CEN = 16.5


def com(ch1):
    lab, n = ndimage.label(ch1 > 3)
    best, bs = None, 0.0
    for j in range(1, n + 1):
        m = lab == j
        w = ch1[m]
        c = np.array([np.sum(w * RR[m]), np.sum(w * CC[m])]) / w.sum()
        if np.hypot(*(c - CEN)) < 6 and w.sum() > bs:
            best, bs = c, w.sum()
            ext = np.hypot(RR[m] - c[0], CC[m] - c[1]).max()
    if best is None:
        win = ch1[13:20, 13:20].clip(0)
        if win.sum() <= 0:
            return np.array([CEN, CEN]), False, 3.0
        return np.array([np.sum(win * RR[13:20, 13:20]), np.sum(win * CC[13:20, 13:20])]) / win.sum(), False, 3.0
    return best, True, ext


def ell(x, r, c, s1, s2, th):
    dr, dc = RR - r, CC - c
    u = dr * np.sin(th) + dc * np.cos(th)
    v = dr * np.cos(th) - dc * np.sin(th)
    return (u / s1) ** 2 + (v / s2) ** 2


def fit_profile(ch1, start, kind, rw):
    """Fit inside the window |x - start| <= rw (neighbours outside are ignored)."""
    win = (np.hypot(RR - start[0], CC - start[1]) <= rw).ravel()
    y = ch1.ravel()[win]

    def res(x):
        q = ell(None, x[0], x[1], x[2], x[3], x[4])
        m = np.exp(-0.5 * q) if kind == "gauss" else (1 + q) ** (-x[5])
        X = np.stack([m.ravel()[win], np.ones(y.size)], 1)
        ab = np.linalg.lstsq(X, y, rcond=None)[0]
        return y - X @ ab

    x0 = [start[0], start[1], 2.0, 2.0, 0.0] + ([3.0] if kind == "moffat" else [])
    lo = [0, 0, 0.3, 0.3, -np.pi] + ([1.0] if kind == "moffat" else [])
    hi = [32, 32, 15, 15, np.pi] + ([10.0] if kind == "moffat" else [])
    try:
        r = optimize.least_squares(res, x0, bounds=(lo, hi), x_scale="jac", max_nfev=200)
        return r.x[:2], r.success
    except Exception:
        return np.array(start, float), False


def gyro_fit(ch1, ch2, start):
    """Fit inside 6 px of the measured path placed at the start position."""
    k = np.nonzero(ch2 > 1e-4)
    w = ch2[k] / ch2[k].sum()
    pts = np.stack([RR[k] - CEN, CC[k] - CEN], 1)          # dwell points, mid-time at 0
    rw = 6 + np.hypot(pts[:, 0], pts[:, 1]).max()
    win = (np.hypot(RR - start[0], CC - start[1]) <= rw).ravel()
    y = ch1.ravel()[win]
    rr, cc = RR.ravel()[win], CC.ravel()[win]

    def res(x):
        dr = rr[:, None] - x[0] - pts[None, :, 0]
        dc = cc[:, None] - x[1] - pts[None, :, 1]
        m = (np.exp(-0.5 * (dr ** 2 + dc ** 2) / x[2] ** 2) * w[None]).sum(1)
        X = np.stack([m, np.ones(y.size)], 1)
        ab = np.linalg.lstsq(X, y, rcond=None)[0]
        return y - X @ ab

    try:
        r = optimize.least_squares(res, [start[0], start[1], 2.0], bounds=([0, 0, 0.3], [32, 32, 15]),
                                   x_scale="jac", max_nfev=200)
        return r.x[:2], r.success
    except Exception:
        return np.array(start, float), False


def tetra3_centroid(ch1):
    import tetra3
    try:
        cents = tetra3.get_centroids_from_image(ch1.astype(np.float32), bg_sub_mode=None, image_th=3.0,
                                                min_area=3, max_area=1000)
    except Exception:
        cents = []
    if len(cents) == 0:
        return np.array([CEN, CEN]), False
    cents = np.asarray(cents)
    return cents[np.argmin(np.hypot(cents[:, 0] - CEN, cents[:, 1] - CEN))], True


def one(args):
    X = args.astype(float)
    ch1, ch2 = X[0], X[1]
    m1 = np.array([np.sum(ch2 * RR), np.sum(ch2 * CC)]) - CEN       # measured light - mid-time
    c, ok_c, ext = com(ch1)
    g, ok_g = fit_profile(ch1, c, "gauss", ext + 4)
    mo, ok_m = fit_profile(ch1, c, "moffat", ext + 4)
    t, ok_t = tetra3_centroid(ch1)
    gf, ok_f = gyro_fit(ch1, ch2, c - m1)
    est = {"centre": np.array([CEN, CEN]), "com": c, "com+gyro": c - m1, "gauss": g, "gauss+gyro": g - m1, "moffat": mo,
           "tetra3": t, "gyro_fit": gf}
    ok = {"com": ok_c, "gauss": ok_g, "moffat": ok_m, "tetra3": ok_t, "gyro_fit": ok_f}
    return {k: v - CEN for k, v in est.items()}, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    import check_step5
    X, y, P = check_step5.load_split(HERE / "datasets" / args.name, "test")
    X, y, P = X[:args.n], y[:args.n], {k: v[:args.n] for k, v in P.items()}
    with Pool(args.workers) as pool:
        res = pool.map(one, list(X), chunksize=50)
    methods = list(res[0][0])
    est = {m: np.array([r[0][m] for r in res]) for m in methods}
    okf = {m: np.array([r[1][m] for r in res]) for m in res[0][1]}
    errors = {m: est[m] - y for m in methods}
    inst = (P["psf_inst"] > 0) & (P["bg_inst"] > 0) & (P["tone_inst"] > 0) & (P["motion_inst"] > 0)
    subsets = {"this instrument": inst, "bad (artefact)": P["bad"] > 0,
               "still, snr>=20": (P["centre_len_px"] < 1) & (P["snr"] >= 20),
               "trail>3, snr>=20": (P["centre_len_px"] > 3) & (P["snr"] >= 20)}
    tab, lines = metrics.table(errors, P, X, subsets)
    OUT.mkdir(parents=True, exist_ok=True)
    head = [f"step 6 baselines, dataset {args.name} test, n {len(y)}; fail = |e| > {metrics.FAIL_PX} px",
            "method found nothing (returned centre): " + ", ".join(f"{m} {np.mean(~v):.3f}" for m, v in okf.items()), ""]
    open(OUT / "baselines.txt", "w").write("\n".join(head + lines) + "\n")
    json.dump(tab, open(OUT / "baselines.json", "w"), indent=1)
    np.savez(OUT / "estimates.npz", y=y, **{m.replace("+", "_"): est[m] for m in methods})
    print("\n".join(head + lines[:1 + 2 * len(methods)]))


if __name__ == "__main__":
    main()
