"""Step 7, last part: hybrid CNN + gyro_fit for bright, clean stars.

Written by Claude, 30 Oct 2026. Usage (from the centroid_cnn folder):
    python src/hybrid.py            # choose the rule on VAL, write configs/hybrid.json
The CNN (checkpoints/step7_full_v3_e120.pt) gives the position mu and sigma.
gyro_fit (src/baselines.py, the best classical method) then starts AT the CNN
position. Rule: use the gyro_fit result if sigma_geo = sqrt(s_row s_col) <=
sigma_max AND |gyro_fit - mu| <= max_shift_px; else keep the CNN. Both inputs
are observable at run time (no true SNR). Choice: lowest val mean Huber
(beta 1 px, sum of the axes), the step-7 criterion. The test split is not used.

CHANGED after run 1 (30 Oct): the sigma rule did nothing (val Huber 0.3838 =
CNN; reports/step7/hybrid_sigma_rule.txt): CNN sigma does not separate SNR
300+ from 50-300, but gyro_fit from the CNN start is better at 300+ (median
0.122 vs 0.172). Added feature "peak" = max of ch1 (noise units) within 2 px
of the CNN position; the rule is "feature passes AND shift <= max_shift";
the choice (lowest val Huber) is over both features.

Predictions BEFORE the run:
- gyro_fit started at the CNN position is better than step-6 gyro_fit (started
  at COM - light offset) in failures, equal in core precision.
- best rule refines about 10-25 % of the stars (sigma_max about 0.15-0.25 px),
  max_shift about 0.5-1 px.
- hybrid val: SNR 300+ median 0.172 -> about 0.13 (= gyro_fit), SNR 50-300
  0.245 -> about 0.235; SNR < 20 unchanged (rule does not fire).
- gyro_fit costs about 2-5 ms per star on the laptop (one core).
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
import baselines as B  # noqa: E402
import check_step5  # noqa: E402
import metrics  # noqa: E402
import train_cnn as T  # noqa: E402

MODEL = "step7_full_v3_e120"
CEN = 16.5


def cnn(X, name=MODEL):
    ck = torch.load(HERE / "checkpoints" / f"{name}.pt", map_location="cuda")
    model = T.Net(ck["in_ch"], ck["cfg"]["ch1_k"], ck["cfg"]["ch2_scale"], ck.get("arch", "a")).cuda()
    model.load_state_dict(ck["state"])
    mu, ls = T.predict(model, torch.from_numpy(X).cuda())
    return mu.cpu().numpy(), np.exp(ls.cpu().numpy())


def _refine_one(a):
    x, start = a
    x = x.astype(float)
    return B.gyro_fit(x[0], x[1], start + CEN)[0] - CEN


def refine(X, mu, workers=16):
    """gyro_fit started at the CNN estimate, for every sample (rows of X)."""
    with Pool(workers) as pool:
        return np.array(pool.map(_refine_one, list(zip(list(X), list(mu))), chunksize=50))


def peak(X, mu):
    """Max of ch1 within 2 px (Chebyshev) of the CNN position, noise units."""
    out = np.empty(len(X))
    for i, (x, m) in enumerate(zip(X, mu)):
        r, c = (np.floor(m + CEN)).astype(int).clip(2, 29)
        out[i] = x[0, r - 2:r + 3, c - 2:c + 3].astype(float).max()
    return out


def apply(rule, mu, sig, gf, pk):
    if rule["feature"] == "sigma":
        ok = np.sqrt(sig[:, 0] * sig[:, 1]) <= rule["threshold"]
    else:
        ok = pk >= rule["threshold"]
    use = ok & (np.linalg.norm(gf - mu, axis=1) <= rule["max_shift_px"])
    return np.where(use[:, None], gf, mu), use


def huber(e):
    a = np.abs(e)
    return np.where(a < 1, 0.5 * a ** 2, a - 0.5).sum(1).mean()


def main():
    X, y, P = check_step5.load_split(HERE / "datasets" / "v2", "val")
    mu, sig = cnn(X)
    t0 = time.time()
    gf = refine(X, mu)
    dt = time.time() - t0
    one = time.time()
    for i in range(200):
        _refine_one((X[i], mu[i]))
    ms = (time.time() - one) / 200 * 1e3
    sg = np.sqrt(sig[:, 0] * sig[:, 1])
    pk = peak(X, mu)
    L = [f"hybrid on v2 val (n {len(y)}), CNN {MODEL}; gyro_fit from the CNN start: {dt:.0f} s on 16 workers, "
         f"{ms:.1f} ms per star on one laptop core", "",
         f"CNN only: val Huber {huber(mu - y):.4f}; gyro_fit from CNN start for ALL stars: {huber(gf - y):.4f}", "",
         "grid (sigma: use if sigma_geo <= threshold px; peak: use if ch1 peak >= threshold; best 12 by val Huber):"]
    res = []
    grids = {"sigma": np.quantile(sg, np.arange(0.02, 0.62, 0.04)),
             "peak": np.quantile(pk, np.arange(0.40, 0.995, 0.03))}
    for feat, ths in grids.items():
        for th in sorted(set(np.round(ths, 3))):
            for g0 in (0.25, 0.5, 1.0, 2.0, 99.0):
                r = {"feature": feat, "threshold": float(th), "max_shift_px": g0}
                est, use = apply(r, mu, sig, gf, pk)
                res.append((huber(est - y), use.mean(), r))
    res.sort(key=lambda t: t[0])
    for h, f, r in res[:12]:
        L.append(f"  {r['feature']:5s} {r['threshold']:8.3f}  max_shift {r['max_shift_px']:5.2f}  huber {h:.4f}  refined {f:.3f}")
    best = res[0][2]
    est, use = apply(best, mu, sig, gf, pk)
    best.update({"model": MODEL, "chosen_on": "v2 val, lowest mean Huber", "val_huber": res[0][0],
                 "val_refined_fraction": float(use.mean()), "gyro_fit_ms_per_star_laptop": ms})
    json.dump(best, open(HERE / "configs" / "hybrid.json", "w"), indent=1)
    L += ["", f"chosen: {json.dumps(best)}", ""]
    old = dict(np.load(HERE / "scratch" / "val_baselines.npz"))["gyro_fit"]
    errors = {"gyro_fit": old - y, "gyro_fit_cnnstart": gf - y, "cnn": mu - y, "hybrid": est - y}
    subsets = {"still, snr>=20": (P["centre_len_px"] < 1) & (P["snr"] >= 20),
               "trail>3, snr>=20": (P["centre_len_px"] > 3) & (P["snr"] >= 20)}
    _, lines = metrics.table(errors, P, X, subsets)
    L += lines
    for lo, hi in ((2, 20), (20, 50), (50, 300), (300, 1e9)):
        m = (P["snr"] >= lo) & (P["snr"] < hi)
        L.append(f"refined fraction snr {lo:g}-{hi:g}: {use[m].mean():.3f}")
    txt = "\n".join(L)
    (HERE / "reports" / "step7" / "hybrid.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
