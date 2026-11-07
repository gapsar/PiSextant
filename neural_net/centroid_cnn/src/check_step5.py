"""Step 5 checks of the dataset generator (guide step 5) + the step-3 star
eye test that needed the generator.

Written by Claude, 29 Oct 2026. Run from the centroid_cnn folder, after
    python src/generator.py --name v1
then
    python src/check_step5.py --name v1
Results: reports/step5/checks.txt and .json, params_hist.png, samples.png,
eye_test_stars.png (+ eye_test_key.json, NOT to be opened before the test).

Checks and PREDICTIONS (written before the first run):
  K1 every drawn parameter has a spread (no accidental constant) and its
     histogram has the intended shape (plot; the txt lists p1/p50/p99).
  K2 regenerating the first 500 training samples gives bit-identical X, y
     and parameters.
  K3 label consistency: clean, zero gyro error, noise-free samples; an
     INDEPENDENT renderer (direct sum of the pixel-integrated PSF over the
     path points, no fine grid, no FFT) fitted to the generator's linear
     star image with free position and flux recovers the label within
     0.01 px (max over 200 samples; super-Gaussian, Gaussian, Moffat, no coma).
  K4 channel 2 geometry: with zero gyro error, the first moment of channel 2
     minus (16.5, 16.5) equals the time mean of the true path within 0.002 px
     (samples whose path stays inside the patch).
  K5 sub-pixel phase of the label, (16.5 + y) mod 1, is uniform (chi2 test
     over 10 bins, p > 0.01) in the training set: no pixel-locking from the
     sampling.
  K6 the label spread: |y| p50 about 0.5 px, p99 up to about 2-6 px (the
     light-centre vs mid-time offset of long uneven trails).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import generator as G  # noqa: E402
import motion as M  # noqa: E402

OUT = HERE / "reports" / "step5"


def load_split(root, split):
    files = sorted((root / split).glob("shard_*.npz"))
    parts = [np.load(f) for f in files]
    keys = [k for k in parts[0].files if k not in ("X", "y")]
    return (np.concatenate([p["X"] for p in parts]), np.concatenate([p["y"] for p in parts]),
            {k: np.concatenate([p[k] for p in parts]) for k in keys})


def direct_render(psf, d, label, n=32, sub=5):
    """Independent renderer: sum over path points of the pixel-integrated PSF
    (sub x sub midpoints per pixel), no truncation, total = sum of the image."""
    o = (np.arange(sub) + 0.5) / sub
    yy, xx = np.mgrid[0:n, 0:n]
    th = np.radians(psf["angle"])
    img = np.zeros((n, n))
    for oy in o:
        for ox in o:
            for q in d:                             # every 1 ms path point
                dr = yy + oy - label[0] - q[0]
                dc = xx + ox - label[1] - q[1]
                u = dr * np.sin(th) + dc * np.cos(th)
                v = dr * np.cos(th) - dc * np.sin(th)
                qq = (u / psf["a1"]) ** 2 + (v / psf["a2"]) ** 2
                if psf["model"] == "moffat":
                    img += (1 + qq) ** (-psf["beta"])
                else:
                    img += np.exp(-0.5 * qq ** psf.get("p", 1.0))
    return img


def k3_one(i):
    X, y, P, ex = G.make_sample("check", i, zero_error=True, noise_free=True, clean=True)
    psf = ex["psf"]
    if psf["model"] == "ring" or psf["coma_frac"] > 0:
        return None
    star = ex["star"]
    d = ex["d_true"]

    def cost(pos):
        m = direct_render(psf, d, pos).ravel()
        a = np.dot(m, star.ravel()) / np.dot(m, m)
        return np.sum((star.ravel() - a * m) ** 2)

    lab = ex["label_local"]
    r = optimize.minimize(cost, lab + 0.05, method="Nelder-Mead",
                          options={"xatol": 1e-4, "fatol": 1e-12, "maxiter": 400})
    return float(np.linalg.norm(r.x - lab)), psf["model"], float(P["centre_len_px"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--k3", type=int, default=200)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    root = HERE / "datasets" / args.name
    lines, res = [], {}

    def say(t):
        print(t, flush=True)
        lines.append(t)

    X, y, P = load_split(root, "train")
    say(f"train: {len(y)} samples, {len(P)} parameters")
    # K1
    const = [k for k, v in P.items() if np.nanstd(v) == 0]
    res["K1_constant_params"] = const
    say(f"K1 constant parameters: {const if const else 'none'}")
    for k, v in P.items():
        say(f"   {k:20s} p1 {np.percentile(v, 1):10.4g}  p50 {np.percentile(v, 50):10.4g}  p99 {np.percentile(v, 99):10.4g}")
    # K2
    G.WINDOWS["train"] = G.load_split_windows("train")
    Xr, yr, Pr = G.gen_chunk(("train", 0, 500))
    same = bool(np.array_equal(Xr, X[:500]) and np.array_equal(yr, y[:500])
                and all(np.array_equal(Pr[k], P[k][:500]) for k in P))
    res["K2_bit_identical_500"] = same
    say(f"K2 regenerate train[0:500] bit-identical: {same}")
    # K5, K6
    ph = np.mod(16.5 + y, 1.0)
    for ax, name in enumerate(["row", "col"]):
        cnt = np.histogram(ph[:, ax], bins=10, range=(0, 1))[0]
        p = stats.chisquare(cnt).pvalue
        res[f"K5_phase_{name}_p"] = float(p)
        say(f"K5 label sub-pixel phase {name}: counts {cnt.tolist()}, chi2 p {p:.3f}")
    a = np.abs(y)
    res["K6_abs_y_p50_p99_max"] = [float(np.percentile(a, 50)), float(np.percentile(a, 99)), float(a.max())]
    say(f"K6 |y| p50 {np.percentile(a, 50):.2f}, p99 {np.percentile(a, 99):.2f}, max {a.max():.2f} px")
    # K3, K4 on 'check' windows (training time range, own seed)
    G.WINDOWS["check"] = G.load_split_windows("check", n_per_run=100)
    from multiprocessing import Pool
    with Pool(16) as pool:
        k3 = [r for r in pool.map(k3_one, range(args.k3)) if r is not None]
    e = np.array([r[0] for r in k3])
    res["K3"] = {"n": len(e), "max_px": float(e.max()), "p50_px": float(np.median(e)),
                 "by_model": {m: float(max(r[0] for r in k3 if r[1] == m)) for m in {r[1] for r in k3}}}
    say(f"K3 label recovery (independent renderer, n {len(e)}): max {e.max():.4f} px, median {np.median(e):.5f} px, "
        f"by model {res['K3']['by_model']}")
    k4 = []
    for i in range(300):
        Xc, yc, Pc, ex = G.make_sample("check", 1000 + i, zero_error=True, clean=True)
        if np.abs(ex["d_meas"]).max() > 14:
            continue
        ch2 = Xc[1].astype(float)
        rr, cc = np.mgrid[0:32, 0:32] + 0.5
        m1 = np.array([np.sum(ch2 * rr), np.sum(ch2 * cc)]) - 16.5
        k4.append(np.linalg.norm(m1 - ex["d_true"].mean(0)))
    res["K4_max_px"] = float(max(k4))
    say(f"K4 channel 2 first moment vs true-path time mean (n {len(k4)}): max {max(k4):.5f} px")
    json.dump(res, open(OUT / "checks.json", "w"), indent=1)
    open(OUT / "checks.txt", "w").write("\n".join(lines) + "\n")
    plots(X, y, P)


def plots(X, y, P):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(P)
    nc = 8
    nr = int(np.ceil(len(keys) / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(2.2 * nc, 1.8 * nr))
    for a, k in zip(ax.ravel(), keys):
        v = P[k]
        a.hist(v, 50, log=True)
        a.set_title(k, fontsize=7)
        a.tick_params(labelsize=5)
    for a in ax.ravel()[len(keys):]:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "params_hist.png", dpi=70)
    rng = np.random.default_rng(5)
    idx = rng.choice(len(y), 50, replace=False)
    fig, ax = plt.subplots(5, 10, figsize=(15, 8))
    for a, i in zip(ax.ravel(), idx):
        a.imshow(X[i, 0].astype(float), cmap="gray", vmin=-2, vmax=max(4, np.percentile(X[i, 0], 99.5)))
        rr, cc = np.nonzero(X[i, 1])
        a.scatter(cc, rr, s=1, c="r", alpha=0.3)
        a.plot(16.5 + y[i, 1] - 0.5, 16.5 + y[i, 0] - 0.5, "c+", ms=6)
        a.set_title(f"snr {P['snr'][i]:.0f} L {P['centre_len_px'][i]:.0f}", fontsize=6)
        a.axis("off")
    fig.suptitle("ch1 (gray), ch2 measured path (red), label (cyan +)")
    fig.tight_layout()
    fig.savefig(OUT / "samples.png", dpi=70)


if __name__ == "__main__":
    main()
