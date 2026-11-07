"""Step 3, part 1: measure the real sky background (guide section 5, step 3).

What it does, per development night (13, 15, 16 Oct):
1. Reads N_FRAMES frames spread over the night.
2. Masks the stars: a disc around every centroid that the solver logged,
   plus every compact blob of at least BLOB_MIN_PX connected pixels at or
   above BLOB_DN (an undetected star), grown by MASK_GROW_PX. Single noise
   pixels stay in: a mask on every bright pixel would cut the noise tail
   (25 Oct: the first version masked every pixel >= 10 DN, so real patches
   could never hold one, while 65% of synthetic patches did).
3. Measures the empty-sky pixels: the histogram of integer values (with
   the zero spike), the mean, the std, and the correlation between
   neighbouring pixels at lags 1-3 px (after removing the 32 x 32 tile
   mean, so the slow background gradient does not count as correlation).
4. Fits a background model to the night:
       z            = K * white noise (unit variance, correlated)
       linear value = mu + sigma * skew(z, s)
       output       = clip(round(linear value), 0, 255)
   K is a separable 5-tap kernel [b, a, 1, a, b] (per axis), normalized so
   that z has unit variance. skew(z, s) = (exp(s z) - m) / d is a
   standardized lognormal shape (mean 0, variance 1); s = 0 is the Gaussian
   limit. The real noise is skewed (a heavy right tail, from the ISP), so
   a Gaussian does not fit (see WORKLOG 25 Oct). The histogram of this
   model has an exact formula, so mu, sigma, s are fitted without
   simulation. a and b come from the lag-1 and lag-2 correlations, by
   simulation (the clip and the rounding lower the measured correlation).
   The fit is also done per radius bin, because the lens shading
   correction of the ISP makes the noise grow toward the corners.
5. Makes synthetic empty patches with the fitted model and compares them
   with real empty patches: histogram, mean, std, correlations, plus a
   mixed grid of 25 real and 25 synthetic patches with a separate key.

Outputs in reports/step3/: background.json, background_hist.png,
background_patches_mixed.png, background_patches_key.json.

Run from the centroid_cnn folder:
    python3 src/measure_background.py
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
from PIL import Image
from scipy.ndimage import binary_dilation, convolve1d, label
from scipy.optimize import least_squares
from scipy.special import ndtr  # standard normal CDF

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"
NIGHTS = ["2026-10-13_16-14-01", "2026-10-15_16-00-55", "2026-10-16_16-05-50"]
SEED = 20261025
N_FRAMES = 20          # frames per night, spread evenly over the run
BLOB_DN = 8            # blob pixels: at or above this value (noise sigma 1.4-2.0 DN)
BLOB_MIN_PX = 5        # a blob needs this many connected pixels (noise rarely makes 5)
MASK_GROW_PX = 4       # grow the blob mask by this many px
CENTROID_R_PX = 8      # mask radius around every logged centroid, px
TILE_PX = 32           # tile for the local mean (correlation) and for patches
HIST_MAX = 15          # histogram values 0..HIST_MAX; HIST_MAX holds ">= HIST_MAX"
SIM_PX = 512           # side of the simulated field for the kernel fit
RADIUS_EDGES = [0, 200, 400, 600, 800, 1000, 1400]  # px from the image centre


def load_centroids(run):
    """Logged centroids per image name, (row, col) px (CONVENTIONS.md).
    Input: run folder. Output: dict image name -> (N, 2) array."""
    out = {}
    for path in (run / "solves").glob("*.jsonl"):
        for line in open(path):
            rec = json.loads(line)
            if rec.get("type") == "frame" and rec.get("centroids"):
                out[Path(rec["image"]).name] = np.asarray(rec["centroids"], dtype=float)
    return out


def star_mask(img, centroids):
    """True where a star (or its wings) may be. Inputs: image, (N, 2)
    centroids or None. Output: bool array like img."""
    lab, n = label(img >= BLOB_DN)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    big = sizes >= BLOB_MIN_PX
    big[0] = False
    mask = binary_dilation(big[lab], iterations=MASK_GROW_PX)
    if centroids is not None and len(centroids):
        rr, cc = np.ogrid[-CENTROID_R_PX:CENTROID_R_PX + 1, -CENTROID_R_PX:CENTROID_R_PX + 1]
        disc = rr ** 2 + cc ** 2 <= CENTROID_R_PX ** 2
        h, w = img.shape
        for r, c in np.round(centroids).astype(int):
            r0, c0 = r - CENTROID_R_PX, c - CENTROID_R_PX
            if r0 < 0 or c0 < 0 or r0 + disc.shape[0] > h or c0 + disc.shape[1] > w:
                continue  # edge stars: the bright-pixel mask covers them
            mask[r0:r0 + disc.shape[0], c0:c0 + disc.shape[1]] |= disc
    return mask


def tile_residual(img, mask):
    """Image minus its 32 x 32 tile mean (unmasked pixels only), NaN where
    masked or where a tile has fewer than 200 empty pixels."""
    res = np.full(img.shape, np.nan)
    for a in range(0, img.shape[0] - TILE_PX + 1, TILE_PX):
        for b in range(0, img.shape[1] - TILE_PX + 1, TILE_PX):
            t, m = img[a:a + TILE_PX, b:b + TILE_PX], ~mask[a:a + TILE_PX, b:b + TILE_PX]
            if m.sum() >= 200:
                block = t - t[m].mean()
                block[~m] = np.nan
                res[a:a + TILE_PX, b:b + TILE_PX] = block
    return res


def lag_correlations(res):
    """Correlation of neighbouring pixels: lags 1-3 along rows (h) and
    columns (v), and the diagonal (1, 1). Input: residual with NaN. Output: dict."""
    def corr(dy, dx):
        a, b = res[:res.shape[0] - dy, :res.shape[1] - dx], res[dy:, dx:]
        ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    return {"h1": corr(0, 1), "h2": corr(0, 2), "h3": corr(0, 3),
            "v1": corr(1, 0), "v2": corr(2, 0), "v3": corr(3, 0), "d11": corr(1, 1)}


def histogram(values):
    """Fractions of the integer values 0..HIST_MAX (last bin = >= HIST_MAX)."""
    v = np.minimum(values.astype(int), HIST_MAX)
    return np.bincount(v, minlength=HIST_MAX + 1) / v.size


def skew_shape(z, s):
    """Standardized lognormal shape: mean 0, variance 1; z itself if s ~ 0.
    s may be a number or an array like z (radius-dependent skew)."""
    s = np.maximum(np.asarray(s, dtype=float), 1e-4)
    m, d = np.exp(s * s / 2), np.sqrt((np.exp(s * s) - 1) * np.exp(s * s))
    return (np.exp(s * z) - m) / d


def model_histogram(mu, sigma, s):
    """Exact fractions of clip(round(mu + sigma * skew(z, s)), 0, 255),
    z ~ N(0, 1), for the values 0..HIST_MAX. P(x < t) = Phi(ln(arg) / s)
    with arg = (t - mu) d / sigma + m (0 when arg <= 0)."""
    edges = np.arange(HIST_MAX) + 0.5  # rounding edges between k and k+1
    if s < 1e-4:
        cdf = ndtr((edges - mu) / sigma)
    else:
        m, d = math.exp(s * s / 2), math.sqrt((math.exp(s * s) - 1) * math.exp(s * s))
        arg = (edges - mu) * d / sigma + m
        cdf = np.where(arg > 0, ndtr(np.log(np.maximum(arg, 1e-300)) / s), 0.0)
    return np.diff(np.concatenate([[0.0], cdf, [1.0]]))


def fit_marginal(hist, n_pix):
    """Fit mu, sigma, s of the clipped rounded skewed model to a histogram.
    Multinomial weights. Output: (mu, sigma, s, chi2 per bin)."""
    def resid(p):
        m = model_histogram(p[0], abs(p[1]), abs(p[2]))
        return (m - hist) / np.sqrt(np.maximum(m, 1e-6) / n_pix)
    sol = least_squares(resid, [hist @ np.arange(HIST_MAX + 1), 1.5, 0.3])
    return float(sol.x[0]), float(abs(sol.x[1])), float(abs(sol.x[2])), float(np.mean(sol.fun ** 2))


def kernel_1d(a, b):
    """Per-axis kernel [b, a, 1, a, b] scaled so that the separable 2-D
    kernel keeps the variance of white noise (sum of squares 1 per axis)."""
    k = np.array([b, a, 1.0, a, b])
    return k / math.sqrt(np.sum(k ** 2))


def correlated_field(shape, a, b, rng=None, white=None):
    """Unit-variance noise field with the separable kernel. Pass white to
    reuse the same white noise (common random numbers in the fit)."""
    if white is None:
        white = rng.standard_normal(shape)
    k = kernel_1d(a, b)
    return convolve1d(convolve1d(white, k, axis=0, mode="wrap"), k, axis=1, mode="wrap")


def render_background(mu, sigma, s, a, b, shape, rng=None, white=None):
    """One synthetic background field, quantized and clipped like the camera."""
    lin = mu + sigma * skew_shape(correlated_field(shape, a, b, rng, white), s)
    return np.clip(np.round(lin), 0, 255)


def fit_kernel(mu, sigma, s, target, rng):
    """Find a, b so that the SIMULATED output (clipped, rounded) has the
    measured lag-1 and lag-2 correlations (mean of h and v). Same white
    noise for every trial, so the objective is smooth enough for
    least_squares. Output: (a, b, simulated correlations)."""
    white = rng.standard_normal((SIM_PX, SIM_PX))
    t1 = 0.5 * (target["h1"] + target["v1"])
    t2 = 0.5 * (target["h2"] + target["v2"])

    def sim_corr(p):
        img = render_background(mu, sigma, s, p[0], p[1], white.shape, white=white)
        return lag_correlations(img - img.mean())

    def resid(p):
        c = sim_corr(p)
        return [0.5 * (c["h1"] + c["v1"]) - t1, 0.5 * (c["h2"] + c["v2"]) - t2]

    sol = least_squares(resid, [0.3, -0.05], diff_step=0.05, bounds=([0.0, -0.5], [1.0, 0.5]))
    return float(sol.x[0]), float(sol.x[1]), sim_corr(sol.x)


def empty_patches(img, mask, n, rng):
    """Up to n random 32 x 32 patches with no masked pixel."""
    out, h, w = [], img.shape[0], img.shape[1]
    for _ in range(50 * n):
        r, c = rng.integers(0, h - TILE_PX), rng.integers(0, w - TILE_PX)
        if not mask[r:r + TILE_PX, c:c + TILE_PX].any():
            out.append(img[r:r + TILE_PX, c:c + TILE_PX].copy())
            if len(out) == n:
                break
    return out


def measure_night(name, rng):
    """All real measurements of one night. Output: dict + list of real patches."""
    run = RUNS_DIR / name
    frames = sorted((run / "images").glob("frame_*.png"))
    pick = [frames[i] for i in np.linspace(0, len(frames) - 1, N_FRAMES).astype(int)]
    cents = load_centroids(run)
    pooled, per_frame, corrs, patches = [], [], [], []
    radial_vals = {edge: [] for edge in zip(RADIUS_EDGES[:-1], RADIUS_EDGES[1:])}
    for path in pick:
        img = np.asarray(Image.open(path), dtype=np.float64)
        mask = star_mask(img, cents.get(path.name))
        vals = img[~mask]
        pooled.append(vals)
        rad = np.hypot(*np.ogrid[-img.shape[0] / 2 + 0.5:img.shape[0] / 2,
                                 -img.shape[1] / 2 + 0.5:img.shape[1] / 2])
        for lo, hi in radial_vals:
            radial_vals[(lo, hi)].append(img[~mask & (rad >= lo) & (rad < hi)])
        per_frame.append({"frame": path.name, "mean": float(vals.mean()), "std": float(vals.std()),
                          "zero_fraction": float((vals == 0).mean()),
                          "masked_fraction": float(mask.mean()),
                          "centroids_logged": int(len(cents.get(path.name, [])))})
        corrs.append(lag_correlations(tile_residual(img, mask)))
        patches += empty_patches(img, mask, 15, rng)
    vals = np.concatenate(pooled)
    corr = {k: float(np.mean([c[k] for c in corrs])) for k in corrs[0]}
    return {"night": name, "n_frames": len(pick), "n_pixels": int(vals.size),
            "mean": float(vals.mean()), "std": float(vals.std()),
            "hist": histogram(vals).tolist(), "corr": corr, "frames": per_frame,
            "radial": [((lo, hi), histogram(np.concatenate(v)).tolist(), int(sum(x.size for x in v)))
                       for (lo, hi), v in radial_vals.items() if sum(x.size for x in v) > 10000]}, patches


def synthetic_stats(mu, sigma, s, a, b, rng):
    """The same statistics on a synthetic field (no mask, no gradient)."""
    img = render_background(mu, sigma, s, a, b, (SIM_PX, SIM_PX), rng)
    return {"mean": float(img.mean()), "std": float(img.std()),
            "hist": histogram(img.ravel()).tolist(), "corr": lag_correlations(img - img.mean())}


def plot_hist(results, out):
    """Real vs synthetic histograms per night (log y, so the tail shows)."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    x = np.arange(HIST_MAX + 1)
    for ax, r in zip(axes, results):
        ax.bar(x - 0.2, r["real"]["hist"], 0.4, label="real empty sky")
        ax.bar(x + 0.2, r["synthetic"]["hist"], 0.4, label="synthetic (fitted model)")
        ax.set_yscale("log")
        ax.set_title(f"{r['real']['night'][:10]}: mu {r['model']['mu']:.2f}, sigma "
                     f"{r['model']['sigma']:.2f} DN, s {r['model']['s']:.2f}", fontsize=9)
        ax.set_xlabel(f"DN (last bin = >= {HIST_MAX})")
    axes[0].set_ylabel("fraction of pixels")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "background_hist.png", dpi=110)
    plt.close(fig)


def plot_mixed(real, synth, rng, out):
    """5 x 10 grid: 25 real + 25 synthetic patches in random order. The key
    (which tile is real) goes to a separate JSON file, as the guide asks."""
    pick_r = rng.choice(len(real), 25, replace=False)  # random over all 20 frames
    pick_s = rng.choice(len(synth), 25, replace=False)
    tiles = [("real", real[i]) for i in pick_r] + [("synthetic", synth[i]) for i in pick_s]
    order = rng.permutation(len(tiles))
    fig, axes = plt.subplots(5, 10, figsize=(15, 7.8))
    key = {}
    for slot, (ax, k) in enumerate(zip(axes.ravel(), order)):
        kind, patch = tiles[k]
        ax.imshow(patch, vmin=0, vmax=8, cmap="gray")
        ax.set_title(str(slot), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        key[str(slot)] = kind
    fig.suptitle("Empty-sky patches, 15 Oct: which are real? (display 0-8 DN; key in "
                 "background_patches_key.json)")
    fig.tight_layout()
    fig.savefig(out / "background_patches_mixed.png", dpi=100)
    plt.close(fig)
    with open(out / "background_patches_key.json", "w") as f:
        json.dump(key, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "reports/step3"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results, patches_15 = [], None
    for name in NIGHTS:
        real, patches = measure_night(name, rng)
        mu, sigma, s, chi2 = fit_marginal(np.array(real["hist"]), real["n_pixels"])
        a, b, sim_c = fit_kernel(mu, sigma, s, real["corr"], rng)
        synth = synthetic_stats(mu, sigma, s, a, b, rng)
        radial = []
        for (lo, hi), hist, npix in real["radial"]:
            rm, rs, rk, rc = fit_marginal(np.array(hist), npix)
            radial.append({"r_lo": lo, "r_hi": hi, "n_pixels": npix, "mu": rm, "sigma": rs,
                           "s": rk, "hist_chi2_per_bin": rc})
        results.append({"real": real, "model": {"mu": mu, "sigma": sigma, "s": s, "a": a, "b": b,
                                                "hist_chi2_per_bin": chi2, "radial": radial},
                        "synthetic": synth})
        print(f"{name[:10]}: real mean {real['mean']:.3f} std {real['std']:.3f} zeros "
              f"{real['hist'][0]:.3f} | fit mu {mu:.3f} sigma {sigma:.3f} DN s {s:.3f}, chi2/bin "
              f"{chi2:.1f}, kernel a {a:.3f} b {b:.3f}", flush=True)
        print(f"   synthetic mean {synth['mean']:.3f} std {synth['std']:.3f} zeros "
              f"{synth['hist'][0]:.3f}", flush=True)
        print("   hist real  " + " ".join(f"{v:.4f}" for v in real["hist"][:10]), flush=True)
        print("   hist synth " + " ".join(f"{v:.4f}" for v in synth["hist"][:10]), flush=True)
        print("   corr real  " + " ".join(f"{k} {v:+.3f}" for k, v in real["corr"].items()), flush=True)
        print("   corr synth " + " ".join(f"{k} {v:+.3f}" for k, v in synth["corr"].items()), flush=True)
        for r in radial:
            print(f"   radius {r['r_lo']:4d}-{r['r_hi']:4d} px: mu {r['mu']:.2f} sigma {r['sigma']:.2f} "
                  f"s {r['s']:.2f} chi2/bin {r['hist_chi2_per_bin']:.1f} ({r['n_pixels']} px)", flush=True)
        if name.startswith("2026-10-15"):
            # Synthetic patches go through the SAME selection as the real ones:
            # a frame-size synthetic field, the same blob mask, the same picker.
            # Parameters vary with radius (lens shading, level gradient): use the
            # ring fits, interpolated at each pixel's radius.
            rad = np.hypot(*np.ogrid[-648 + 0.5:648, -1152 + 0.5:1152])
            rc = [0.5 * (r["r_lo"] + r["r_hi"]) for r in radial]
            par = {k: np.interp(rad, rc, [r[k] for r in radial]) for k in ("mu", "sigma", "s")}
            field = render_background(par["mu"], par["sigma"], par["s"], a, b, (1296, 2304), rng)
            fmask = star_mask(field, None)
            print(f"   blob mask on a synthetic frame (false masks): {fmask.mean() * 100:.2f}% "
                  f"of pixels", flush=True)
            synth_p = empty_patches(field, fmask, 300, rng)
            for kind, ps in (("real", patches), ("synthetic", synth_p)):
                ps = np.array(ps)
                mx = ps.max(axis=(1, 2))
                print(f"   patches {kind:9s} n={len(ps)}: patch max mean {mx.mean():.2f}, any px >= 10 "
                      f"{100 * (mx >= 10).mean():.0f}%, patch std {ps.std(axis=(1, 2)).mean():.2f}, "
                      f"mean {ps.mean():.2f}", flush=True)
            patches_15 = (patches, synth_p)
    plot_hist(results, out)
    plot_mixed(patches_15[0], patches_15[1], rng, out)
    with open(out / "background.json", "w") as f:
        json.dump({"seed": SEED, "n_frames_per_night": N_FRAMES, "blob_dn": BLOB_DN, "blob_min_px": BLOB_MIN_PX,
                   "mask_grow_px": MASK_GROW_PX, "centroid_r_px": CENTROID_R_PX,
                   "nights": results}, f, indent=1)
    print(f"wrote background.json, background_hist.png, background_patches_mixed.png to {out}")


if __name__ == "__main__":
    main()
