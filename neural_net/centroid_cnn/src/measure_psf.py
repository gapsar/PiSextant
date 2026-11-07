"""Step 3, part 3: the PSF of still stars per radius ring (guide 5, step 3;
parameters 4.1.2, 4.1.3, 4.1.9a). Prototypes and their tests: WORKLOG 25 Oct,
step 3 parts 3, 4 and 4b (scratch/psf_*_claude.py, tone_proto_claude.py).

What it does, per development night (13, 15, 16 Oct):
1. Still frames: during the exposure, |cumulative gyro angle| x (f + r_max)
   < STILL_PX, gyro offset = mean over +-150 s. So the trail is < 1 px even
   in the corners.
2. Stars: logged centroids, isolated (no other centroid within ISO_PX, no
   other blob in the patch), 3 x 3 peak >= PEAK_MIN DN above the patch
   median, and NOT on a fixed pixel: the night's 60-frame median must be
   < HOT_MED_DN above background within 1 px (hot pixels are logged as
   centroids and pass all other cuts, see WORKLOG part 3).
3. Model of one star, from linear light to the 8-bit Y value:
       lin = B + A * f(x)       f = elliptical super-Gaussian exp(-0.5 q^p),
                                q = (u/a1)^2 + (v/a2)^2, pixel-integrated
       Y   = gamma(ce(lin))     the Pi ISP curve and contrast stretch, from
                                configs/imx708_vc4_tuning.json
       the fit compares E[clip(round(Y + noise))] with the data (night noise
       model of part 1), so the 0-clip does not bias the wings.
   The linear unit is the Y DN of the first gamma segment (slope 1 below
   about 19 DN). Fit weights are FIXED (LESSONS 19): taken once from a plain
   Gaussian fit, then the super-Gaussian is fitted with the same weights.
4. Per radius ring: medians of a1, a2, p, FWHM, axis ratio, and the angle of
   the major axis to the radial direction. Stars with A >= A_SHAPE only (the
   fit widens faint stars). Standard errors by bootstrap over BLOCK_S time
   blocks: one star appears in many consecutive still frames, so the stars
   are not independent.
5. Check (guide step 3): every real star gets a synthetic twin with the
   shape (a1, a2, p) of a random PEER star of the same ring (its own
   amplitude, background night and angle), the
   measured correlated skewed noise, photon noise, the ISP curve and the
   8-bit rounding; the twins go through the same fits. Real and synthetic
   stacked radial profiles must agree within 3 combined SE in every bin
   (criterion fixed before the run). The twins also measure the fit bias
   (fitted median minus input median) per ring.
   Why peer shapes, not the ring median: the shapes vary inside a ring, and
   a stack of mixed widths has heavier wings than the median shape. With
   median twins the check failed at r 5-8 px in 4 of 5 rings; with each
   star's own shape or a peer shape it passed there
   (scratch/psf_twin_test_claude.py, 27 Oct). So the generator must DRAW
   the shape per star from the ring distribution (psf.json shape_quantiles).

6. Shape vs flux: in the centre (r < 600) the faint stars are wider and
   flatter than the bright ones, inside one night too, and the selection
   does not explain it (scratch/psf_faint_test*_claude.py, 27 Oct; cause not
   proven, maybe the ISP denoise). So psf.json also gives the shape
   quantiles per ring AND flux tercile, over all fitted stars: the
   generator draws the shape from the matching (ring, flux) cell. These
   are FITTED shapes: for faint stars they include the fit scatter, so the
   spread is a little wider than the truth.

Outputs in reports/step3/: psf.json, psf.png.

Run from the centroid_cnn folder:
    python3 src/measure_psf.py
"""

import json
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib
from PIL import Image
from scipy.ndimage import label
from scipy.optimize import least_squares
from scipy.special import ndtr

from measure_background import load_centroids, correlated_field, skew_shape
from measure_hot_pixels import read_night

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
RUNS_DIR = HERE.parents[1] / "runs"
NIGHTS = ["2026-10-13_16-14-01", "2026-10-15_16-00-55", "2026-10-16_16-05-50"]
SEED = 20261027
F_PX, R_MAX = 1700.0, 1322.0        # focal length and corner radius, px
C_ROW, C_COL = 648.0, 1152.0        # image centre
STILL_PX = 1.0                      # max trail bound during the exposure, px
HALF = 12                           # patch = 25 x 25 px
ISO_PX = 18                         # no other centroid closer than this
PEAK_MIN = 10.0                     # 3 x 3 peak above the patch median, DN
HOT_MED_DN = 4                      # fixed pixel: median image >= this above bkg
SUB = 3                             # sub-pixels per pixel and axis (integration)
A_SHAPE = 18.0                      # shape statistics: fitted A >= this
RINGS = [0, 300, 600, 900, 1100, 1400]
BLOCK_S = 600                       # bootstrap block, s
N_BOOT = 300
R_BINS = 13                         # profile bins 0..12 px (rounded radius)
# photon-noise slope (DN^2 per DN) per radius, from part 1 (WORKLOG 25 Oct):
C1_EDGES = [0, 200, 400, 600, 800, 1000, 1400]
C1_VALUES = [0.72, 0.80, 0.94, 1.19, 1.48, 1.77]
BKG = json.load(open(HERE / "reports" / "step3" / "background.json"))
TUNING = json.load(open(HERE / "configs" / "imx708_vc4_tuning.json"))

N = 2 * HALF + 1
_FINE = (np.arange(N * SUB) + 0.5) / SUB - 0.5
FR, FC = np.meshgrid(_FINE, _FINE, indexing="ij")
PR, PC = np.mgrid[0:N, 0:N].astype(float)


# ---------- ISP curve and noise model ----------

def _gamma_tables():
    """Gamma curve in Y DN, with the linear unit = slope 1 in the first segment."""
    algs = {list(a)[0]: list(a.values())[0] for a in TUNING["algorithms"]}
    g = np.array(algs["rpi.contrast"]["gamma_curve"], float).reshape(-1, 2)
    scale = g[1, 1] / g[1, 0] / 256          # input unit -> Y DN of segment 1
    return g[:, 0] * scale, g[:, 1] / 256, scale


G_IN, G_OUT, G_SCALE = _gamma_tables()
# contrast stretch on a dark frame (both clamps hold; WORKLOG part 4b):
# identity below lo_level, then a straight line to (hi_hist_clamp, hi_level)
CE_KNEE = 0.015 * 65536 * G_SCALE
CE_GAIN = (0.95 * 65536 - 0.015 * 65536) / (0.95 * 65536 - 2000 - 0.015 * 65536)


def tone(lin):
    """Linear light (Y DN units of segment 1) -> Y DN, as the Pi ISP does."""
    x = np.where(lin > CE_KNEE, CE_KNEE + CE_GAIN * (lin - CE_KNEE), lin)
    return np.where(x <= 0, x, np.interp(x, G_IN, G_OUT))


def night_model(night):
    return next(n["model"] for n in BKG["nights"] if n["real"]["night"] == night)


def expected_lookup(sigma, s):
    """Table m -> E[clip(round(m + sigma * skew(z, s)), 0, 255)], m in [-6, 14]."""
    ms = np.linspace(-6, 14, 401)
    mm, dd = math.exp(s * s / 2), math.sqrt((math.exp(s * s) - 1) * math.exp(s * s))
    edges = np.arange(0, 40) + 0.5
    es = []
    for m in ms:
        arg = (edges - m) * dd / sigma + mm
        cdf = np.where(arg > 0, ndtr(np.log(np.maximum(arg, 1e-300)) / s), 0.0)
        es.append(np.diff(np.concatenate([[0.0], cdf, [1.0]])) @ np.arange(41))
    return ms, np.array(es)


def c1_at(r):
    return C1_VALUES[min(np.searchsorted(C1_EDGES, r, side="right") - 1, len(C1_VALUES) - 1)]


# ---------- selection ----------

def still_frames(run):
    """Still frames of a run. Output: list of (image name, t_mid)."""
    imu = [json.loads(line) for line in open(run / "imu.jsonl")]
    t = np.array([r["t"] for r in imu])
    g = np.array([r["gyro"] for r in imu])
    order = np.argsort(t)
    t, g = t[order], g[order]
    cs = np.vstack([np.zeros(3), np.cumsum(g, axis=0)])
    out = []
    for line in open(run / "capture.jsonl"):
        c = json.loads(line)
        if not c.get("image_saved"):
            continue
        a, b = np.searchsorted(t, [c["t_exposure_mid"] - 150, c["t_exposure_mid"] + 150])
        i0, i1 = np.searchsorted(t, [c["t_exposure_start"], c["t_exposure_end"]])
        if b - a < 1000 or i1 - i0 < 15:
            continue
        w = g[i0 - 1:i1 + 1] - (cs[b] - cs[a]) / (b - a)
        dt = np.diff(t[i0 - 1:i1 + 1])
        theta = np.cumsum(w[:-1] * dt[:, None], axis=0)
        if np.linalg.norm(theta, axis=1).max() * (F_PX + R_MAX) < STILL_PX:
            out.append((Path(c["image"]).name, c["t_exposure_mid"]))
    return out


def cut_stars(img, cents, fixed):
    """Isolated, bright enough, not-fixed stars. Output: list of (row, col, patch)."""
    h, w = img.shape
    out = []
    for r, c in cents:
        ri, ci = int(round(r)), int(round(c))
        if not (HALF <= ri < h - HALF and HALF <= ci < w - HALF):
            continue
        if np.sum(np.hypot(cents[:, 0] - r, cents[:, 1] - c) < ISO_PX) > 1:
            continue
        if fixed[ri - 1:ri + 2, ci - 1:ci + 2].any():
            continue
        p = img[ri - HALF:ri + HALF + 1, ci - HALF:ci + HALF + 1].astype(float)
        if p[HALF - 1:HALF + 2, HALF - 1:HALF + 2].mean() - np.median(p) < PEAK_MIN or p.max() >= 250:
            continue
        lab, k = label(p >= 8)
        sizes = np.bincount(lab.ravel(), minlength=k + 1)
        own = lab[HALF, HALF]
        if any(j != own and sizes[j] >= 3 for j in range(1, k + 1)):
            continue
        out.append((ri, ci, p))
    return out


# ---------- fits ----------

def unit_shape(p, form):
    """Pixel-integrated unit-peak shape. p = [r0, c0, a1, a2, theta, (pexp)]."""
    r0, c0, a1, a2, th = p[:5]
    u = (FR - r0) * math.cos(th) + (FC - c0) * math.sin(th)
    v = -(FR - r0) * math.sin(th) + (FC - c0) * math.cos(th)
    q = (u / a1) ** 2 + (v / a2) ** 2
    f = np.exp(-0.5 * q) if form == "G" else np.exp(-0.5 * q ** p[5])
    return f.reshape(N, SUB, N, SUB).mean(axis=(1, 3))


def fit_one(patch, lut, var0, c1, form, wvar):
    """One fit. wvar None: weights from the model (start fit only). Output:
    (params [.., A, B], fixed weights from this fit)."""
    ms, es = lut
    k = 5 if form == "G" else 6

    def expected(p):
        y = tone(p[k + 1] + p[k] * unit_shape(p, form))
        return np.where(y > ms[-1], y, np.interp(y, ms, es)), y

    def resid(p):
        m, y = expected(p)
        v = wvar if wvar is not None else var0 + c1 * np.maximum(y - p[k + 1], 0)
        return ((m - patch) / np.sqrt(v)).ravel()

    peak = patch[HALF - 1:HALF + 2, HALF - 1:HALF + 2].mean()
    extra0, extra_lo, extra_hi = ([], [], []) if form == "G" else ([1.0], [0.4], [3.0])
    sol = least_squares(resid, [HALF, HALF, 2.5, 2.2, 0.0] + extra0 + [peak, 1.0],
                        bounds=([HALF - 3, HALF - 3, 0.5, 0.5, -10] + extra_lo + [0, -5],
                                [HALF + 3, HALF + 3, 15, 15, 10] + extra_hi + [800, 10]),
                        x_scale="jac")
    p = sol.x.copy()
    if p[3] > p[2]:
        p[2], p[3], p[4] = p[3], p[2], p[4] + math.pi / 2
    p[4] = (p[4] + math.pi / 2) % math.pi - math.pi / 2
    y = tone(p[k + 1] + p[k] * unit_shape(p, form))
    return p, var0 + c1 * np.maximum(y - p[k + 1], 0)


def fit_star(args):
    """Gaussian start fit (gives the fixed weights), then the super-Gaussian.
    Output: SG params [r0, c0, a1, a2, theta, p, A, B] or None."""
    patch, lut, var0, c1 = args
    try:
        _, w = fit_one(patch, lut, var0, c1, "G", None)
        return fit_one(patch, lut, var0, c1, "SG", w)[0]
    except Exception:
        return None


def profile(patch, p, lut):
    """Radial profile (data - E[B]) / (Y peak above B), bins 0..R_BINS-1 px."""
    ms, es = lut
    eb = np.interp(p[7], ms, es)
    amp = float(tone(p[7] + p[6]) - p[7])
    rb = np.minimum(np.round(np.hypot(PR - p[0], PC - p[1])).astype(int), R_BINS - 1)
    s = np.bincount(rb.ravel(), (patch - eb).ravel(), R_BINS)
    n = np.bincount(rb.ravel(), None, R_BINS)
    return s / np.maximum(n, 1) / max(amp, 1e-6)


# ---------- synthetic twins ----------

def render_twin(args):
    """Synthetic still star with a given SG shape, through the camera model."""
    shape, amp, night, r_frame, seed = args
    rng = np.random.default_rng(seed)
    m = night_model(night)
    p = [HALF + rng.uniform(-0.5, 0.5), HALF + rng.uniform(-0.5, 0.5)] + list(shape)
    star = amp * unit_shape(p, "SG")
    bkg = m["mu"] + m["sigma"] * skew_shape(correlated_field((N, N), m["a"], m["b"], rng), m["s"])
    photon = np.sqrt(c1_at(r_frame) * star) * correlated_field((N, N), m["a"], m["b"], rng)
    return np.clip(np.round(tone(bkg + star + photon)), 0, 255)


# ---------- statistics ----------

def block_boot(values, blocks, stat, rng):
    """Bootstrap SE of stat(values) over resampled time blocks."""
    ub = np.unique(blocks)
    groups = [np.nonzero(blocks == b)[0] for b in ub]
    out = []
    for _ in range(N_BOOT):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(ub), len(ub))])
        out.append(stat(values[pick]))
    return np.std(out, axis=0)


def sg_integral(a1, a2, p):
    """Volume of exp(-0.5 q^p) over the plane: pi a1 a2 2^(1/p) Gamma(1 + 1/p)."""
    return math.pi * a1 * a2 * 2 ** (1 / p) * math.gamma(1 + 1 / p)


def fwhm(a, p):
    return 2 * a * np.sqrt((2 * math.log(2)) ** (1 / p))


def main():
    rng = np.random.default_rng(SEED)
    pool = Pool(os.cpu_count())
    luts = {n: expected_lookup(night_model(n)["sigma"], night_model(n)["s"]) for n in NIGHTS}

    patches, meta = [], []      # meta: night index, row, col, t_mid
    for k, night in enumerate(NIGHTS):
        run = RUNS_DIR / night
        still = still_frames(run)
        cents = load_centroids(run)
        med = np.median(read_night(run)[0], axis=0)
        fixed = med - med.mean() >= HOT_MED_DN
        n0 = len(patches)
        for name, t in still:
            if name in cents:
                img = np.asarray(Image.open(run / "images" / name))
                for r, c, p in cut_stars(img, cents[name], fixed):
                    patches.append(p)
                    meta.append((k, r, c, t))
        print(f"{night}: still frames {len(still)}, stars {len(patches) - n0}", flush=True)
    meta = np.array(meta)
    rad = np.hypot(meta[:, 1] - C_ROW, meta[:, 2] - C_COL)
    var0 = {n: night_model(n)["sigma"] ** 2 for n in NIGHTS}
    jobs = [(p, luts[NIGHTS[int(m[0])]], var0[NIGHTS[int(m[0])]], c1_at(r)) for p, m, r in zip(patches, meta, rad)]
    fits = pool.map(fit_star, jobs, chunksize=8)
    ok = np.array([f is not None for f in fits])
    P = np.array([f if f is not None else np.full(8, np.nan) for f in fits])
    good = ok & (P[:, 6] >= A_SHAPE) & (P[:, 2] < 14) & (P[:, 3] >= 1.1)
    blocks = meta[:, 0] * 1e7 + np.floor(meta[:, 3] / BLOCK_S)
    radial = np.arctan2(meta[:, 2] - C_COL, meta[:, 1] - C_ROW)
    dang = np.degrees(np.abs((P[:, 4] - radial + np.pi / 2) % np.pi - np.pi / 2))
    print(f"stars fitted {ok.sum()}, used for shape (A >= {A_SHAPE}) {good.sum()}", flush=True)

    rings = []
    for lo, hi in zip(RINGS[:-1], RINGS[1:]):
        idx = np.nonzero(good & (rad >= lo) & (rad < hi))[0]
        if len(idx) < 10:
            rings.append({"r_px": [lo, hi], "n": int(len(idx))})
            continue
        sub = P[idx]
        bl = blocks[idx]
        med = lambda v: np.median(v, axis=0)
        cols = np.column_stack([sub[:, 2], sub[:, 3], sub[:, 5], sub[:, 2] / sub[:, 3],
                                fwhm(sub[:, 2], sub[:, 5]), fwhm(sub[:, 3], sub[:, 5])])
        m_all, se_all = med(cols), block_boot(cols, bl, med, rng)
        el = sub[:, 2] / sub[:, 3] > 1.15
        ang = np.histogram(dang[idx][el], bins=[0, 30, 60, 90])[0]

        # synthetic twins with a peer's shape, through the same fits
        peers = idx[rng.integers(0, len(idx), len(idx))]
        twin_args = []
        for j, (i, q) in enumerate(zip(idx, peers)):
            sh = [P[q, 2], P[q, 3], P[i, 4], P[q, 5]]
            twin_args.append((sh, P[i, 6], NIGHTS[int(meta[i, 0])], rad[i], SEED + 1000 * lo + j))
        twins = pool.map(render_twin, twin_args, chunksize=8)
        tjobs = [(tw, luts[a[2]], var0[a[2]], c1_at(a[3])) for tw, a in zip(twins, twin_args)]
        tfits = pool.map(fit_star, tjobs, chunksize=8)
        tok = [j for j, f in enumerate(tfits) if f is not None]
        TP = np.array([tfits[j] for j in tok])
        given = np.array([twin_args[j][0] for j in tok])[:, [0, 1, 3]]
        bias = np.median(TP[:, [2, 3, 5]], axis=0) - np.median(given, axis=0)

        prof_r = np.array([profile(patches[i], P[i], jobs[i][1]) for i in idx])
        prof_s = np.array([profile(twins[j], tfits[j], tjobs[j][1]) for j in tok])
        pr, ps = prof_r.mean(axis=0), prof_s.mean(axis=0)
        se_r = block_boot(prof_r, bl, lambda v: v.mean(axis=0), rng)
        se_s = prof_s.std(axis=0) / np.sqrt(len(prof_s))
        z = (pr - ps) / np.sqrt(se_r ** 2 + se_s ** 2)
        rings.append({
            "r_px": [lo, hi], "n": int(len(idx)), "n_blocks": int(len(np.unique(bl))),
            "a1": [round(m_all[0], 3), round(se_all[0], 3)], "a2": [round(m_all[1], 3), round(se_all[1], 3)],
            "p": [round(m_all[2], 3), round(se_all[2], 3)], "axis_ratio": [round(m_all[3], 3), round(se_all[3], 3)],
            "fwhm_major": [round(m_all[4], 3), round(se_all[4], 3)], "fwhm_minor": [round(m_all[5], 3), round(se_all[5], 3)],
            "major_vs_radial_deg_counts_0_30_60_90": ang.tolist(),
            "fit_bias_a1_a2_p": np.round(bias, 3).tolist(),
            "shape_quantiles_p10_p50_p90": {k: np.round(np.percentile(v, [10, 50, 90]), 3).tolist() for k, v in
                                            [("a1", sub[:, 2]), ("a2", sub[:, 3]), ("p", sub[:, 5]),
                                             ("axis_ratio", sub[:, 2] / sub[:, 3])]},
            "profile_real": np.round(pr, 4).tolist(), "profile_real_se": np.round(se_r, 4).tolist(),
            "profile_synth": np.round(ps, 4).tolist(), "profile_synth_se": np.round(se_s, 4).tolist(),
            "profile_z": np.round(z, 2).tolist(),
            "check_pass": bool(np.all(np.abs(z) < 3)),
        })
        rr = rings[-1]
        print(f"ring {lo:4d}-{hi:4d}: n {rr['n']:4d} blocks {rr['n_blocks']:3d} | a1 {rr['a1']} a2 {rr['a2']} p {rr['p']} "
              f"ratio {rr['axis_ratio']} | FWHM {rr['fwhm_minor'][0]:.2f} x {rr['fwhm_major'][0]:.2f} | angle {ang.tolist()} "
              f"| bias a1 a2 p {rr['fit_bias_a1_a2_p']} | profile max|z| {np.max(np.abs(z)):.1f} "
              f"{'PASS' if rr['check_pass'] else 'FAIL'}", flush=True)

    # shape per ring and flux tercile, all fitted stars (step 6 of the docstring)
    usable = ok & (P[:, 2] < 14) & (P[:, 3] >= 1.1)
    flux = np.array([P[i, 6] * sg_integral(P[i, 2], P[i, 3], P[i, 5]) if usable[i] else np.nan
                     for i in range(len(P))])
    by_flux = []
    for lo, hi in zip(RINGS[:-1], RINGS[1:]):
        ring = usable & (rad >= lo) & (rad < hi)
        if ring.sum() < 30:
            continue
        edges = np.nanpercentile(flux[ring], [0, 100 / 3, 200 / 3, 100])
        for flo, fhi in zip(edges[:-1], edges[1:]):
            m = ring & (flux >= flo) & (flux <= fhi)
            q = lambda v: np.round(np.percentile(v[m], [10, 50, 90]), 3).tolist()
            by_flux.append({"r_px": [lo, hi], "flux_lin": [round(flo, 1), round(fhi, 1)], "n": int(m.sum()),
                            "a1": q(P[:, 2]), "a2": q(P[:, 3]), "p": q(P[:, 5]),
                            "peak_A_lin": q(P[:, 6])})
            print(f"  shape r {lo:4d}-{hi:4d} flux {flo:6.0f}-{fhi:6.0f} n {m.sum():3d}: a1 {by_flux[-1]['a1']} "
                  f"a2 {by_flux[-1]['a2']} p {by_flux[-1]['p']}", flush=True)

    out = {
        "settings": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, list, str))
                     and k not in ("NIGHTS",)},
        "nights": NIGHTS,
        "model": "lin = B + A exp(-0.5 q^p), q = (u/a1)^2 + (v/a2)^2; Y = gamma(ce(lin)); then noise, round, clip",
        "ce": {"knee_lin": round(CE_KNEE, 3), "gain": round(CE_GAIN, 4)},
        "stars_fitted": int(ok.sum()), "stars_shape": int(good.sum()),
        "rings": rings,
        "shape_by_ring_and_flux_p10_p50_p90": by_flux,
    }
    json.dump(out, open(HERE / "reports" / "step3" / "psf.json", "w"), indent=1)
    plot(rings, HERE / "reports" / "step3" / "psf.png")


def plot(rings, path):
    done = [r for r in rings if "profile_real" in r]
    fig, ax = plt.subplots(1, len(done) + 1, figsize=(3.4 * (len(done) + 1), 3.4))
    for a, r in zip(ax[:-1], done):
        x = np.arange(R_BINS)
        a.errorbar(x, r["profile_real"], 2 * np.array(r["profile_real_se"]), fmt="o", ms=3, label="real")
        a.errorbar(x + 0.15, r["profile_synth"], 2 * np.array(r["profile_synth_se"]), fmt="s", ms=3, label="synthetic")
        a.set_yscale("symlog", linthresh=0.01)
        a.set_title(f"r {r['r_px'][0]}-{r['r_px'][1]} px, n {r['n']} {'PASS' if r['check_pass'] else 'FAIL'}", fontsize=9)
        a.set_xlabel("radius (px)")
    ax[0].set_ylabel("profile / peak (2 SE bars)")
    ax[0].legend(fontsize=8)
    mid = [np.mean(r["r_px"]) for r in done]
    for key, lab in [("a1", "a1 major"), ("a2", "a2 minor"), ("p", "flat-top p")]:
        ax[-1].errorbar(mid, [r[key][0] for r in done], [2 * r[key][1] for r in done], fmt="o-", ms=3, label=lab)
    ax[-1].set_xlabel("radius from centre (px)")
    ax[-1].set_title("shape vs radius (raw fits)", fontsize=9)
    ax[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
