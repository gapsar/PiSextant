"""Step 4 part 1: do real star trails follow the gyro prediction?

Written by Claude, 27 Oct 2026. Run from the centroid_cnn folder:
    python src/check_trails.py [--frames 40] [--run 2026-10-16_16-05-50 --rate-max 8]

Data: run 2026-10-15 (development), saved frames whose gyro rate during the
exposure is 1-4 deg/s. Stars: bright, isolated blobs (own detection).

Prediction of a trail (guide 4.2.1, CONVENTIONS "Gyro sign and projection"):
  - pixel (row, col), corner convention -> tetra3 undistortion about
    (648, 1152) with the run's median distortion k -> pinhole with
    f = 1152 / tan(FOV / 2) -> camera direction s0 (boresight, left, up).
  - gyro in the camera frame, w_cam = M_case w_imu, offset = mean over
    +-150 s; theta(t) = integral of w_cam from the star's mid time to t.
  - s(t) = rotation(-theta(t)) s0 (delta_s = -theta x s); project and
    distort back -> path p(t) - p(t_mid) in px.
  - star mid time on the IMU clock: t_exposure_mid + tau (one tau per star;
    with a rolling shutter, tau_best(row) = tau0 + row / 1296 * readout).
Model of the patch: B + A * sum over the path of a round super-Gaussian
(p = 1.3, width a free: the real width changes over the frame, step 3, and
a fixed a trades against the trail length), 2 ms time steps; free offset
(row, col), a, A, B. Stars where EVERY hypothesis has chi2/px > 2 x the
background variance are left out of the summary (neighbours, bad fits). Uniform weights, the same for every hypothesis
(LESSONS 19). The gamma curve is ignored (trail pixels are mostly < 19 DN).

Hypotheses, all fitted on the same patch at the star's best tau:
  nominal, mirror_col, mirror_row, swap (row <-> col), neg (point
  reflection = opposite rotation sign), no_roll (boresight rate = 0),
  roll_neg (boresight rate negated).

PREDICTIONS (written before the first run):
  P1 nominal has the lowest chi2 for >= 80% of the stars against
     mirror_col, mirror_row and swap. neg differs only for curved or
     uneven trails; where the trail is curved (> 1 px from its chord),
     nominal beats neg (the sign was measured frame to frame on 21 Oct).
  P2 the median best tau is 0-30 ms; best tau grows with the row, and the
     slope gives a readout time of 10-35 ms (estimate ~18 ms). The
     per-star tau may be too noisy for the slope; then report the SE.
  P3 the best length scale is 0.95-1.05 (f, FOV and gyro scale correct).
  P4 at r > 900 px, no_roll and roll_neg are clearly worse than nominal
     when the boresight rate is a large part of the motion (field rotation
     is real and has the expected sign).
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, optimize
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE.parents[1] / "runs"
RUN = RUNS / "2026-10-15_16-00-55"     # default; --run changes it
OUT = HERE / "reports" / "step4"
SEED = 20261027
H, W = 1296, 2304
C_ROW, C_COL = H / 2, W / 2
RATE_MIN, RATE_MAX = 1.0, 4.0        # deg/s, max during the exposure
HALF = 22                            # patch 45 x 45
PSF_A, PSF_P = 2.6, 1.3
DT = 0.002                           # path time step, s
TAUS = np.arange(-0.10, 0.1201, 0.02)
SCALES = np.arange(0.80, 1.201, 0.05)
HYPS = ["nominal", "mirror_col", "mirror_row", "swap", "neg", "no_roll", "roll_neg"]
STARS_PER_FRAME = 6


def jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def load_run():
    """Gyro (camera frame), capture records, median FOV and distortion."""
    log = jsonl(RUN / "solves" / "live-optimized.jsonl")
    m_case = np.array(log[0]["M_case"])
    fov, dist = [], []
    for d in log[1:]:
        s = d.get("solution")
        s = json.loads(s) if isinstance(s, str) else s
        if s and s.get("RA") is not None:
            fov.append(s["FOV"])
            dist.append(s["distortion"])
    imu = jsonl(RUN / "imu.jsonl")
    t = np.array([r["t"] for r in imu])
    order = np.argsort(t)
    t = t[order]
    w = (np.array([r["gyro"] for r in imu]) @ m_case.T)[order]
    cap = [c for c in jsonl(RUN / "capture.jsonl") if c.get("image_saved")]
    return t, w, cap, float(np.median(fov)), float(np.median(dist))


def undistort(rc, k):
    d = rc - [C_ROW, C_COL]
    r = np.linalg.norm(d, axis=-1, keepdims=True)
    return d * (1 - k * (2 / W) ** 2 * r ** 2) / (1 - k) + [C_ROW, C_COL]


def distort(rc, k):
    """Inverse of undistort (Newton on the radius, correct derivative)."""
    d = rc - [C_ROW, C_COL]
    ru = np.linalg.norm(d, axis=-1, keepdims=True)
    kp = k * (2 / W) ** 2
    rd = ru.copy()
    for _ in range(20):
        rd -= (rd * (1 - kp * rd ** 2) / (1 - k) - ru) / ((1 - 3 * kp * rd ** 2) / (1 - k))
    return d * rd / np.maximum(ru, 1e-9) + [C_ROW, C_COL]


def to_dir(rc, f, k):
    u = undistort(rc, k)
    s = np.stack([np.ones(len(u)), -(u[:, 1] - C_COL) / f, -(u[:, 0] - C_ROW) / f], -1)
    return s / np.linalg.norm(s, axis=1, keepdims=True)


def to_px(s, f, k):
    rc = np.stack([C_ROW - f * s[:, 2] / s[:, 0], C_COL - f * s[:, 1] / s[:, 0]], -1)
    return distort(rc, k)


def path(rc0, t_mid, tg, cg, f, k, hyp="nominal"):
    """Path p(t) - p(t_mid) (px, (row, col)) over the exposure."""
    ts = t_mid + np.arange(-0.15, 0.15 + 1e-9, DT)
    th = np.stack([np.interp(ts, tg, cg[:, j]) for j in range(3)], -1)
    th -= [np.interp(t_mid, tg, cg[:, j]) for j in range(3)]
    if hyp == "no_roll":
        th[:, 0] = 0
    if hyp == "roll_neg":
        th[:, 0] *= -1
    s0 = to_dir(np.array([rc0]), f, k)[0]
    s = Rotation.from_rotvec(-th).apply(s0)
    d = to_px(s, f, k) - to_px(s0[None], f, k)
    if hyp == "mirror_col":
        d[:, 1] *= -1
    elif hyp == "mirror_row":
        d[:, 0] *= -1
    elif hyp == "swap":
        d = d[:, ::-1].copy()
    elif hyp == "neg":
        d = -d
    return d


_YY, _XX = np.mgrid[0:2 * HALF + 1, 0:2 * HALF + 1] + 0.5


def render(d, off, a=PSF_A):
    """Unit-amplitude trail template; off = patch position of p(t_mid)."""
    q = ((_YY[None] - off[0] - d[:, 0, None, None]) ** 2
         + (_XX[None] - off[1] - d[:, 1, None, None]) ** 2) / a ** 2
    return np.exp(-0.5 * q ** PSF_P).sum(0) / len(d)


def fit(patch, d, start):
    """Least squares over offset and width a (A, B linear).
    Output: chi2 (DN^2), (row, col, a)."""
    y = patch.ravel()

    def cost(x):
        if not 1.0 < x[2] < 5.0:
            return 1e12
        m = render(d, x[:2], x[2]).ravel()
        X = np.stack([m, np.ones_like(m)], 1)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        return np.sum((y - X @ coef) ** 2)

    x0 = np.r_[start[:2], start[2] if len(start) > 2 else PSF_A]
    r = optimize.minimize(cost, x0, method="Nelder-Mead",
                          options={"xatol": 0.01, "fatol": 1e-3, "initial_simplex":
                                   [x0, x0 + [1, 0, 0], x0 + [0, 1, 0], x0 + [0, 0, 0.5]]})
    return float(r.fun), r.x


def frame_list(t, w, cap, n_frames, rng):
    cs = np.vstack([np.zeros(3), np.cumsum(w, 0)])
    good = []
    for c in cap:
        a, b = np.searchsorted(t, [c["t_exposure_mid"] - 150, c["t_exposure_mid"] + 150])
        i0, i1 = np.searchsorted(t, [c["t_exposure_start"], c["t_exposure_end"]])
        if b - a < 1000 or i1 - i0 < 15:
            continue
        rate = np.degrees(np.linalg.norm(w[i0:i1] - (cs[b] - cs[a]) / (b - a), axis=1)).max()
        if RATE_MIN < rate < RATE_MAX:
            good.append(c)
    pick = rng.choice(len(good), min(n_frames, len(good)), replace=False)
    return [good[i] for i in sorted(pick)], len(good)


def find_stars(img):
    """Bright isolated blobs. Output: list of (row, col, flux)."""
    bkg = np.median(img)
    sm = ndimage.gaussian_filter(img - bkg, 2.0)
    lab, n = ndimage.label(sm > 1.5)
    idx = np.arange(1, n + 1)
    flux = ndimage.sum(img - bkg, lab, idx)
    size = ndimage.sum(np.ones_like(img), lab, idx)
    com = ndimage.center_of_mass(np.clip(img - bkg, 0, None), lab, idx)
    out = []
    for j in np.argsort(-flux):
        if flux[j] < 300 or size[j] < 20:
            break
        r, c = com[j]
        if not (HALF + 2 < r < H - HALF - 3 and HALF + 2 < c < W - HALF - 3):
            continue
        box = lab[int(r) - HALF:int(r) + HALF + 1, int(c) - HALF:int(c) + HALF + 1]
        others = set(np.unique(box)) - {0, j + 1}
        if any(flux[o - 1] > 60 for o in others):
            continue
        out.append((r, c, float(flux[j])))
        if len(out) == STARS_PER_FRAME:
            break
    return out


def one_star(args):
    patch, rc, t_mid, tg, cg, f, k = args
    top = np.array([int(rc[0]) - HALF, int(rc[1]) - HALF], float)
    best = None
    tau_chi = []
    for tau in TAUS:
        d = path(rc, t_mid + tau, tg, cg, f, k)
        start = rc - top - d.mean(0)
        chi, off = fit(patch, d, start)
        tau_chi.append(chi)
        if best is None or chi < best[0]:
            best = (chi, tau, off)
    chi0, tau_b, off_b = best
    d0 = path(rc, t_mid + tau_b, tg, cg, f, k)
    res = {"row": float(rc[0]), "col": float(rc[1]),
           "r_px": float(np.hypot(rc[0] - C_ROW, rc[1] - C_COL)),
           "tau_best_s": float(tau_b), "tau_chi2": tau_chi,
           "length_px": float(np.linalg.norm(np.diff(d0, axis=0), axis=1).sum()),
           "chord_px": float(np.linalg.norm(d0[-1] - d0[0])),
           "curve_px": float(_curvature(d0)), "chi2": {}, "scale_chi2": []}
    th = np.stack([np.interp(t_mid + tau_b + np.array([-0.15, 0.15]), tg, cg[:, j]) for j in range(3)], -1)
    dth = th[1] - th[0]
    res["roll_share"] = float(abs(dth[0]) * res["r_px"] / max(np.linalg.norm(dth) * 1700, 1e-9))
    for h in HYPS:
        d = path(rc, t_mid + tau_b, tg, cg, f, k, h)
        res["chi2"][h] = fit(patch, d, rc - top - d.mean(0))[0] if h != "nominal" else chi0
    for s in SCALES:
        res["scale_chi2"].append(fit(patch, d0 * s, off_b)[0])
    res["offset"] = off_b.tolist()
    res["patch"] = patch
    res["path"] = d0
    return res


def _curvature(d):
    a, b = d[0], d[-1]
    u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    return np.abs((d[:, 0] - a[0]) * u[1] - (d[:, 1] - a[1]) * u[0]).max()


def main():
    global RUN, RATE_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--run", default=RUN.name)
    ap.add_argument("--rate-max", type=float, default=RATE_MAX)
    ap.add_argument("--summarize-only", action="store_true",
                    help="rebuild the summary from the existing JSON of --run")
    args = ap.parse_args()
    if args.summarize_only:
        summarize_only(args.run[:10])
        return
    RUN, RATE_MAX = RUNS / args.run, args.rate_max
    rng = np.random.default_rng(SEED)
    t, w, cap, fov, k = load_run()
    f = C_COL / np.tan(np.radians(fov) / 2)
    frames, n_good = frame_list(t, w, cap, args.frames, rng)
    print(f"FOV {fov:.3f} deg, f {f:.1f} px, distortion {k:.4f}; frames 1-4 deg/s: {n_good}, used {len(frames)}")
    jobs = []
    for c in frames:
        tm = c["t_exposure_mid"]
        a, b = np.searchsorted(t, [tm - 150, tm + 150])
        off = w[a:b].mean(0)
        sel = (t > tm - 1) & (t < tm + 1)
        tg = np.arange(tm - 0.6, tm + 0.6, 0.001)
        wg = np.stack([np.interp(tg, t[sel], w[sel, j] - off[j]) for j in range(3)], -1)
        cg = np.vstack([np.zeros((1, 3)), np.cumsum(0.5 * (wg[1:] + wg[:-1]) * 0.001, 0)])
        tg = tg[:len(cg)]
        img = np.asarray(Image.open(RUN / c["image"]), float)
        for r, cc, fl in find_stars(img):
            p = img[int(r) - HALF:int(r) + HALF + 1, int(cc) - HALF:int(cc) + HALF + 1]
            jobs.append((p, np.array([r, cc]), tm, tg, cg, f, k))
    print(f"stars: {len(jobs)}")
    with Pool(16) as pool:
        res = pool.map(one_star, jobs)
    summarize(res, fov, k, args.run[:10])


def summarize(res, fov, k, tag):
    """Summary of the star results: printed AND written to
    reports/step4/trail_check_<tag>.txt and .json (works offline)."""
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def say(text):
        print(text)
        lines.append(text)

    var = 3.3                         # DN^2, background variance (step 3, 15 Oct)
    npx = (2 * HALF + 1) ** 2
    sig = 30                          # |dchi2| / var: the test can tell the two apart
    long_ = [r for r in res if r["length_px"] >= 6 and min(r["chi2"].values()) / var / npx < 2]
    say(f"run {tag}: FOV {fov:.3f} deg, distortion {k:.4f}, stars {len(res)}")
    say(f"stars with a predicted trail >= 6 px and a usable fit: {len(long_)}")
    a_fit = np.array([r["offset"][2] for r in long_])
    say(f"fitted width a: median {np.median(a_fit):.2f}, p10-p90 {np.percentile(a_fit, [10, 90]).round(2)}")
    summ = {"fov_deg": fov, "distortion": k, "n_stars": len(res), "n_long": len(long_),
            "width_a_p10_p50_p90": np.percentile(a_fit, [10, 50, 90]).tolist()}
    dchi = lambda rs, h: np.array([(r["chi2"][h] - r["chi2"]["nominal"]) / var for r in rs])
    say(f"hypothesis vs nominal (all stars | only stars with |dchi2|/var > {sig}):")
    for h in HYPS[1:]:
        dd = dchi(long_, h)
        m = np.abs(dd) > sig
        summ[h] = {"nominal_better_frac": float(np.mean(dd > 0)), "dchi2_median": float(np.median(dd)),
                   "n_significant": int(m.sum()),
                   "nominal_better_frac_significant": float(np.mean(dd[m] > 0)) if m.any() else None}
        say(f"  {h:10s}: nominal better {np.mean(dd > 0):.2f} (median dchi2 {np.median(dd):7.1f}) | "
            f"n {m.sum():3d}, nominal better {np.mean(dd[m] > 0) if m.any() else float('nan'):.2f}")
    curved = [r for r in long_ if r["curve_px"] > 1]
    summ["curved_n"] = len(curved)
    if curved:
        dd = dchi(curved, "neg")
        m = np.abs(dd) > sig
        summ["neg_curved"] = {"n": len(curved), "nominal_better_frac": float(np.mean(dd > 0)),
                              "dchi2_p10_p50_p90": np.percentile(dd, [10, 50, 90]).tolist(),
                              "n_significant": int(m.sum()),
                              "nominal_better_frac_significant": float(np.mean(dd[m] > 0)) if m.any() else None}
        say(f"rotation sign (neg), curved > 1 px, n {len(curved)}: nominal better {np.mean(dd > 0):.2f}, "
            f"dchi2 p10/p50/p90 {np.percentile(dd, [10, 50, 90]).round(1)}; significant n {m.sum()}, "
            f"nominal better {np.mean(dd[m] > 0) if m.any() else float('nan'):.2f}")
    else:
        say("rotation sign (neg): no curved trail (> 1 px), not testable")
    edge = [r for r in long_ if r["r_px"] > 900 and r["roll_share"] > 0.3]
    for h in ["no_roll", "roll_neg"]:
        dd = dchi(edge, h)
        summ[h + "_edge"] = {"n": len(edge), "nominal_better_frac": float(np.mean(dd > 0)) if len(edge) else None}
        say(f"{h} at r > 900, roll share > 0.3 (n {len(edge)}): nominal better "
            f"{np.mean(dd > 0) if len(edge) else float('nan'):.2f}")
    for name, rs in [("all", long_), ("curved", curved)]:
        tau = np.array([r["tau_best_s"] for r in rs])
        row = np.array([r["row"] for r in rs])
        if len(rs) < 4:
            say(f"tau ({name}): too few stars")
            continue
        inner = (tau > TAUS[0] + 1e-6) & (tau < TAUS[-1] - 1e-6)
        tsum = np.sum([np.array(r["tau_chi2"]) - min(r["tau_chi2"]) for r in rs], 0) / var
        ent = {"median_s": float(np.median(tau)), "n": len(rs), "n_at_scan_edge": int((~inner).sum()),
               "summed_dchi2_curve": tsum.round(1).tolist()}
        say(f"tau ({name}, n {len(rs)}): median {np.median(tau)*1e3:.0f} ms, at scan edge {(~inner).sum()}")
        say(f"  summed dchi2/var per tau {np.round(TAUS*1e3).astype(int).tolist()} ms:")
        say(f"  {tsum.round(0).astype(int).tolist()}")
        if inner.sum() >= 4:
            A = np.stack([np.ones(inner.sum()), row[inner] / H], 1)
            coef, *_ = np.linalg.lstsq(A, tau[inner], rcond=None)
            resid = tau[inner] - A @ coef
            cov = np.linalg.inv(A.T @ A) * resid.var(ddof=2)
            ent.update({"tau0_s": float(coef[0]), "readout_s": float(coef[1]),
                        "readout_se_s": float(np.sqrt(cov[1, 1])), "resid_std_s": float(resid.std())})
            say(f"  fit (stars inside the scan): tau0 {coef[0]*1e3:.1f} ms + row/1296 * {coef[1]*1e3:.1f} "
                f"(SE {np.sqrt(cov[1,1])*1e3:.1f}) ms")
        summ["tau_" + name] = ent
    sc = np.array([SCALES[int(np.argmin(r["scale_chi2"]))] for r in long_])
    summ["scale"] = {"median": float(np.median(sc)), "p10_p90": np.percentile(sc, [10, 90]).tolist(),
                     "n_at_bounds": int(np.sum((sc < SCALES[0] + 1e-6) | (sc > SCALES[-1] - 1e-6)))}
    say(f"length scale: median {np.median(sc):.2f}, p10-p90 {np.percentile(sc, [10, 90]).round(2)}, "
        f"at the scan bounds {summ['scale']['n_at_bounds']}")
    keep = ["row", "col", "r_px", "tau_best_s", "length_px", "chord_px", "curve_px", "roll_share", "chi2", "scale_chi2", "tau_chi2", "offset"]
    json.dump({"summary": summ, "taus_s": TAUS.tolist(), "scales": SCALES.tolist(),
               "stars": [{k2: r[k2] for k2 in keep} for r in res]},
              open(OUT / f"trail_check_{tag}.json", "w"), indent=1)
    open(OUT / f"trail_check_{tag}.txt", "w").write("\n".join(lines) + "\n")
    if long_ and "patch" in long_[0]:
        plot(long_, tag)


def summarize_only(tag):
    """Rebuild the summary from an existing trail_check_<tag>.json (no plot)."""
    d = json.load(open(OUT / f"trail_check_{tag}.json"))
    summarize(d["stars"], d["summary"]["fov_deg"], d["summary"]["distortion"], tag)


def plot(stars, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 16 stars spread over the frame: sort by angle and radius bins
    pick = sorted(stars, key=lambda r: (r["r_px"] > 900, np.arctan2(r["row"] - C_ROW, r["col"] - C_COL)))
    pick = [pick[i] for i in np.linspace(0, len(pick) - 1, min(16, len(pick))).astype(int)]
    fig, ax = plt.subplots(4, 4, figsize=(12, 12))
    for a, r in zip(ax.ravel(), pick):
        a.imshow(r["patch"], cmap="gray", vmin=0, vmax=np.percentile(r["patch"], 99.5))
        p = r["path"] + np.array(r["offset"][:2]) - 0.5      # imshow pixel centres at integers
        a.plot(p[:, 1], p[:, 0], "r-", lw=1)
        a.plot(p[0, 1], p[0, 0], "go", ms=3)
        a.set_title(f"r{r['row']:.0f} c{r['col']:.0f} tau {r['tau_best_s']*1e3:.0f} ms", fontsize=8)
        a.axis("off")
    fig.suptitle(f"{tag} trails, gyro prediction (red, green = exposure start), nominal convention")
    fig.tight_layout()
    fig.savefig(OUT / f"trail_overlay_{tag}.png", dpi=90)


if __name__ == "__main__":
    main()
