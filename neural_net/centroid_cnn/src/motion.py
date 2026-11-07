"""Step 4 part 3: true and measured motion, channels 2 and 3, and the checks.

Written by Claude, 29 Oct 2026. Run from the centroid_cnn folder:
    python src/motion.py [--n 2000]
Results: reports/step4/motion_checks.txt and .json, motion_samples.png.
The step-5 generator imports the functions (load_windows, draw_motion,
channels, render_trail); all parameters are in configs/step4.json.

Definitions (guide sections 2, 4.1.5, 4.2; CONVENTIONS):
- TRUE motion: PCHIP interpolation (monotone, no overshoot) through the
  samples of a real gyro window (camera frame, offset removed), 1 ms steps.
  A cubic spline was tried first: on 16 Oct it made the trails 17% longer
  than the data, by overshoot between noisy samples
  (scratch/c5_cause_claude.py). A fraction of the windows is sped up
  (rate x k, time / k). Plus a true-only vibration (4.1.6).
- MEASURED motion: the true rate sampled at a real interval sequence of the
  same run, with the 4.2 errors (time error tau, rolling-shutter readout
  error, offset residual, white noise, scale, misalignment), then linear
  interpolation; projected with a perturbed f and centre.
- Path of a star at pixel (row, col): rotation(-theta(t)) of its direction
  (CONFIRMED on real trails, step 4 part 1), tetra3 distortion, over the
  300 ms window of ITS row; d(t) = p(t) - p(t_mid), px, (row, col).
- Channel 2: dwell-time map of the MEASURED path, 32 x 32, its mid-time point
  at the patch centre (16.5, 16.5), bilinear weights, sum 1. Channel 3: per
  pixel, the dwell-weighted mean time fraction (0 = exposure start, 1 = end);
  0 where channel 2 is 0.

Checks and PREDICTIONS (written before the first run):
  C1 zero error, dense sampling: measured path = true path within 1e-3 px.
  C2 trail = PSF convolved with channel 2 (zero error): difference < 2% of
     the peak (bilinear splat vs exact sub-pixel), light centroid equal
     within 0.005 px (the bilinear splat keeps the first moment).
  C3 sampling only (real 79 Hz intervals, no other error): rms path error
     p50 < 0.02 px, p99 < 0.2 px; larger for long and curved trails.
  C4 full error model: rms path error p50 0.05-0.2 px, p90 0.2-0.6 px; the
     largest share from tau (rate x 10 ms x 9 px per deg/s).
     (Added after the smoke run, before the full run: C4 is reported on the
     accepted samples, without and with the true-only vibration, binned by
     the centre trail length; the vibration made the path-length bins wrong.)
  C5 centre trail lengths of the real-share samples match gyro.json of the
     same nights (quantiles within 10%); the synthetic share moves p99 up to
     about 30-60 px; the max-extent cut rejects < 5% of the samples.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parents[1]
RUNS = HERE.parents[1] / "runs"
OUT = HERE / "reports" / "step4"
CFG = json.load(open(HERE / "configs" / "step4.json"))


# ---------------------------------------------------------------- windows
def load_windows(cfg, run, n, rng, frac=(0.0, 1.0), end_min=None):
    """n random real gyro windows of one run, with the exposure mid inside the
    time fraction frac of the run (splits: no window shared between them);
    end_min cuts the run (20 Oct: hand-held after minute 75). Output: list of dicts
    {t: sample times relative to the exposure mid (s), w: (N, 3) rad/s camera
    frame, offset removed, dt: the run's sample intervals (pool)}."""
    wc = cfg["windows"]
    log = sorted((RUNS / run / "solves").glob("*.jsonl"))[0]
    with open(log) as fh:
        m_case = np.array(json.loads(fh.readline())["M_case"])
    t, g = [], []
    with open(RUNS / run / "imu.jsonl") as fh:
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
    mids = [json.loads(line).get("t_exposure_mid") for line in open(RUNS / run / "capture.jsonl")]
    mids = np.array([m for m in mids if m is not None])
    t_end = mids.max() if end_min is None else min(mids.max(), mids.min() + 60 * end_min)
    span = t_end - mids.min()
    mids = mids[(mids >= mids.min() + frac[0] * span) & (mids <= mids.min() + frac[1] * span)]
    dt_pool = np.diff(t)
    dt_pool = dt_pool[dt_pool > 0]
    out = []
    for m in rng.permutation(mids):
        a, b = np.searchsorted(t, [m - wc["offset_window_s"], m + wc["offset_window_s"]])
        i0, i1 = np.searchsorted(t, [m - wc["half_s"], m + wc["half_s"]])
        e0, e1 = np.searchsorted(t, [m - 0.3, m + 0.3])
        if b - a < 1000 or e1 - e0 < 30 or i1 - i0 < 2:
            continue
        if t[i0] > m - wc["half_s"] + 0.1 or t[i1 - 1] < m + wc["half_s"] - 0.1:
            continue                       # the samples must cover the window
        ww = w[i0:i1] - (cs[b] - cs[a]) / (b - a)
        near = np.abs(t[i0:i1] - m) < 0.3  # rate limit near the exposure only
        if np.degrees(np.linalg.norm(ww[near], axis=1)).max() > wc["max_rate_deg_s"]:
            continue
        out.append({"t": t[i0:i1] - m, "w": ww, "dt": dt_pool})
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------- geometry
def undistort(rc, cam):
    k = cam["distortion_k"]
    c = np.array([cam["c_row"], cam["c_col"]])
    d = rc - c
    r2 = np.sum(d ** 2, axis=-1, keepdims=True)
    return d * (1 - k * (2 / cam["width"]) ** 2 * r2) / (1 - k) + c


def distort(rc, cam):
    k = cam["distortion_k"]
    kp = k * (2 / cam["width"]) ** 2
    c = np.array([cam["c_row"], cam["c_col"]])
    d = rc - c
    ru = np.linalg.norm(d, axis=-1, keepdims=True)
    rd = ru.copy()
    for _ in range(20):
        rd -= (rd * (1 - kp * rd ** 2) / (1 - k) - ru) / ((1 - 3 * kp * rd ** 2) / (1 - k))
    return d * rd / np.maximum(ru, 1e-9) + c


def project_path(theta, rc, cam):
    """theta (T, 3) rad camera frame (0 at mid time) -> d(t) px (T, 2)."""
    u = undistort(np.asarray(rc, float)[None], cam)[0]
    f = cam["f_px"]
    s0 = np.array([1.0, -(u[1] - cam["c_col"]) / f, -(u[0] - cam["c_row"]) / f])
    s0 /= np.linalg.norm(s0)
    s = Rotation.from_rotvec(-theta).apply(s0)
    px = np.stack([cam["c_row"] - f * s[:, 2] / s[:, 0], cam["c_col"] - f * s[:, 1] / s[:, 0]], -1)
    p = distort(px, cam)
    p0 = distort(np.array([[cam["c_row"] - f * s0[2] / s0[0], cam["c_col"] - f * s0[1] / s0[0]]]), cam)[0]
    return p - p0


def integrate(tg, wg, t_mid):
    """Trapezoid integral of the rate on the grid tg, zero at t_mid."""
    c = np.vstack([np.zeros((1, 3)), np.cumsum(0.5 * (wg[1:] + wg[:-1]) * np.diff(tg)[:, None], 0)])
    return c - np.array([np.interp(t_mid, tg, c[:, j]) for j in range(3)])


# ---------------------------------------------------------------- motion
def draw_motion(win, cfg, rng, rc, errors=True, sampling="real", synthetic=None):
    """One sample: true and measured paths of a star at pixel rc.
    errors: apply the 4.2 error model; sampling: "real" (run intervals) or
    "dense" (1 ms, random phase) or ("uniform", hz, jitter): regular samples
    at hz with a relative gaussian jitter of the intervals (other IMUs, 4.4). synthetic: None = draw from the config share,
    or a rate factor. Output: dict with d_true, d_meas (T, 2) px over the
    exposure (uniform time steps), the drawn parameters, centre_length_px."""
    cam, ge, dt = cfg["camera"], cfg["gyro_errors"], cfg["path_dt_s"]
    half = cam["exposure_s"] / 2
    if synthetic is None:
        sh = cfg["synthetic_share"]
        k = rng.uniform(*sh["rate_factor"]) if rng.random() < sh["fraction"] else 1.0
    else:
        k = synthetic
    spl = PchipInterpolator(win["t"], win["w"])
    rate = lambda tt: k * spl(k * tt)
    rs = cfg["rolling_shutter"]
    frac = rc[0] / cam["height"]
    ro_true = rng.uniform(*rs["readout_true_s"]) if errors else rs["readout_nominal_s"]
    tc_true = frac * ro_true
    tc_meas = frac * rs["readout_nominal_s"]
    # true path
    tg = tc_true + np.arange(-half, half + dt / 2, dt)
    th = integrate(tg, rate(tg), tc_true)
    d_true = project_path(th, rc, cam)
    vt = cfg["true_motion"]
    vib = {"amp_px": 0.0}
    d_true_novib = d_true
    if errors:
        amp, hz = rng.uniform(*vt["vibration_px"]), rng.uniform(*vt["vibration_hz"])
        ang, ph = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
        osc = amp * np.sin(2 * np.pi * hz * (tg - tc_true) + ph)
        d_true = d_true + osc[:, None] * [np.sin(ang), np.cos(ang)]
        vib = {"amp_px": amp, "hz": hz}
    centre_len = cam["f_px"] * np.sum(np.linalg.norm(0.5 * (rate(tg[1:]) + rate(tg[:-1]))[:, 1:], axis=1)) * dt
    # measured samples
    lo, hi = tc_meas - half - 0.05, tc_meas + half + 0.05
    if sampling == "dense":
        ts = lo + rng.uniform(0, 0.001) + np.arange(0, hi - lo, 0.001)
    elif isinstance(sampling, (tuple, list)):
        _, hz, jit = sampling
        steps = np.clip(1 / hz * (1 + jit * rng.standard_normal(int((hi - lo + 0.1) * hz) + 3)), 0.2 / hz, None)
        ts = lo - rng.uniform(0, 1 / hz) + np.concatenate([[0], np.cumsum(steps)])
    else:
        j = rng.integers(0, len(win["dt"]) - 200)
        steps = win["dt"][j:j + 200]
        ts = lo - rng.uniform(0, steps[0]) + np.concatenate([[0], np.cumsum(steps)])
        ts = ts[ts <= hi + 0.05]
    par = {"k": k, "readout_s": ro_true, "row": float(rc[0]), "col": float(rc[1]), "vib": vib}
    if errors:
        tau = (rng.uniform(-1, 1) * ge["tau_outlier_range_s"] if rng.random() < ge["tau_outlier_fraction"]
               else rng.normal(0, ge["tau_sigma_s"]))
        off = np.radians(rng.normal(0, ge["offset_residual_deg_s"], 3))
        sig = np.radians(ge["white_noise_deg_s"] * rng.uniform(*ge["white_noise_factor"]))
        scale = 1 + rng.normal(0, ge["scale_error"])
        mis = Rotation.from_rotvec(np.radians(rng.normal(0, ge["misalignment_deg"], 3))).as_matrix()
        # sample k is labelled ts[k] but holds the true rate at ts[k] + tau
        wm = scale * rate(ts + tau) @ mis.T + off + rng.normal(0, sig, (len(ts), 3))
        cam_m = dict(cam, f_px=cam["f_px"] * (1 + rng.normal(0, ge["f_error"])),
                     c_row=cam["c_row"] + rng.normal(0, ge["centre_error_px"]),
                     c_col=cam["c_col"] + rng.normal(0, ge["centre_error_px"]))
        par.update({"tau_s": tau, "noise_deg_s": float(np.degrees(sig)), "scale": scale})
    else:
        wm = rate(ts)
        cam_m = cam
    tgm = tc_meas + np.arange(-half, half + dt / 2, dt)
    wgm = np.stack([np.interp(tgm, ts, wm[:, j]) for j in range(3)], -1)
    d_meas = project_path(integrate(tgm, wgm, tc_meas), rc, cam_m)
    return {"d_true": d_true, "d_meas": d_meas, "d_true_novib": d_true_novib, "par": par,
            "centre_length_px": float(centre_len)}


# ---------------------------------------------------------------- channels
def channels(d, cfg):
    """Channels 2 (dwell, sum 1) and 3 (time ramp) of a path d (T, 2) px,
    mid-time point at the patch centre. Points outside the patch are lost
    (the caller rejects such samples with max_extent_px)."""
    n, c = cfg["patch"]["size"], cfg["patch"]["centre"]
    pos = d + c - 0.5                      # pixel-index coords (pixel i centre at i)
    tfrac = np.linspace(0, 1, len(d))
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    ch2 = np.zeros((n, n))
    ch3 = np.zeros((n, n))
    for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        wgt = (fr[:, 0] if dr else 1 - fr[:, 0]) * (fr[:, 1] if dc else 1 - fr[:, 1]) / len(d)
        r, cc = i0[:, 0] + dr, i0[:, 1] + dc
        ok = (r >= 0) & (r < n) & (cc >= 0) & (cc < n)
        np.add.at(ch2, (r[ok], cc[ok]), wgt[ok])
        np.add.at(ch3, (r[ok], cc[ok]), wgt[ok] * tfrac[ok])
    ch3 = np.where(ch2 > 0, ch3 / np.maximum(ch2, 1e-12), 0.0)
    return ch2, ch3


def psf_pixel(du, dv, a, p, sub=5):
    """Pixel-integrated round super-Gaussian (peak 1) at offsets (du, dv)."""
    o = (np.arange(sub) + 0.5) / sub - 0.5
    acc = 0.0
    for oy in o:
        for ox in o:
            q = ((du + oy) ** 2 + (dv + ox) ** 2) / a ** 2
            acc = acc + np.exp(-0.5 * q ** p)
    return acc / sub ** 2


def render_trail(d, label, cfg, a=2.5, p=1.3):
    """Noise-free trail on the patch, total light 1 (direct sum of the
    pixel-integrated PSF over the path points). label = patch position of the
    mid-time point (corner coords)."""
    n = cfg["patch"]["size"]
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    img = np.zeros((n, n))
    for q in d:
        img += psf_pixel(yy - label[0] - q[0], xx - label[1] - q[1], a, p)
    return img / img.sum()


def conv_channel(ch2, cfg, a=2.5, p=1.3):
    """PSF convolved with channel 2 (pixel kernel), total 1."""
    n = cfg["patch"]["size"]
    from scipy.signal import fftconvolve
    r = np.arange(-12, 13)
    ker = psf_pixel(r[:, None] + 0.0 * r[None], r[None] + 0.0 * r[:, None], a, p)
    img = fftconvolve(ch2, ker, mode="same")
    return img / img.sum()


# ---------------------------------------------------------------- checks
def path_err(s, key="d_true"):
    e = np.linalg.norm(s["d_meas"] - s[key], axis=1)
    return float(np.sqrt(np.mean(e ** 2))), float(e.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    cfg = CFG
    rng = np.random.default_rng(cfg["seed"])
    cam = cfg["camera"]
    lines, res = [], {}

    def say(text):
        print(text)
        lines.append(text)

    wins = {}
    for run in cfg["windows"]["runs"]:
        wins[run] = load_windows(cfg, run, args.n // len(cfg["windows"]["runs"]), rng)
        say(f"windows {run}: {len(wins[run])}")
    allw = [(run, w) for run in wins for w in wins[run]]

    def rc_draw():
        return np.array([rng.uniform(20, cam["height"] - 20), rng.uniform(20, cam["width"] - 20)])

    # C1 zero error, dense sampling
    e1 = [path_err(draw_motion(w, cfg, rng, rc_draw(), errors=False, sampling="dense", synthetic=1.0))
          for _, w in allw[:300]]
    e1 = np.array(e1)
    res["C1_dense_zero_error_max_px"] = float(e1[:, 1].max())
    say(f"C1 zero error, dense sampling (300): max path error {e1[:, 1].max():.2e} px")
    # C2 trail vs PSF * channel 2
    c2 = []
    for _, w in allw[:100]:
        s = draw_motion(w, cfg, rng, rc_draw(), errors=False, sampling="dense", synthetic=1.0)
        if np.abs(s["d_true"]).max() > cfg["patch"]["max_extent_px"] - 4:
            continue
        ch2, _ = channels(s["d_meas"], cfg)
        lab = np.array([cfg["patch"]["centre"]] * 2)
        t_img = render_trail(s["d_true"], lab, cfg)
        c_img = conv_channel(ch2, cfg)
        yy, xx = np.mgrid[0:32, 0:32] + 0.5
        cen = lambda im: np.array([np.sum(im * yy), np.sum(im * xx)])
        c2.append([np.abs(t_img - c_img).max() / t_img.max(), np.linalg.norm(cen(t_img) - cen(c_img))])
    c2 = np.array(c2)
    res["C2"] = {"n": len(c2), "max_diff_over_peak_p50_max": [float(np.median(c2[:, 0])), float(c2[:, 0].max())],
                 "centroid_diff_px_max": float(c2[:, 1].max())}
    say(f"C2 trail vs PSF * ch2 (n {len(c2)}): |diff|/peak median {np.median(c2[:, 0]):.4f}, "
        f"max {c2[:, 0].max():.4f}; light centroid diff max {c2[:, 1].max():.4f} px")
    # C3, C4, C5 on all windows
    rows = []
    for run, w in allw:
        rc = rc_draw()
        s3 = draw_motion(w, cfg, rng, rc, errors=False, sampling="real", synthetic=1.0)
        s4 = draw_motion(w, cfg, rng, rc, errors=True, sampling="real")
        ext = np.abs(s4["d_meas"]).max()
        L4 = float(np.linalg.norm(np.diff(s4["d_true"], axis=0), axis=1).sum())
        rows.append({"run": run, "len_real": s3["centre_length_px"], "len_all": s4["centre_length_px"],
                     "k": s4["par"]["k"], "e3": path_err(s3), "e4": path_err(s4, "d_true_novib"),
                     "e4v": path_err(s4), "ext": float(ext), "tau": s4["par"]["tau_s"]})
    acc = np.array([r["ext"] <= cfg["patch"]["max_extent_px"] for r in rows])
    e3 = np.array([r["e3"][0] for r in rows])[acc]
    e4 = np.array([r["e4"][0] for r in rows])[acc]
    e4v = np.array([r["e4v"][0] for r in rows])[acc]
    L = np.array([r["len_all"] for r in rows])[acc]
    tau = np.array([r["tau"] for r in rows])[acc]
    q = lambda x: np.percentile(x, [50, 90, 99]).round(3).tolist()
    res["C3_sampling_rms_px_p50_p90_p99"] = q(e3)
    res["C4_gyro_errors_rms_px_p50_p90_p99"] = q(e4)
    res["C4_with_vibration_rms_px_p50_p90_p99"] = q(e4v)
    say(f"accepted samples (max extent <= {cfg['patch']['max_extent_px']} px): {acc.sum()} of {len(rows)}")
    say(f"C3 sampling only, rms path error p50/p90/p99: {q(e3)} px")
    say(f"C4 gyro error model, rms path error p50/p90/p99: {q(e4)} px")
    say(f"C4 plus true-only vibration, p50/p90/p99: {q(e4v)} px")
    res["C4_by_centre_length"] = {}
    for lo, hi in [(0, 3), (3, 10), (10, 1e9)]:
        m = (L >= lo) & (L < hi)
        if m.sum() > 10:
            res["C4_by_centre_length"][f"{lo}-{hi}"] = {"n": int(m.sum()), "C3_p50_p90": np.percentile(e3[m], [50, 90]).tolist(),
                                                     "C4_p50_p90": np.percentile(e4[m], [50, 90]).tolist()}
            say(f"   centre length {lo}-{hi if hi < 1e9 else 'inf'} px (n {m.sum()}): C3 p50/p90 "
                f"{np.percentile(e3[m], [50, 90]).round(3).tolist()}, C4 p50/p90 {np.percentile(e4[m], [50, 90]).round(3).tolist()}")
    for name, m in [("|tau| < 5 ms", np.abs(tau) < 0.005), ("|tau| > 20 ms", np.abs(tau) > 0.02)]:
        if m.sum() > 5:
            say(f"   C4 {name} (n {m.sum()}): p50/p90 {np.percentile(e4[m], [50, 90]).round(3).tolist()}")
    gy = json.load(open(OUT / "gyro.json"))
    res["C5"] = {}
    for run in cfg["windows"]["runs"]:
        lr = np.array([r["len_real"] for r in rows if r["run"] == run])
        la = np.array([r["len_all"] for r in rows if r["run"] == run])
        real = gy[run]["centre_length_px_p10_p50_p90_p99"]
        mine = np.percentile(lr, [10, 50, 90, 99]).round(2).tolist()
        withs = np.percentile(la, [10, 50, 90, 99]).round(2).tolist()
        res["C5"][run] = {"gyro_json": real, "real_share": mine, "with_synthetic": withs}
        say(f"C5 {run[:10]} centre length p10/p50/p90/p99: gyro.json {real} | samples k=1 {mine} | with synthetic share {withs}")
    ext = np.array([r["ext"] for r in rows])
    rej = float(np.mean(ext > cfg["patch"]["max_extent_px"]))
    res["C5_rejected_frac"] = rej
    say(f"C5 rejected by max_extent {cfg['patch']['max_extent_px']} px: {rej:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "motion_checks.json", "w"), indent=1)
    open(OUT / "motion_checks.txt", "w").write("\n".join(lines) + "\n")
    plot(allw, cfg, rng, rc_draw)


def plot(allw, cfg, rng, rc_draw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(6, 6, figsize=(12, 12))
    picks, j = [], 0
    while len(picks) < 12 and j < len(allw):
        s = draw_motion(allw[j][1], cfg, rng, rc_draw(), synthetic=None)
        j += 1
        if np.abs(s["d_meas"]).max() < cfg["patch"]["max_extent_px"] and \
                np.linalg.norm(np.diff(s["d_true"], axis=0), axis=1).sum() > 2:
            picks.append(s)
    lab = np.array([cfg["patch"]["centre"]] * 2)
    for i, s in enumerate(picks):
        ch2, ch3 = channels(s["d_meas"], cfg)
        a = ax.ravel()[3 * i:3 * i + 3]
        a[0].imshow(render_trail(s["d_true"], lab, cfg), cmap="gray")
        a[1].imshow(ch2, cmap="gray")
        a[2].imshow(np.ma.masked_where(ch2 == 0, ch3), cmap="viridis", vmin=0, vmax=1)
        a[0].set_title(f"true k{s['par']['k']:.1f} tau {s['par']['tau_s']*1e3:.0f}ms", fontsize=7)
        a[1].set_title("ch2 measured", fontsize=7)
        a[2].set_title("ch3 time", fontsize=7)
        for x in a:
            x.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "motion_samples.png", dpi=80)


if __name__ == "__main__":
    main()
