"""Step 5: the dataset generator (guide step 5, sections 2, 4.1, 4.2, 4.4).

Written by Claude, 29 Oct 2026. Run from the centroid_cnn folder:
    python src/generator.py --name v1 --train 100000 --val 10000 --test 10000
Writes datasets/<name>/<split>/shard_NNN.npz (X float16 (n, 3, 32, 32),
y float32 (n, 2), one array per drawn parameter) and datasets/<name>/meta.json
(all configs copied, seeds, window ranges). Same seed -> bit-identical data,
whatever the number of workers (one RNG per sample: (seed, split, index)).

Parameters: configs/step5.json (this file's ranges and mixture), with the
measured instrument values from configs/step3.json and configs/step4.json.
Each parameter group draws either THIS instrument (p_instrument) or the wide
guide 4.4 ranges, independently.

Pipeline per sample (CONVENTIONS, configs/step3.json pipeline_order):
 1. frame position (row, col) uniform: picks the instrument PSF and noise
    ring and drives the motion projection; NOT given to the network.
 2. motion (src/motion.py): true path (renders the light) and measured path
    (channels 2, 3) from a real gyro window of the split.
 3. (v2) the whole padded field is rendered with noise, and a cedar-like
    detector (smooth, threshold, blob COM) finds the target; patch centre
    pixel = floor(coarse centroid); an undetected target, or a patch that
    belongs to a brighter closer neighbour, is redrawn. label = true mid-time
    position - (16.5, 16.5). (v1 used light centre + N(0, 0.7 px): a far too
    good prior at low SNR and in blends, step 6.)
 4. linear render on a 5x supersampled padded field: dwell map of the true
    path (target + neighbours) convolved with the PSF, summed per pixel.
 5. + hot pixels, artefacts, background (+ slope), skewed correlated
    background noise, shot noise; tone curve; round, clip, bit depth.
 6. channel 1 = (Y - median of the outer 3-px frame) / sigma_frame.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, label
from scipy.signal import fftconvolve

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import motion as M  # noqa: E402

CFG5 = json.load(open(HERE / "configs" / "step5.json"))
CFG3 = json.load(open(HERE / "configs" / "step3.json"))
CFG4 = json.load(open(HERE / "configs" / "step4.json"))

_g = CFG3["tone"]["gamma_curve_16bit_xy"]
GX, GY = np.array(_g[0::2], float), np.array(_g[1::2], float)
G_SCALE = CFG3["tone"]["lin_to_y8_scale"]
CE_KNEE, CE_GAIN = CFG3["tone"]["ce_knee_lin"], CFG3["tone"]["ce_gain"]
HOT_T = np.array(CFG3["hot_pixels"]["template_5x5"])
CELLS = CFG3["psf"]["cells_p10_p50_p90"]
BG = CFG3["background"]
C1E, C1V = BG["var_vs_level_per_ring"]["r_edges_px"], BG["var_vs_level_per_ring"]["c1_dn2_per_dn"]
CAM = CFG4["camera"]
PSF_CODES = {"super_gaussian": 0, "gaussian": 1, "moffat": 2, "ring": 3}
TONE_CODES = {"instrument": 0, "linear": 1, "rational": 2, "s_curve": 3}
WINDOWS = {}                                   # split -> list of (run index, window)


# ---------------------------------------------------------------- helpers
def q_draw(rng, p10, p50, p90):
    """Draw from the piecewise-linear quantile function through p10/p50/p90
    (extrapolated linearly to the 0 and 1 quantiles)."""
    u = rng.random()
    if u < 0.5:
        return p10 + (u - 0.1) * (p50 - p10) / 0.4
    return p50 + (u - 0.5) * (p90 - p50) / 0.4


def ring_of(r, edges):
    return int(np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(edges) - 2))


def gauss_sigma_eq(psf):
    """Per-axis Gaussian sigma with the same FWHM (CONVENTIONS SNR), geometric mean."""
    m, a1, a2 = psf["model"], psf["a1"], psf["a2"]
    if m in ("super_gaussian", "gaussian"):
        p = psf.get("p", 1.0)
        f = (2 * np.log(2)) ** (1 / (2 * p)) / np.sqrt(2 * np.log(2))
        return np.sqrt(a1 * a2) * f
    if m == "moffat":
        fwhm = 2 * np.sqrt(2 ** (1 / psf["beta"]) - 1)
        return np.sqrt(a1 * a2) * fwhm / 2.3548
    return np.sqrt(psf["r0"] ** 2 + psf["w"] ** 2)


def psf_fine(psf, S, R):
    """PSF on the fine grid (cell centres), sum 1. Size (2 R S + 1)^2."""
    o = (np.arange(2 * R * S + 1) - R * S) / S
    dr, dc = np.meshgrid(o, o, indexing="ij")

    def shape(dr, dc, k=1.0):
        th = np.radians(psf["angle"])
        u = dr * np.sin(th) + dc * np.cos(th)          # along the major axis
        v = dr * np.cos(th) - dc * np.sin(th)
        m = psf["model"]
        if m in ("super_gaussian", "gaussian"):
            q = (u / (k * psf["a1"])) ** 2 + (v / (k * psf["a2"])) ** 2
            return np.exp(-0.5 * q ** psf.get("p", 1.0))
        if m == "moffat":
            q = (u / (k * psf["a1"])) ** 2 + (v / (k * psf["a2"])) ** 2
            return (1 + q) ** (-psf["beta"])
        r = np.hypot(dr, dc)
        return np.exp(-0.5 * ((r - k * psf["r0"]) / (k * psf["w"])) ** 2)

    ker = shape(dr, dc)
    if psf["coma_frac"] > 0:
        a = psf["coma_frac"] * shape(dr - psf["coma_dr"], dc - psf["coma_dc"], psf["coma_k"])
        ker = (ker / ker.sum()) * (1 - psf["coma_frac"]) + a / a.sum() * psf["coma_frac"]
    return ker / ker.sum()


def tone(x, kind, par):
    """Tone curve on x = lin / 255 (x >= 0). Output 0..1 (before scaling)."""
    if kind == "instrument":
        lin = 255 * x
        lin = np.where(lin > CE_KNEE, CE_KNEE + CE_GAIN * (lin - CE_KNEE), lin)
        return np.interp(lin / G_SCALE, GX, GY) / 256 / 255
    if kind == "linear":
        return x
    if kind == "rational":
        return x * (1 + par) / (1 + par * x)
    return 0.5 * (1 + np.tanh(par * (x - 0.5)) / np.tanh(par / 2))


def corr_field(rng, n, a, b):
    """Unit-variance noise field (n, n) with the separable kernel [b, a, 1, a, b]."""
    k = np.array([b, a, 1.0, a, b])
    k /= np.sqrt(np.sum(k ** 2))
    w = rng.standard_normal((n + 4, n + 4))
    w = np.apply_along_axis(lambda x: np.convolve(x, k, "valid"), 0, w)
    return np.apply_along_axis(lambda x: np.convolve(x, k, "valid"), 1, w)


def skew(z, s):
    if s < 1e-6:
        return z
    m, d = np.exp(s * s / 2), np.sqrt((np.exp(s * s) - 1) * np.exp(s * s))
    return (np.exp(s * z) - m) / d


def deposit(fine, pos, wgt):
    """Bilinear add of weights at fine-grid index coords pos (n, 2)."""
    i0 = np.floor(pos).astype(int)
    fr = pos - i0
    n = fine.shape[0]
    for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        w = wgt * (fr[:, 0] if dr else 1 - fr[:, 0]) * (fr[:, 1] if dc else 1 - fr[:, 1])
        r, c = i0[:, 0] + dr, i0[:, 1] + dc
        ok = (r >= 0) & (r < n) & (c >= 0) & (c < n)
        np.add.at(fine, (r[ok], c[ok]), w[ok])


# ---------------------------------------------------------------- draws
def draw_psf(rng, P, r, inst):
    c = CFG5["psf"]
    if inst:
        rings = sorted({tuple(x["r_px"]) for x in CELLS})
        ring = [x for x in rings if x[0] <= r < x[1]] or [rings[-1]]
        cells = [x for x in CELLS if tuple(x["r_px"]) == ring[0]]
        cell = cells[rng.integers(len(cells))]
        f = rng.uniform(*c["instrument"]["defocus_factor"])
        a1 = max(q_draw(rng, *cell["a1"]), 0.8) * f
        a2 = max(q_draw(rng, *cell["a2"]), 0.8) * f
        a1, a2 = max(a1, a2), min(a1, a2)
        p = float(np.clip(q_draw(rng, *cell["p"]), 0.5, 4.0))
        if r > c["instrument"]["tangential_r_px"]:
            ang = np.degrees(np.arctan2(P["row"] - CAM["c_row"], P["col"] - CAM["c_col"])) + 90 \
                + rng.normal(0, c["instrument"]["tangential_spread_deg"])
        else:
            ang = rng.uniform(0, 180)
        psf = {"model": "super_gaussian", "a1": a1, "a2": a2, "p": p, "angle": ang, "coma_frac": 0.0}
    else:
        cr = c["random"]
        names = list(cr["p_model"])
        m = names[rng.choice(len(names), p=list(cr["p_model"].values()))]
        w = np.exp(rng.uniform(*np.log(cr["width_px"])))
        q = rng.uniform(*cr["axis_ratio"])
        psf = {"model": m, "a1": w * np.sqrt(q), "a2": w / np.sqrt(q), "angle": rng.uniform(0, 180), "coma_frac": 0.0}
        if m == "super_gaussian":
            psf["p"] = rng.uniform(*cr["sg_p"])
        elif m == "moffat":
            psf["beta"] = rng.uniform(*cr["moffat_beta"])
            # alpha so that the FWHM equals that of a Gaussian of sigma w (per axis)
            fw = 2 * np.sqrt(2 ** (1 / psf["beta"]) - 1)
            psf["a1"], psf["a2"] = psf["a1"] * 2.3548 / fw, psf["a2"] * 2.3548 / fw
        elif m == "ring":
            psf["r0"], psf["w"] = rng.uniform(*cr["ring_r0_px"]), rng.uniform(*cr["ring_w_px"])
        if rng.random() < cr["p_coma"]:
            th = rng.uniform(0, 2 * np.pi)
            off = rng.uniform(*cr["coma_offset_in_widths"]) * w
            psf.update({"coma_frac": rng.uniform(*cr["coma_fraction"]), "coma_dr": off * np.sin(th),
                        "coma_dc": off * np.cos(th), "coma_k": cr["coma_width_factor"]})
    return psf


def draw_background(rng, r, inst):
    if inst:
        night = list(BG["per_night_per_ring"])[rng.integers(3)]
        rings = BG["per_night_per_ring"][night]
        ring = [x for x in rings if x["r_px"][0] <= r < x["r_px"][1]] or [rings[-1]]
        b = {"mu": ring[0]["mu"], "sigma": ring[0]["sigma"], "s": ring[0]["s"],
             "ka": BG["kernel_a"], "kb": BG["kernel_b"], "c1": C1V[ring_of(r, C1E)]}
    else:
        cr = CFG5["background"]["random"]
        a = rng.uniform(*cr["kernel_a"])
        b = {"mu": rng.uniform(*cr["mu"]), "sigma": rng.uniform(*cr["sigma"]), "s": rng.uniform(*cr["s"]),
             "ka": a, "kb": a * rng.uniform(*cr["kernel_b_over_a"]), "c1": rng.uniform(*cr["c1"])}
    cb = CFG5["background"]
    b["slope"] = rng.uniform(*cb["slope_dn_per_patch"]) if rng.random() < cb["p_slope"] else 0.0
    b["slope_dir"] = rng.uniform(0, 2 * np.pi)
    return b


def motion_cfg(rng, inst):
    """Per-sample motion config and sampling (instrument = configs/step4.json)."""
    if inst:
        return CFG4, "real", 0.0
    cr = CFG5["motion"]["random"]
    ro = rng.uniform(*cr["readout_s"])
    ge = dict(CFG4["gyro_errors"], tau_sigma_s=cr["tau_sigma_s"], tau_outlier_fraction=cr["tau_outlier_fraction"],
              tau_outlier_range_s=cr["tau_outlier_range_s"], offset_residual_deg_s=cr["offset_residual_deg_s"],
              white_noise_factor=[1.0, 1.0], scale_error=cr["scale_error"], misalignment_deg=cr["misalignment_deg"])
    ge["white_noise_deg_s"] = CFG4["gyro_errors"]["white_noise_deg_s"] * np.exp(rng.uniform(*np.log(cr["white_noise_factor_log"])))
    cfg = dict(CFG4, gyro_errors=ge,
               rolling_shutter={"readout_true_s": [ro, ro],
                                "readout_nominal_s": max(ro + rng.normal(0, cr["readout_error_sigma_s"]), 0.0)},
               true_motion=dict(CFG4["true_motion"], vibration_px=cr["vibration_px"]))
    if rng.random() < cr["p_uniform_sampling"]:
        hz = float(np.exp(rng.uniform(*np.log(cr["sampling_hz_log"]))))
        return cfg, ("uniform", hz, rng.uniform(*cr["sampling_jitter"])), hz
    return cfg, "real", 0.0


# ---------------------------------------------------------------- one sample
def make_sample(split, index, zero_error=False, noise_free=False, inst_all=False, flux_lin=None, clean=False):
    """One sample. Output: X (3, 32, 32) float32, y (2,), params dict, extras.
    Check options: zero_error (no gyro error, dense sampling), noise_free (no
    noise, no rounding), inst_all (all groups = instrument), flux_lin (fixed
    flux), clean (no neighbours, hot pixels or artefacts)."""
    rng = np.random.default_rng([CFG5["seed"], ["train", "val", "test", "check"].index(split), index])
    why = {"undetected": 0, "owned": 0, "off_field": 0}
    for _ in range(CFG5["detector"]["max_attempts"]):
        out, reason = _attempt(rng, split, zero_error, noise_free, inst_all, flux_lin, clean)
        if out is not None:
            out[2].update({f"rej_{k}": v for k, v in why.items()})
            return out
        why[reason] += 1
    raise RuntimeError(f"no detected sample after {CFG5['detector']['max_attempts']} attempts ({why})")


def _attempt(rng, split, zero_error, noise_free, inst_all, flux_lin, clean):
    """One try of make_sample. Output: ((X, y, P, extras), None) or (None, reason)."""
    pc = CFG5["patch"]
    N, S, PAD, R = pc["size"], pc["supersample"], pc["pad_px"], pc["psf_radius_px"]
    pg = CFG5.get("p_instrument_groups", {})            # step 8 test sets: per-group override
    inst = (lambda g: True) if inst_all else (lambda g: rng.random() < pg.get(g, CFG5["p_instrument"]))
    P = {"row": rng.uniform(20, CAM["height"] - 20), "col": rng.uniform(20, CAM["width"] - 20)}
    r = float(np.hypot(P["row"] - CAM["c_row"], P["col"] - CAM["c_col"]))
    P["r_px"] = r
    # draws
    P["psf_inst"] = float(inst("psf"))
    psf = draw_psf(rng, P, r, P["psf_inst"] > 0)
    P["bg_inst"] = float(inst("bg"))
    bg = draw_background(rng, r, P["bg_inst"] > 0)
    P["tone_inst"] = float(inst("tone"))
    if P["tone_inst"]:
        tkind, tpar, depth = "instrument", 0.0, 8
    else:
        tr = CFG5["tone"]["random"]
        names = list(tr["p_kind"])
        tkind = names[rng.choice(len(names), p=list(tr["p_kind"].values()))]
        tpar = rng.uniform(*tr["rational_c"]) if tkind == "rational" else (
            rng.uniform(*tr["s_curve_k"]) if tkind == "s_curve" else 0.0)
        bd = CFG5["bit_depth"]["random"]
        depth = int(list(bd)[rng.choice(len(bd), p=list(bd.values()))])
    fx = CFG5["flux"]
    snr = (np.exp(rng.uniform(*np.log(fx["bright_snr"]))) if rng.random() < fx["p_bright"]
           else np.exp(rng.uniform(*np.log(fx["snr_log_uniform"]))))
    s_eff = gauss_sigma_eq(psf)
    F = flux_lin if flux_lin is not None else snr * bg["sigma"] * np.sqrt(2 * np.pi) * s_eff
    # motion, rejection of samples that leave the patch
    P["motion_inst"] = float(inst("motion"))
    mcfg, sampling, hz = motion_cfg(rng, P["motion_inst"] > 0)
    if zero_error:
        sampling = "dense"
    wins = WINDOWS[split]
    for tries in range(CFG5["motion"]["max_resample"]):
        wi = int(rng.integers(len(wins)))
        mo = M.draw_motion(wins[wi][1], mcfg, rng, np.array([P["row"], P["col"]]),
                           errors=not zero_error, sampling=sampling)
        if np.abs(mo["d_meas"]).max() <= CFG4["patch"]["max_extent_px"] and \
                np.abs(mo["d_true"]).max() <= CFG4["patch"]["max_extent_px"]:
            break
    d_true, d_meas = mo["d_true"], mo["d_meas"]
    # target on the padded field (field coords, corner convention)
    Wf = N + 2 * PAD
    lc = d_true.mean(0)                                        # light centre - mid-time point
    light_f = Wf / 2 + rng.random(2)                           # uniform sub-pixel phase
    mid_f = light_f - lc
    # linear render of the whole field (units: fraction of the target flux)
    fine = np.zeros((Wf * S, Wf * S))
    T = len(d_true)
    to_fine = lambda pts: pts * S - 0.5
    deposit(fine, to_fine(mid_f + d_true), np.full(T, 1.0 / T))
    nb = CFG5["neighbours"]
    lam = rng.uniform(*nb["mean_per_patch"]) * Wf ** 2 / N ** 2
    n_nb = 0 if clean else int(rng.poisson(lam))
    nb_pos = [rng.uniform(0, Wf, 2) for _ in range(n_nb)]
    nb_rat = [float(np.exp(rng.uniform(*np.log(nb["flux_ratio_log_uniform"])))) for _ in range(n_nb)]
    hard = (rng.random() < nb["p_hard"]) and not clean
    if hard:
        th = rng.uniform(0, 2 * np.pi)
        nb_pos.append(mid_f + rng.uniform(*nb["hard_dist_px"]) * np.array([np.sin(th), np.cos(th)]))
        nb_rat.append(rng.uniform(*nb["hard_flux_ratio"]))
    for pos, rat in zip(nb_pos, nb_rat):
        deposit(fine, to_fine(pos + d_true), np.full(T, rat / T))
    s_ax = max(psf["a1"], psf["a2"]) * s_eff / np.sqrt(psf["a1"] * psf["a2"])  # widest axis, Gaussian-eq sigma
    R = int(np.clip(np.ceil(4 * s_ax + (psf.get("r0", 0) if psf["model"] == "ring" else 0)), R, 30))
    img = fftconvolve(fine, psf_fine(psf, S, R), mode="same")
    star_f = F * img.reshape(Wf, S, Wf, S).sum((1, 3))
    # hot pixels and artefacts (linear units)
    extra = np.zeros((Wf, Wf))
    hp = CFG5["hot_pixels"]
    dens = rng.uniform(*(hp["instrument_per_mpx"] if P["bg_inst"] else hp["random_per_mpx"]))
    n_hot = 0 if clean else int(rng.poisson(dens * Wf * Wf / 1e6))
    for _ in range(n_hot):
        rr, cc = rng.integers(0, Wf, 2)
        pk = rng.uniform(*(hp["instrument_peak_dn"] if P["bg_inst"] else hp["random_peak_dn"]))
        for i in range(5):
            for j in range(5):
                if 0 <= rr + i - 2 < Wf and 0 <= cc + j - 2 < Wf:
                    extra[rr + i - 2, cc + j - 2] += pk * HOT_T[i, j]
    ar = CFG5["artefacts"]
    yy, xx = np.mgrid[0:Wf, 0:Wf] + 0.5
    streak = (rng.random() < ar["p_streak"]) and not clean
    if streak:
        th, p0 = rng.uniform(0, np.pi), rng.uniform(0, Wf, 2)
        dist = (yy - p0[0]) * np.cos(th) - (xx - p0[1]) * np.sin(th)
        extra += rng.uniform(*ar["streak_snr"]) * bg["sigma"] * np.exp(-0.5 * (dist / rng.uniform(*ar["streak_sigma_px"])) ** 2)
    cosmic = (rng.random() < ar["p_cosmic"]) and not clean
    if cosmic:
        rr, cc = rng.integers(2, Wf - 2, 2)
        dr, dc = rng.integers(-1, 2, 2)
        for k in range(int(rng.integers(ar["cosmic_len_px"][0], ar["cosmic_len_px"][1] + 1))):
            if 0 <= rr + k * dr < Wf and 0 <= cc + k * dc < Wf:
                extra[rr + k * dr, cc + k * dc] += rng.uniform(*ar["cosmic_dn"])
    # background, noise, tone, quantization (whole field)
    slope = bg["slope"] * ((yy - Wf / 2) * np.sin(bg["slope_dir"]) + (xx - Wf / 2) * np.cos(bg["slope_dir"])) / N
    signal = star_f + extra
    full = 2 ** depth - 1

    def to_y(lin):
        return np.clip(np.round(tone(np.maximum(lin, 0) / 255, tkind, tpar) * full), 0, full)

    if noise_free:
        Yf = tone(np.maximum(bg["mu"] + slope + signal, 0) / 255, tkind, tpar) * full
    else:
        z1 = corr_field(rng, Wf, bg["ka"], bg["kb"])
        z2 = corr_field(rng, Wf, bg["ka"], bg["kb"])
        lin = bg["mu"] + slope + signal + bg["sigma"] * skew(z1, bg["s"]) \
            + np.sqrt(bg["c1"] * np.maximum(signal, 0)) * z2
        Yf = to_y(lin)
    # sigma_frame: empty 64 x 64 patch with the same background, tone, depth
    ze = corr_field(rng, 64, bg["ka"], bg["kb"])
    Ye = to_y(bg["mu"] + bg["sigma"] * skew(ze, bg["s"]))
    sigma_frame = max(float(Ye.std()), 1 / np.sqrt(12))
    # detector (guide 4.1.1): where the production code would cut the patch
    dc_ = CFG5["detector"]
    if noise_free or zero_error:                # check modes: v1 placement, label tests only
        coarse = light_f + rng.normal(0, pc["detector_sigma_px"], 2)
        det_snr = float("nan")
    else:
        sm = gaussian_filter(Yf - np.median(Yf), dc_["smooth_sigma_px"])
        sm_e = gaussian_filter(Ye - np.median(Ye), dc_["smooth_sigma_px"])
        thr = dc_["threshold_sigma"] * max(float(sm_e.std()), 1e-6)
        lab, nl = label(sm > thr)
        coarse = None
        if nl:
            idx = np.arange(1, nl + 1)
            w = np.clip(Yf - np.median(Yf), 0, None)
            coms = np.array(center_of_mass(w + 1e-12, lab, idx)) + 0.5     # corner coords
            dist = np.hypot(*(coms - light_f).T)
            j = int(np.argmin(dist))
            if dist[j] <= dc_["assoc_px"] or lab[int(light_f[0]), int(light_f[1])] == j + 1:
                coarse = coms[j]
        if coarse is None:
            return None, "undetected"
        d_t = np.hypot(*(coarse - light_f))
        for pos, rat in zip(nb_pos, nb_rat):     # the patch belongs to a brighter, closer star
            if rat >= dc_["owner_flux_ratio"] and np.hypot(*(coarse - (pos + lc))) < d_t:
                return None, "owned"
        det_snr = float(sm[int(coarse[0]), int(coarse[1])] / thr * dc_["threshold_sigma"]) \
            if 0 <= coarse[0] < Wf and 0 <= coarse[1] < Wf else float("nan")
    top = np.floor(coarse).astype(int) - N // 2
    if top.min() < 0 or top.max() + N > Wf:
        return None, "off_field"
    Y = Yf[top[0]:top[0] + N, top[1]:top[1] + N]
    star = star_f[top[0]:top[0] + N, top[1]:top[1] + N]
    label_local = mid_f - top
    nb_pos = [p - top for p in nb_pos]
    frame = np.ones((N, N), bool)
    frame[3:-3, 3:-3] = False
    bg_est = float(np.median(Y[frame]))
    ch1 = (Y - bg_est) / sigma_frame
    ch2, ch3 = M.channels(d_meas, CFG4)
    X = np.stack([ch1, ch2, ch3]).astype(np.float32)
    y = (label_local - pc["centre"]).astype(np.float32)
    # parameters (all scalars)
    nd = [np.linalg.norm(p - label_local) for p in nb_pos]
    P.update({
        "snr": snr, "flux_lin": F, "s_eff": s_eff, "psf_model": PSF_CODES[psf["model"]], "a1": psf["a1"], "a2": psf["a2"],
        "angle": psf["angle"] % 180, "psf_p": psf.get("p", psf.get("beta", 0.0)), "coma_frac": psf["coma_frac"],
        "tone_kind": TONE_CODES[tkind], "tone_par": tpar, "bit_depth": depth,
        "mu": bg["mu"], "sigma": bg["sigma"], "skew_s": bg["s"], "kernel_a": bg["ka"], "kernel_b": bg["kb"], "c1": bg["c1"],
        "slope": bg["slope"], "n_neigh": len(nb_pos), "hard_neigh": float(hard),
        "nearest_neigh_px": min(nd) if nd else 99.0, "n_hot": n_hot, "hot_per_mpx": dens,
        "streak": float(streak), "cosmic": float(cosmic), "bad": float(streak or cosmic),
        "rate_factor": mo["par"]["k"], "tau_s": mo["par"].get("tau_s", 0.0), "readout_s": mo["par"]["readout_s"],
        "gyro_noise_deg_s": mo["par"].get("noise_deg_s", 0.0), "sampling_hz": hz,
        "vib_px": mo["par"]["vib"]["amp_px"], "centre_len_px": mo["centre_length_px"],
        "trail_chord_px": float(np.linalg.norm(d_true[-1] - d_true[0])),
        "meas_err_rms_px": float(np.sqrt(np.mean(np.sum((d_meas - mo["d_true_novib"]) ** 2, 1)))),
        "light_minus_label_r": float(lc[0]), "light_minus_label_c": float(lc[1]),
        "resamples": tries, "psf_radius_px": R, "sigma_frame": sigma_frame, "bg_est": bg_est,
        "window_run": wins[wi][0], "window_index": wi, "det_snr": det_snr,
        "coarse_minus_light_px": float(np.hypot(*(coarse - light_f))),
    })
    extras = {"d_true": d_true, "d_meas": d_meas, "psf": psf, "Y": Y, "star": star, "label_local": label_local}
    return (X, y, P, extras), None


# ---------------------------------------------------------------- dataset
def load_split_windows(split, n_per_run=None, seed_off=0):
    sp = CFG5["splits"]
    out = []
    rng = np.random.default_rng([CFG5["seed"], 100 + ["train", "val", "test", "check"].index(split) + seed_off])
    frac = sp["train"] if split == "check" else sp[split]
    n = n_per_run or sp["windows_per_run"]["train" if split == "check" else split]
    for i, (run, end) in enumerate(sp["runs"].items()):
        for w in M.load_windows(CFG4, run, n, rng, frac=frac, end_min=end):
            out.append((i, w))
    return out


def gen_chunk(args):
    split, start, n = args
    Xs, ys, Ps = [], [], []
    for i in range(start, start + n):
        X, y, P, _ = make_sample(split, i)
        Xs.append(X.astype(np.float16))
        ys.append(y)
        Ps.append(P)
    keys = Ps[0].keys()
    return np.stack(Xs), np.stack(ys), {k: np.array([p[k] for p in Ps], np.float32) for k in keys}


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict) and not k.startswith("p_"):
            deep_update(d[k], v)
        else:
            d[k] = v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v1")
    ap.add_argument("--train", type=int, default=100000)
    ap.add_argument("--val", type=int, default=10000)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard", type=int, default=10000)
    ap.add_argument("--override", default=None,
                    help="JSON {step5: {...}, step4: {...}} deep-merged into the configs (step 8 test sets)")
    args = ap.parse_args()
    if args.override:
        ov = json.load(open(args.override))
        deep_update(CFG5, ov.get("step5", {}))
        deep_update(CFG4, ov.get("step4", {}))
    root = HERE / "datasets" / args.name
    root.mkdir(parents=True, exist_ok=True)
    meta = {"name": args.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": {
        "step5": CFG5, "step4": CFG4, "step3": CFG3}, "sizes": {}, "windows": {},
        "X": "float16 (n, 3, 32, 32): ch1 (Y - bg_est) / sigma_frame, ch2 dwell (sum 1), ch3 time ramp",
        "y": "float32 (n, 2): (d_row, d_col) px = true mid-time position - patch centre (16.5, 16.5), corner coords"}
    for split in ["train", "val", "test"]:
        n = getattr(args, split)
        if n == 0:
            continue
        WINDOWS[split] = load_split_windows(split)
        meta["windows"][split] = len(WINDOWS[split])
        print(f"{split}: {len(WINDOWS[split])} windows, {n} samples", flush=True)
        (root / split).mkdir(exist_ok=True)
        t0 = time.time()
        chunks = [(split, s, min(500, n - s)) for s in range(0, n, 500)]
        per = args.shard // 500
        with Pool(args.workers) as pool:
            res = pool.map(gen_chunk, chunks)
        for k in range(0, len(res), per):
            part = res[k:k + per]
            X = np.concatenate([p[0] for p in part])
            y = np.concatenate([p[1] for p in part])
            P = {key: np.concatenate([p[2][key] for p in part]) for key in part[0][2]}
            np.savez(root / split / f"shard_{k // per:03d}.npz", X=X, y=y, **P)
        meta["sizes"][split] = n
        print(f"  done in {time.time() - t0:.0f} s", flush=True)
    json.dump(meta, open(root / "meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
