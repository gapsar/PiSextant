"""Step 2: estimator error vs SNR on synthetic patches (guide section 5, step 2).

COM and Gaussian/Moffat fits (point-sampled templates, review item 5a) on
synthetic patches per SNR bin, for stars at rest and on straight trails.
The Cramer-Rao lower bound comes from Fisher information of this noise
model, computed numerically (review d). See CONVENTIONS.md section SNR for
the SNR definition, written before any plot (review a). The noise model is
the step-1 placeholder (step 3 fits it): slopes and ratios to the bound are
the result, absolute values are provisional (review e). The trail renderer
lives in star_model.py now (review item 4); this file keeps the estimators,
the checks and the run.

Estimator rules (review b and g): fits start from the COM and the
edge-estimated background, never from the truth. Matched fits run first:
matched means same model as rendered; the fit evaluates the PSF at pixel
CENTRES, so the name says "matched, point-sampled" (review 24 Oct item 5a).
Fits are unweighted plain least squares plus weighted fits (review 5a-5b
and 24 Oct v2 item 3, LESSONS 13): the weighted fit is the same
least_squares call with residuals (model - data) divided by sigma_i =
sqrt(model variance at the start model), sigma fixed at the start. Run it
only on the matched point-star (rest) cells: a weighted fit of the wrong
model on a trail is not informative. Rows carry a fit_kind column (v2
item 4): "matched, point-sampled" for a matched fit at rest;
"PSF-shape mismatch" for the Gaussian-on-Moffat rows; "point-PSF on trail
(mismatch)" for any point-PSF fit on a trail state (review item 2: the
point-PSF-on-trail mismatch explains the trail ratios; rounding and clip
explain nothing, because the point star has the same rounding and clip and
gives 1.03-1.23, LESSONS 10). The two mismatches are different and the
fit_kind column keeps them apart. The COM is not a fit; its fit_kind is
null in the data. Bias and scatter are separated over many noise draws at
random sub-pixel positions (review c); the number of draws and the
per-point uncertainty are in the table.

Trail rule (guide 4.1.5): light is summed over every 0.1 px sub-step of the
motion (one sub-step per 0.1 px), equal time steps share the light equally.
The mean patch is the weighted direct sum of peak-1 shape functions at the
sub-step positions (this direct sum is the reference; pinned check 1 checks
it against render_star for a noise-free single render at rest, pinned check
1b checks the kernel weighting itself on identical positions). Per-draw
positions then apply by bilinear sampling of the fine grid: smooth field,
3.3 px scale, error far under 0.01 px, negligible against 1 DN shot noise;
the sampling convention is pinned numerically (pinned check 2), not derived.
Per-draw Fisher gradients use the same path, central difference over one
fine cell (grid-aligned, exact); Fisher is evaluated on the noise-free mean
at truth per draw (review d). The noise-free checks run at one fixed config
flux, before cells, with sign tolerance on the mechanism states (see
noise_free_checks). Trails run along +col (CONVENTIONS.md). Label = mid-time
position = patch centre. The mirror trail is the mechanism test for the COM
bias direction (LESSONS 1): the discrete model predicts the COM (light mean)
sits toward the fast end of the trail at high SNR, so its bias sign flips
against the nonuniform ramp; the guide's step-2 check says "biased toward
the slow end". Numbers settle it; record the answer in the log.

Review items (24 Oct, both reviews) are implemented as: the biased bound
(v2 item 2): in each row fit the error (estimate - truth) against the true
offset (drawn uniformly over +/-1 px), the slope is db/dtheta, and check
scatter >= |1 + slope| * CRB in every row, all bins (the plain bound 1/I
holds for UNBIASED estimators only); RMS columns and the rms_vs_snr plot
(RMS error = sqrt(bias^2 + scatter^2), full range, all bins reported); the
fit_kind column in the data and the plot legends (v2 item 4); dual_test
with the three predictions of LESSONS 12, each with its because (v2 item
1); the weighted fit as the standard least_squares call (v2 item 3); and
the plots in src/plot_step2.py, which reads table.json (v2 item 5).

Usage (run from the centroid_cnn folder):
    python3 src/error_vs_snr.py            # full run, writes reports/step2
    python3 src/error_vs_snr.py --smoke    # small test, pinned checks, scratch/smoke
    python3 src/plot_step2.py              # the four plots, from table.json (v2 item 5)
"""

import argparse
import json
import math
import os  # stat for the provenance fields (review item 4)
import shutil  # provenance copy of the config
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

import star_model  # the renderer, incl. the trail code (step 2 review item 4)


def load_config(path):
    """Read the JSON config file. Input: path. Output: dict."""
    with open(path) as f:
        return json.load(f)


def rng_for_check(seed, check_number):
    """One random generator per cell (same habit as render_star.py, review
    item 7). Inputs: seed, number (a stable index list). Output: Generator."""
    return np.random.default_rng([seed] + list(check_number))


def sigma_pix_dn(cfg):
    """Per-pixel noise std at the patch background, DN (CONVENTIONS SNR).
    Input: config. Output: float."""
    gain = cfg["noise"]["electrons_per_dn"]
    return math.sqrt(cfg["background_dn"] / gain
                     + cfg["noise"]["read_noise_sigma_dn"] ** 2)


def s_sigma_px(psf_params):
    """The s of the SNR formula: sigma for a Gaussian, the sigma of the
    Gaussian with the same FWHM for a Moffat (CONVENTIONS SNR). Inputs: PSF
    params dict. Output: px."""
    if psf_params["shape"] == "gaussian":
        return psf_params["major_px"]
    fwhm = 2.0 * psf_params["major_px"] * math.sqrt(2.0 ** (1.0 / psf_params["beta"]) - 1.0)
    return fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def flux_for_snr(snr, cfg):
    """Solve the SNR formula for the star total light F, DN (infinite
    plane). Inputs: SNR number, config. Output: F, DN. Both shapes give the
    same answer because their widths are FWHM-matched (step 1)."""
    s = s_sigma_px(cfg["gaussian_params"])
    return snr * sigma_pix_dn(cfg) * math.sqrt(2.0 * math.pi * s * s)


def truth_grid(cfg):
    """Truth positions for the estimator grids: centre pixel index + 0.5,
    centre coords, CONVENTIONS Patch centre. Input: config. Output:
    (row, col) px."""
    c = cfg["center_pixel_index"] + 0.5
    return (c, c)


def edge_background(patches, cfg):
    """Mean of the one-pixel edge ring per patch, DN (the COM background
    estimate and the fit starting value, review b). Inputs: quantized
    patches (N, patch, patch) DN, config. Output: (N,) array, DN."""
    ring = cfg["com_edge_ring_px"]
    parts = [patches[:, :ring], patches[:, -ring:],
             patches[:, ring:-ring, :ring], patches[:, ring:-ring, -ring:]]
    return np.concatenate([p.reshape(len(patches), -1) for p in parts], axis=1).mean(axis=1)


def com_estimates(patches, cfg, threshold_sigmas=None):
    """COM per draw on quantized patches: subtract the edge-ring background
    estimate, keep pixels above the threshold, centroid over the kept
    positives at centre coords (i + 0.5, CONVENTIONS), normalise by the
    kept sum (render_star rule: normalise by the kept sum). Inputs:
    quantized patches (N, patch, patch) DN, config, threshold_sigmas:
    override for cfg["com_threshold_sigmas"], used by dual_test. Output:
    (N, 2) position px, NaN rows where the threshold left nothing."""
    p = cfg["patch_px"]
    d = patches - edge_background(patches, cfg)[:, None, None]
    sigmas = cfg["com_threshold_sigmas"] if threshold_sigmas is None else threshold_sigmas
    thr = sigmas * sigma_pix_dn(cfg)
    m = np.where(d > thr, d, 0.0)
    rs = (np.arange(p) + 0.5)[None, :, None]
    cs = (np.arange(p) + 0.5)[None, None, :]
    total = m.sum(axis=(1, 2))
    out = np.full((len(patches), 2), np.nan)
    ok = total > 0.0
    out[ok, 0] = (m * rs).sum(axis=(1, 2))[ok] / total[ok]
    out[ok, 1] = (m * cs).sum(axis=(1, 2))[ok] / total[ok]
    return out


def fit_init(patches, com, cfg):
    """Starting values for the fits (review b): position from the COM,
    background from the edge ring, flux from the sum above background
    (data-driven, never the truth). Inputs: quantized patches, COM
    estimates, config. Output: (N, 4) initial (row, col, flux, bkg)."""
    centre = cfg["center_pixel_index"] + 0.5  # CONVENTIONS centre pixel
    bkg = edge_background(patches, cfg)
    pos = np.where(np.isnan(com), centre, com)  # empty-COM rows start at the centre
    flux = (patches - bkg[:, None, None]).sum(axis=(1, 2))
    return np.stack([pos[:, 0], pos[:, 1], np.maximum(flux, 1.0), bkg], axis=1)


def lsq_fits(patches, template_params, init, cfg):
    """Unweighted least-squares fit per draw: centre, total light,
    background, on the point-sampled template model (model = flux /
    normalization * peak-1 shape + bkg, evaluated at pixel CENTRES, review
    item 5a; grid in centre coords, CONVENTIONS). When the weights are all
    1 this IS the weighted fit, so these rows match the previous run
    exactly. Inputs: quantized patches (N, patch, patch) DN, template
    params, init per draw (N, 4), config. Output: (N, 4) fitted (row, col,
    flux, bkg), NaN rows where the fit did not converge or could not start.
    """
    p = cfg["patch_px"]
    grid = np.arange(p) + 0.5  # centre coords, CONVENTIONS
    rr = grid[:, None] * np.ones((1, p))
    cc = grid[None, :] * np.ones((p, 1))
    norm = star_model.psf_normalization(template_params)
    out = np.full((len(patches), 4), np.nan)
    for i in range(len(patches)):
        x0 = init[i]
        if not np.all(np.isfinite(x0)) or x0[2] <= 0.0:
            continue

        def resid(x, data=patches[i]):
            model = (x[2] / norm) * star_model.psf_unit_shape(
                cc - x[1], rr - x[0], template_params) + x[3]
            return (model - data).ravel()

        res = least_squares(resid, x0, x_scale=[1.0, 1.0, max(x0[2], 1.0), 1.0],
                            ftol=1e-8, xtol=1e-8)
        if res.success:
            out[i] = res.x
    return out


def weighted_fits(patches, template_params, init, cfg, unity=False):
    """Weighted fit per draw: the standard least_squares call with residuals
    (model - data) / sigma_i, sigma_i = sqrt(model / gain + read^2) at the
    START model, the model including the background (the same variance
    formula as fisher_crlb). One pass.

    History (25 Oct, scratch/followup_a2_claude.py): the v3 version left the
    background out of sigma_i, so far from the star the variance was read^2
    (0.18 DN^2) instead of about 2.18 DN^2, and the wing pixels got about 12x
    too much weight: ratio 1.29 vs plain 1.05 at bin 20. With the full
    variance, one pass from the start model gives the same ratios as weights
    from the true model (within 0.01), so the start position does not matter
    and no second pass is needed.

    Run on the matched point-star (rest) cells only. Inputs: quantized
    patches, template params, init (N, 4), config, unity flag (smoke
    equivalence test: sigma forced to 1, so this call must equal the plain
    fit on the same data). Output: (N, 4) fitted (row, col, flux, bkg), NaN
    rows where the fit did not converge or could not start."""
    p = cfg["patch_px"]
    grid = np.arange(p) + 0.5  # centre coords, CONVENTIONS
    rr = grid[:, None] * np.ones((1, p))
    cc = grid[None, :] * np.ones((p, 1))
    params = template_params
    norm = star_model.psf_normalization(params)
    gain = cfg["noise"]["electrons_per_dn"]
    read2 = cfg["noise"]["read_noise_sigma_dn"] ** 2

    def model_at(x):
        return (x[2] / norm) * star_model.psf_unit_shape(cc - x[1], rr - x[0], params) + x[3]

    out = np.full((len(patches), 4), np.nan)
    for i in range(len(patches)):
        x0 = init[i]
        if not np.all(np.isfinite(x0)) or x0[2] <= 0.0:
            continue
        sig = np.ones((p, p)) if unity else np.sqrt(np.maximum(model_at(x0), 0.0) / gain + read2)
        res = least_squares(lambda x, d=patches[i], s=sig: ((model_at(x) - d) / s).ravel(),
                            x0, x_scale=[1.0, 1.0, max(x0[2], 1.0), 1.0], ftol=1e-8, xtol=1e-8)
        if res.success:
            out[i] = res.x
    return out


def fisher_crlb(mean_blocks, unit_blocks, g_row, g_col, cfg):
    """Numeric Fisher information per draw, from this noise model (review
    d), parameters row, col, flux, background. Inputs: mean blocks at truth
    (N,p,p) DN background-included, unit-shape blocks at truth (N,p,p)
    peak-1 (d mean / d flux: exact, linear), position gradients per pixel
    DN/px (N,p,p), config. Output: (N, 2) CRLB std px per axis. Ignores
    8-bit rounding and the clip at 0: a floor, the gap grows at low SNR."""
    gain = cfg["noise"]["electrons_per_dn"]
    var = mean_blocks / gain + cfg["noise"]["read_noise_sigma_dn"] ** 2
    G = np.stack([g_row, g_col, unit_blocks, np.ones_like(mean_blocks)])  # (4,N,p,p)
    w = 1.0 / var  # Fisher weights: var > 0 because background > 0
    info = np.einsum("ankl,bnkl,nkl->nab", G, G, w)
    inv = np.linalg.inv(info)
    return np.sqrt(np.maximum(np.stack([inv[:, 0, 0], inv[:, 1, 1]]), 0.0)).T  # (N, 2)


def fisher_blocks(params, peak1_base, cfg, centre, off, step):
    """Peak-1 unit-shape blocks and one-cell central-difference gradients
    at one draw offset, all via the sampled path (grid-aligned: step = one
    fine cell = 1 index unit, so gradient values come from exact grid
    points). Inputs: params, peak-1 base field, config, centre px, offset
    px, fine-cell px size. Output: (unit, g_row, g_col) block-mean arrays."""
    unit = star_model.sample_blocks(peak1_base, cfg, [off])[0]
    up = star_model.sample_blocks(peak1_base, cfg, [(off[0] + step, off[1])])[0]
    dn = star_model.sample_blocks(peak1_base, cfg, [(off[0] - step, off[1])])[0]
    ri = star_model.sample_blocks(peak1_base, cfg, [(off[0], off[1] + step)])[0]
    le = star_model.sample_blocks(peak1_base, cfg, [(off[0], off[1] - step)])[0]
    return unit, (up - dn) / (2 * step), (ri - le) / (2 * step)


def stats(errors, offsets=None):
    """Bias mean, scatter std, their standard errors, RMS error
    sqrt(bias^2 + scatter^2) (review item 1), count used (review c). When
    offsets are given (v2 review item 2, the biased bound): fit the error
    (estimate - truth) against the true offset (drawn uniformly over
    +/-1 px, degree-1 polyfit), the slope is db/dtheta and its SE comes
    from the residual scatter (needs 3 points or more, else None). Input:
    per-draw errors px array (can hold NaN), per-draw true offsets px array
    (same length). Output: dict."""
    e = np.asarray(errors, dtype=np.float64)
    n = int(np.sum(np.isfinite(e)))
    mean = float(np.nanmean(e))
    std = float(np.nanstd(e, ddof=1))
    out = {"bias": mean, "bias_se": std / math.sqrt(n), "scatter": std,
           "scatter_se": std / math.sqrt(n), "rms": math.sqrt(mean * mean + std * std),
           "n": n, "bias_slope": None, "bias_slope_se": None}
    if offsets is not None:
        off = np.asarray(offsets, dtype=np.float64)
        m = np.isfinite(e) & np.isfinite(off)
        if int(np.sum(m)) >= 3:  # degree-1 polyfit needs 3 points for an SE
            x, y = off[m], e[m]
            cf = np.polyfit(x, y, 1)
            out["bias_slope"] = float(cf[0])
            resid = y - np.polyval(cf, x)
            denom = float(np.sum((x - np.mean(x)) ** 2))
            s2 = float(np.sum(resid * resid) / (len(x) - 2))
            out["bias_slope_se"] = math.sqrt(s2 / denom) if denom > 0 else None
    return out


FIT_MATCHED = "matched, point-sampled"  # fit_kind codes (v2 review item 4)
FIT_SHAPE = "PSF-shape mismatch"
FIT_TRAIL = "point-PSF on trail (mismatch)"


def fit_kind(est_name, state_name):
    """fit_kind per row (v2 review item 4, used by the plot legends in
    src/plot_step2.py). The COM is not a fit, so its fit_kind is null. The
    Gaussian-on-Moffat rows are "PSF-shape mismatch" even on trail states:
    the shape mismatch wins over the trail mismatch, a documented decision
    for the review. Any other point-PSF fit on a trail state is
    "point-PSF on trail (mismatch)". Inputs: estimator name, state name.
    Output: string or None."""
    if est_name == "com":
        return None
    if est_name.startswith("gaussian_on_moffat"):
        return FIT_SHAPE
    if state_name == "rest":
        return FIT_MATCHED
    return FIT_TRAIL


def pinned_checks(cfg):
    """Run these before quoting any cell or plot number (guide section B;
    LESSONS 1: verify, do not derive; a check that cannot fail catches
    nothing). Check 1 and 1b run the two checks that live in star_model.py
    (moved with the code they test, review item 4). Check 2: the sampling
    sign and index scale, tested EXACT at whole fine cells (coordinates
    land on grid points, so bilinear equals the source roll exactly, 1e-9,
    away from field edges): a scene shift of +0.1 px along +col must show
    source cell (j, b) at output cell (j, b + 1). Check 3: the fractional-
    offset claim, one draw at (0.13, 0.37) px, bilinear sampling path vs
    the direct sub-step sum on the same peak-1 Gaussian field (tolerance
    2e-3, documented, where the claim can fail). Inputs: config. Output:
    list of (name, passed, detail); all must pass before any cell line
    prints."""
    results = [star_model.check_zero_trail_equals_render_star(cfg),
               star_model.check_zero_trail_weighted_kernel_equals_direct(cfg)]
    p, s = cfg["patch_px"], cfg["supersample_per_side"]
    g = p + 2 * star_model.PAD
    # Check 2: shift sign on a curved field (so a wrong sign is visible),
    # tested at whole fine cells where sampling is exact.
    cellidx = np.arange(g * s)
    field = 0.001 * cellidx[:, None] ** 2 + 0.7 * cellidx[None, :] + 0.01 * cellidx[:, None] * cellidx[None, :]
    field = field % 50.0 + 10.0  # keep values in (0, 60), positive, curved
    sampled = map_coordinates(field, [np.arange(g * s)[:, None] * np.ones((1, g * s)),
                                      np.arange(g * s)[None, :] * np.ones((g * s, 1)) - 1],
                              order=1, mode="nearest")  # scene moved +0.1 px along +col
    want = np.roll(field, 1, axis=1)  # source column j shows at output column j+1
    interior = slice(2, g * s - 2)  # roll wraps, nearest clamps: compare interior
    d2 = float(np.max(np.abs(sampled[interior, interior] - want[interior, interior])))
    results.append(("check2_sample_shift_sign", d2 < 1e-9, f"max diff {d2:.2e}"))
    # Check 3: fractional offset (0.13, 0.37) px, sampling path vs direct sum.
    gf = cfg["gaussian_params"]
    gc = star_model.grid_centre(cfg)
    field0 = star_model.build_base(gf, 1.0, [0.0], [1.0], cfg, gc)  # peak-1 field
    got3 = star_model.sample_blocks(field0, cfg, [(0.13, 0.37)])[0]
    row_px3, col_px3 = star_model.fine_grids(cfg)
    direct = star_model.block_mean(star_model.eval_unit(gf, row_px3, col_px3, gc, (0.13, 0.37)), cfg)
    d3 = float(np.max(np.abs(got3 - direct)))
    results.append(("check3_fractional_sampling", d3 < 2e-3, f"max diff {d3:.2e} peak-1 units"))
    return results


def run_cell(cfg, state_name, segments, shape_name, snr, seed_idx, out_rows,
             counts_rows, allow_mismatch, samples=None, label_shift_px=(0.0, 0.0),
             threshold_sigmas=None, weighted=False, unity=False, label=""):
    """Run one cell (state x shape x SNR bin): N noise draws at random
    sub-pixel offsets (offsets drawn FIRST, then noise, one rng per cell,
    review c), COM and fits on quantized pixels, Fisher CRLB on the first
    n_fisher draws, and one noise-free single render check against the
    noise-free checks (separate function, fixed config flux). Inputs:
    config, state name, segments, shape name, SNR bin, cell index list,
    rows lists to fill (stats, counts), mismatch-allowed flag, samples dict
    for the example grid (can be None), label_shift_px (dual_test: moves
    scene centre AND truth reference together, so the test measures the
    geometry effect only), threshold_sigmas (dual_test COM override),
    weighted (also run the weighted fit; the matched point-star (rest)
    cells only, v2 review item 3), unity (smoke only: sigma forced to 1,
    the weighted fit must then equal the plain fit on the same data),
    label (cell line suffix). Output: None (fills rows); prints the cell
    line for the log.
    """
    t0 = time.time()
    if label_shift_px is None:  # dual_test passes None for the normal label
        label_shift_px = (0.0, 0.0)
    rng = rng_for_check(cfg["seed"], seed_idx)
    n = cfg["n_draws"]
    params = cfg[shape_name + "_params"]
    flux = flux_for_snr(snr, cfg)
    peak = star_model.shape_peak(params, flux)
    label_px = star_model.label_at(segments)
    pos_arr, w_arr = star_model.path_weights(segments, label_px)
    centre = (truth_grid(cfg)[0] + label_shift_px[0], truth_grid(cfg)[1] + label_shift_px[1])
    gc = (star_model.grid_centre(cfg)[0] + label_shift_px[0],
          star_model.grid_centre(cfg)[1] + label_shift_px[1])
    base = star_model.build_base(params, peak, pos_arr, w_arr, cfg, gc)
    offsets = rng.uniform(-cfg["offset_range_px"], cfg["offset_range_px"], (n, 2))
    blocks = star_model.sample_blocks(base, cfg, offsets)  # (N, p, p) mean blocks, no bkg
    lin, quant = star_model.noise_and_quantize(blocks, cfg, rng)
    truth = np.asarray(centre)[None, :] + offsets  # per-draw truth, centre coords
    # Estimators on quantized data:
    com = com_estimates(quant, cfg, threshold_sigmas)
    init = fit_init(quant, com, cfg)
    fits = {"com": com}
    fits[f"{params['shape']}_on_{shape_name}"] = lsq_fits(quant, params, init, cfg)
    if weighted:  # matched point-star (rest) cells only (v2 review item 3)
        fits[f"{params['shape']}_on_{shape_name}_weighted"] = weighted_fits(
            quant, params, init, cfg, unity=unity)
    if allow_mismatch:  # gaussian template on moffat star (review g)
        fits["gaussian_on_moffat"] = lsq_fits(quant, cfg["gaussian_params"], init, cfg)
    # Fisher CRLB on the first n_fisher draws (numeric, this noise model):
    nf = min(cfg["n_fisher_draws"], n)
    step = 0.1  # px, one fine cell; same sampled path as the draws
    peak1_base = base / peak if peak > 0 else base
    mean_t = blocks[:nf] + cfg["background_dn"]  # noise-free mean at truth (review d)
    norm = star_model.psf_normalization(params)  # px^2, the analytic integral
    Gs = np.empty((nf, 4, cfg["patch_px"], cfg["patch_px"]))
    for i in range(nf):
        unit, g_row, g_col = fisher_blocks(params, peak1_base, cfg, centre, offsets[i], step)
        Gs[i, 0] = (flux / norm) * g_row  # d mean / d row_px, DN/px
        Gs[i, 1] = (flux / norm) * g_col  # d mean / d col_px, DN/px
        Gs[i, 2] = unit / norm  # d mean / d flux, DN/DN (model linear in flux)
        Gs[i, 3] = 1.0  # d mean / d bkg, DN/DN
    crlb = fisher_crlb(mean_t, Gs[:, 2], Gs[:, 0], Gs[:, 1], cfg)
    crlb_mean = [float(np.mean(crlb[:, 0])), float(np.mean(crlb[:, 1]))]
    # Stats per estimator, per axis (review c: bias and scatter separated):
    counts = {"state": state_name, "shape": shape_name, "snr_bin": snr,
              "com_empty": int(np.sum(~np.isfinite(com[:, 0])))}
    est_errors = {}
    for name, est in fits.items():  # errors on the position columns only
        counts[name] = int(np.sum(~np.isfinite(est[:, 0])))
        est_errors[name] = np.asarray(est)[:, :2] - truth
    counts_rows.append(counts)
    if samples is not None and snr == 20:  # example grid bin, noise on
        samples[(state_name, shape_name)] = quant[:9].copy()
    row_list = []
    for name, err in est_errors.items():
        for ax, axis in enumerate(("row", "col")):
            st = stats(err[:, ax], offsets[:, ax])  # v2 item 2: slope on the offset
            adj = (abs(1.0 + st["bias_slope"]) * crlb_mean[ax]
                   if st["bias_slope"] is not None and crlb_mean[ax] > 0 else None)
            row_list.append({"state": state_name, "shape": shape_name, "snr_bin": snr,
                             "estimator": name, "axis": axis, "test": label,
                             "fit_kind": fit_kind(name, state_name), "crlb_px": crlb_mean[ax],
                             "ratio_scatter_over_crlb": (st["scatter"] / crlb_mean[ax]
                                                         if crlb_mean[ax] > 0 else None),
                             "biased_bound_px": adj,
                             "ratio_scatter_over_biased_bound": (st["scatter"] / adj
                                                                 if adj and adj > 1e-9 else None),
                             "ratio_rms_over_crlb": (st["rms"] / crlb_mean[ax]
                                                     if crlb_mean[ax] > 0 else None), **st})
    out_rows.extend(row_list)
    print(f"cell {state_name}/{shape_name}/snr{snr}{label}: n={n} "
          f"com_col_bias={stats(com[:, 1] - truth[:, 1], offsets[:, 1])['bias']:+.4f}px "
          f"com_col_scatter={stats(com[:, 1] - truth[:, 1], offsets[:, 1])['scatter']:.4f}px "
          f"crlb_col={crlb_mean[1]:.4f}px [{time.time() - t0:.1f}s]", flush=True)


def discrete_predictions(cfg):
    """Table of discrete-model predictions for the COM on noise-free
    high-SNR data (light mean minus mid-time label, px, trail axis +col;
    row axis 0 for all states). Input: config. Output: dict state->px."""
    out = {}
    for name, segments in cfg["trail_states"].items():
        out[name] = star_model.light_mean_bias(segments)
    out[cfg["nonuniform_mirror"]["name"]] = star_model.light_mean_bias(
        cfg["nonuniform_mirror"]["segments"])
    return out


def dual_test(cfg, rows, seed_lists, out_rows):
    """Test for the two broken symmetries (review item 3, LESSONS 11) with
    the three predictions of LESSONS 12 (v2 review item 1: the v2 criteria
    were wrong, not the hypothesis; the v2 data fit predictions a, b and c
    exactly and the v2 verdict said "not proven"). The hypothesis, tested
    not assumed: the label sits at 16.5 + offset, but the geometric centre
    of the patch is 16.0; at low SNR the thresholded COM shrinks toward the
    patch centre, so the mean bias goes toward -0.5 px.

    Prediction (a): label at 16.0 -> the six low-SNR beater-cell biases go
    to 0, BECAUSE the pull comes from the label-centre asymmetry: with the
    label AT the patch centre the threshold trim is symmetric about the
    label, so the mean pull is zero. TRUE branch (threshold fixed before
    the run, LESSONS 9): every cell within 2 SE of 0. FALSE branch: every
    cell within 2 SE of its normal-run bias (unchanged; the v2 values,
    measured before the run, were -0.38 to -0.55 px).
    Prediction (b): label at 16.0 -> the mirror pair becomes symmetric
    (equal size, opposite sign), BECAUSE with the label at the patch centre
    the mean pull is gone and what is left is the discrete mechanism, which
    is mirror-symmetric by construction. The pair SIZE (about 0.45 px, not
    the noise-free 0.875) is the threshold trim at 976 DN: the hypothesis
    does not touch it and no criterion lands on it. TRUE: pair sum within
    3 SE of 0, both shapes. FALSE: pair sum within 3 SE of its normal-run
    value (asymmetry unchanged; the v2 values were 5.05 and 5.49 SE).
    Prediction (c): threshold 1 sigma at label 16.5 -> MORE noise pixels
    kept, a STRONGER pull toward 16.0, so both biases shift by the SAME
    amount in the -col direction: the pair centre moves below 0, BECAUSE
    the pull is the threshold term and it acts on both members equally.
    TRUE: the bin-80 pair centre at 1 sigma is below its normal-run centre
    by more than 3 SE, both shapes. FALSE: the centre does not move (the
    mechanism adds nothing). The bins 3-5 1-sigma rows are reported for
    consistency only: the pull is nearly saturated there (LESSONS 9).

    Each affected cell (the 10 cells: the 6 holding the bound-beating rows
    of the previous run, plus the bin-80 pair cells and their Moffat
    partners) is run twice more with the same seed index as the normal cell
    (same offsets and noise streams: paired). SEs are treated as
    independent, which is conservative (pairing can only shrink them;
    documented for the review). The v2 numbers reproduce exactly on re-run
    (seed fixed, plain rows unchanged), so the expected outcome measured
    before the run from scratch/table_v2_old.json is: a 6/6 TRUE and 0
    FALSE, b 2/2 and 0, c 2/2 and 0 -> verdict "cause proven". Inputs:
    config, normal-run rows, the fixed cell seed lists, rows list to fill.
    Output: dict (predictions, because-lines, hit counts, verdict; stored
    in table.json); fills rows, prints the table."""
    states = {name: segs for name, segs in cfg["trail_states"].items()}
    mirror = cfg["nonuniform_mirror"]
    states[mirror["name"]] = mirror["segments"]
    pred = discrete_predictions(cfg)
    mirror_name = mirror["name"]

    def find_row(state, shape, bin_, axis, est, test):
        m = [r for r in rows if r["state"] == state and r["shape"] == shape
             and r["snr_bin"] == bin_ and r["axis"] == axis
             and r["estimator"] == est and r["test"] == test]
        return m[0] if m else None

    print("--- dual test (review item 3; criteria rewritten per LESSONS 12, v2 item 1) ---",
          flush=True)
    print("prediction (a): label at 16.0 -> the six low-SNR beater-cell biases go to 0, "
          "because the pull comes from the label-centre asymmetry and with the label at the "
          "patch centre the trim is symmetric about the label. TRUE (fixed 2 SE): within "
          "2 SE of 0. FALSE (fixed 2 SE): unchanged, within 2 SE of the normal-run bias.",
          flush=True)
    print("prediction (b): label at 16.0 -> the mirror pair becomes symmetric (equal size, "
          "opposite sign), because the mean pull is gone and the discrete mechanism is "
          "mirror-symmetric. The size (about 0.45 px, not 0.875) is the threshold trim at "
          "976 DN, which no criterion judges. TRUE (fixed 3 SE): pair sum within 3 SE of 0. "
          "FALSE (fixed 3 SE): unchanged, within 3 SE of the normal-run pair sum.", flush=True)
    print("prediction (c): threshold 1 sigma at label 16.5 -> MORE noise pixels kept, a "
          "STRONGER pull toward 16.0, so both biases shift by the same amount in the -col "
          "direction (pair centre below 0), because the pull is the threshold term and acts "
          "on both members equally. TRUE (fixed 3 SE): the bin-80 pair centre moves below "
          "its normal-run centre by more than 3 SE. FALSE (fixed 3 SE): the centre does not "
          "move. Bins 3-5 reported for consistency only (pull nearly saturated, LESSONS 9).",
          flush=True)
    for (state, shape, bin_, seed_idx) in seed_lists:  # 20 test runs (10 cells x 2)
        for lab, thr, tag in (((-0.5, -0.5), None, " TEST label16.0"),
                              (None, 1.0, " TEST thr1sigma")):
            run_cell(cfg, state, states[state], shape, bin_, seed_idx, out_rows, [],
                     False, None, label_shift_px=lab, threshold_sigmas=thr,
                     weighted=False, label=tag)
    for (state, shape, bin_, seed_idx) in seed_lists:
        for axis in ("row", "col"):
            nd = find_row(state, shape, bin_, axis, "com", "")
            v1 = find_row(state, shape, bin_, axis, "com", " TEST label16.0")
            v2 = find_row(state, shape, bin_, axis, "com", " TEST thr1sigma")
            pd = pred[state] if axis == "col" else 0.0
            print(f"  {state}/{shape}/snr{bin_}/{axis}: normal {fmt(nd['bias'], 4)}+/-{fmt(nd['bias_se'], 4)} "
                  f"| label16.0 {fmt(v1['bias'], 4)}+/-{fmt(v1['bias_se'], 4)} "
                  f"| thr1sigma {fmt(v2['bias'], 4)}+/-{fmt(v2['bias_se'], 4)} "
                  f"| discrete pred {pd:+.4f}", flush=True)
    # Criterion (a), fixed before the run: the six low-SNR beater cells
    # (the cells whose COM rows beat the plain bound in the previous run),
    # col axis, the axis the bias lives on. Row-axis rows are reported in
    # the table only (prediction 0 there, the mechanism does not touch the
    # row axis for a +col trail).
    a_cells = [("uniform_15px", "gaussian", 3), ("uniform_15px", "moffat", 3),
               ("nonuniform_15px", "gaussian", 3), ("nonuniform_15px", "moffat", 3),
               ("nonuniform_15px", "moffat", 5), ("uniform_15px", "moffat", 5)]
    print("criterion (a), the six low-SNR beater cells, col axis:", flush=True)
    a_true = a_false = 0
    for (state, shape, bin_) in a_cells:
        nd = find_row(state, shape, bin_, "col", "com", "")
        v1 = find_row(state, shape, bin_, "col", "com", " TEST label16.0")
        se_t = 2.0 * v1["bias_se"]  # TRUE: bias at 0, mean pull gone
        se_f = 2.0 * math.hypot(v1["bias_se"], nd["bias_se"])  # FALSE: unchanged
        t_ok = abs(v1["bias"]) < se_t
        f_ok = abs(v1["bias"] - nd["bias"]) < se_f
        a_true += t_ok
        a_false += f_ok
        print(f"  {state}/{shape}/snr{bin_}/col: label16.0 bias {fmt(v1['bias'], 4)}; TRUE "
              f"(pred 0) {'HITS' if t_ok else 'misses'} (< 2SE {fmt(se_t, 4)}); FALSE "
              f"(unchanged {fmt(nd['bias'], 4)}) {'HITS' if f_ok else 'misses'} "
              f"(< 2SE {fmt(se_f, 4)})", flush=True)
    print("criterion (b), the bin-80 mirror pair at label 16.0, pair sum (equal-and-opposite "
          "test):", flush=True)
    b_true = b_false = 0
    for shape in ("gaussian", "moffat"):
        na = find_row("nonuniform_15px", shape, 80, "col", "com", " TEST label16.0")
        nb = find_row(mirror_name, shape, 80, "col", "com", " TEST label16.0")
        na0 = find_row("nonuniform_15px", shape, 80, "col", "com", "")
        nb0 = find_row(mirror_name, shape, 80, "col", "com", "")
        se = math.hypot(na["bias_se"], nb["bias_se"])
        sum16 = na["bias"] + nb["bias"]
        sum0 = na0["bias"] + nb0["bias"]
        se_f = math.hypot(se, math.hypot(na0["bias_se"], nb0["bias_se"]))
        t_ok = abs(sum16) < 3.0 * se  # TRUE: symmetric pair, sum 0
        f_ok = abs(sum16 - sum0) < 3.0 * se_f  # FALSE: asymmetry unchanged
        b_true += t_ok
        b_false += f_ok
        print(f"  {shape}: pair sum at label16.0 {fmt(sum16, 4)} px; TRUE (pred 0) "
              f"{'HITS' if t_ok else 'misses'} (< 3SE {fmt(3.0 * se, 4)}); FALSE (unchanged "
              f"{fmt(sum0, 4)}) {'HITS' if f_ok else 'misses'} (< 3SE {fmt(3.0 * se_f, 4)})",
              flush=True)
    print("criterion (c), the bin-80 mirror pair at threshold 1 sigma, pair-centre shift "
          "(both members move the same way):", flush=True)
    c_true = c_false = 0
    for shape in ("gaussian", "moffat"):
        na = find_row("nonuniform_15px", shape, 80, "col", "com", " TEST thr1sigma")
        nb = find_row(mirror_name, shape, 80, "col", "com", " TEST thr1sigma")
        na0 = find_row("nonuniform_15px", shape, 80, "col", "com", "")
        nb0 = find_row(mirror_name, shape, 80, "col", "com", "")
        c1 = 0.5 * (na["bias"] + nb["bias"])
        c0 = 0.5 * (na0["bias"] + nb0["bias"])
        se = math.hypot(0.5 * math.hypot(na["bias_se"], nb["bias_se"]),
                        0.5 * math.hypot(na0["bias_se"], nb0["bias_se"]))
        shift = c1 - c0
        t_ok = shift < -3.0 * se  # TRUE: stronger pull, centre moves -col
        f_ok = abs(shift) < 3.0 * se  # FALSE: the mechanism adds nothing
        c_true += t_ok
        c_false += f_ok
        print(f"  {shape}: pair centre {fmt(c0, 4)} -> {fmt(c1, 4)} (1 sigma), shift "
              f"{fmt(shift, 4)} px; TRUE (pred: -col shift) {'HITS' if t_ok else 'misses'} "
              f"(below {fmt(-3.0 * se, 4)}); FALSE (no shift) {'HITS' if f_ok else 'misses'} "
              f"(< 3SE {fmt(3.0 * se, 4)})", flush=True)
    # Pair difference and its standard error at bin 80 (LESSONS 11, all
    # three tags; the numbers the criteria above judge):
    pair_d = {}
    for shape in ("gaussian", "moffat"):
        for tag, tname in (("", "normal"), (" TEST label16.0", "label16.0"), (" TEST thr1sigma", "thr1sigma")):
            ra, rb = find_row("nonuniform_15px", shape, 80, "col", "com", tag), \
                     find_row(mirror_name, shape, 80, "col", "com", tag)
            if ra is None or rb is None:
                continue
            se = math.hypot(ra["bias_se"], rb["bias_se"])
            d = abs(ra["bias"]) - abs(rb["bias"])
            pair_d[(shape, tname)] = (d, se)
            print(f"  pair bin80 {shape}/{tname}: |diff| {fmt(d, 4)} px, SE {fmt(se, 4)}, "
                  f"ratio {fmt(d / se, 2)} SE", flush=True)
    verdict = (f"mixed counts only: a {a_true}/6 TRUE ({a_false} FALSE), b {b_true}/2 TRUE "
               f"({b_false} FALSE), c {c_true}/2 TRUE ({c_false} FALSE) - read the printed "
               f"rows, the expectation measured before the run does not reproduce")
    if a_true == 6 and b_true == 2 and c_true == 2 and a_false == 0 and b_false == 0 and c_false == 0:
        verdict = ("cause proven (LESSONS 12: predictions a, b and c all TRUE, both FALSE "
                   "branches empty): a 6/6, b 2/2, c 2/2 TRUE hits")
    print(f"DUAL TEST VERDICT (review item 3): {verdict}", flush=True)
    return {"hypothesis": "the COM shrinks toward the patch centre 16.0 while the label "
                           "sits at 16.5; the threshold trim creates the pull",
            "pred_a": {"prediction": "label 16.0 -> the six low-SNR beater-cell biases go to 0",
                       "because_true": "the pull comes from the label-centre asymmetry; with "
                                       "the label at the patch centre the trim is symmetric",
                       "because_false": "if the mechanism is not the cause each bias stays at "
                                        "its normal-run value",
                       "true_branch": "within 2 SE of 0", "false_branch": "within 2 SE of normal",
                       "true_hits": a_true, "false_hits": a_false, "of": 6},
            "pred_b": {"prediction": "label 16.0 -> the mirror pair becomes symmetric "
                                     "(equal size, opposite sign)",
                       "because_true": "the mean pull is gone; the discrete mechanism is "
                                       "mirror-symmetric by construction",
                       "because_false": "if the mechanism is not the cause the pair keeps its "
                                        "normal-run asymmetry",
                       "true_branch": "pair sum within 3 SE of 0, both shapes",
                       "false_branch": "pair sum within 3 SE of normal",
                       "true_hits": b_true, "false_hits": b_false, "of": 2},
            "pred_c": {"prediction": "threshold 1 sigma at label 16.5 -> both biases shift "
                                     "the same amount in -col (pair centre below 0)",
                       "because_true": "more noise pixels kept, a stronger pull toward 16.0, "
                                       "equal on both members",
                       "because_false": "if the mechanism is not the cause the centre does "
                                        "not move",
                       "true_branch": "bin-80 pair centre below normal centre by > 3 SE",
                       "false_branch": "centre does not move (within 3 SE)",
                       "true_hits": c_true, "false_hits": c_false, "of": 2},
            "verdict": verdict}


def loglog_slope(bins_used, scatters):
    """Log-log slope of scatter vs SNR (guide check 1): polyfit degree 1 on
    the points that are positive on both axes. Inputs: two lists. Output:
    slope float, or None when fewer than two points qualify."""
    pts = [(b, s) for b, s in zip(bins_used, scatters) if b > 0 and s > 0]
    if len(pts) < 2:
        return None
    xs = [math.log10(p[0]) for p in pts]
    ys = [math.log10(p[1]) for p in pts]
    return float(np.polyfit(xs, ys, 1)[0])


def fmt(x, nd=3):
    """Print helper for numbers that can be None. Inputs: value, digits.
    Output: string."""
    return "n/a" if x is None else f"{x:.{nd}f}"


def noise_free_checks(cfg):
    """Noise-free single render checks (guide section B, run before cells)
    at the fixed config flux, zero offset, noise off, quantize on. At the
    SNR-bin fluxes the trail light falls under the COM threshold, so the
    fixed flux keeps the blocks alive. The fixed flux clips the rest cells
    (symmetric, no shift) and lets the COM threshold trim the trail wings;
    the discrete prediction does not model that trim. So the symmetric
    states (prediction 0) are checked to 0.05 px, and the mechanism pair
    (nonuniform and its mirror) to 0.25 px plus the sign, which is the
    actual test (LESSONS 1). Inputs: config. Output: list of row dicts.
    Row-axis prediction is 0 for every state (trail runs along +col)."""
    flux = float(cfg["noise_free_check_flux_dn"])
    tol = {  # px, per state; mechanism states get sign + magnitude room
        "rest": 0.05, "uniform_5px": 0.05, "uniform_15px": 0.05,
        cfg["nonuniform_mirror"]["name"]: 0.25, "nonuniform_15px": 0.25,
    }
    centre = truth_grid(cfg)
    gc = star_model.grid_centre(cfg)
    states = [(name, segs) for name, segs in cfg["trail_states"].items()]
    mirror = cfg["nonuniform_mirror"]
    states.append((mirror["name"], mirror["segments"]))
    rows = []
    for state, segments in states:
        for shape in cfg["shapes"]:
            params = cfg[shape + "_params"]
            peak = star_model.shape_peak(params, flux)
            label_px = star_model.label_at(segments)
            pos, w = star_model.path_weights(segments, label_px)
            base = star_model.build_base(params, peak, pos, w, cfg, gc)
            lin = star_model.block_mean(base, cfg) + cfg["background_dn"]  # zero offset
            q = star_model.quantize_clip(lin, cfg["quantize"], tuple(cfg["clip_dn"]))
            com = com_estimates(q[None], cfg)[0]
            pred = star_model.light_mean_bias(segments)  # +col axis, discrete model
            t = tol[state]
            meas_c = float(com[1] - centre[1]) if np.isfinite(com[1]) else None
            meas_r = float(com[0] - centre[0]) if np.isfinite(com[0]) else None
            ok = False
            if meas_c is not None and meas_r is not None:
                ok = abs(meas_r) <= 0.05 and abs(meas_c - pred) <= t
                if abs(pred) >= 0.1:  # sign test only above float noise (1e-16)
                    ok = ok and bool(np.sign(meas_c) == np.sign(pred))
                ok = bool(ok)
            rows.append({"state": state, "shape": shape, "flux_dn": flux,
                         "pred_bias_col_px": pred,
                         "measured_col_minus_label_px": meas_c,
                         "measured_row_minus_label_px": meas_r,
                         "tolerance_px": t, "sign_check": pred != 0.0,
                         "passed": ok})
            print(f"noise-free check {state}/{shape} at {flux:.0f} DN: pred col "
                  f"{pred:+.4f} px, measured col {fmt(meas_c, 4)} px, row "
                  f"{fmt(meas_r, 4)} px, tol {t}, {'PASS' if ok else 'FAIL'}",
                  flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2.json")
    ap.add_argument("--smoke", action="store_true", help="small test, pinned checks, scratch/smoke only")
    ap.add_argument("--seed", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    bins = [int(b) for b in cfg["snr_bins"]]
    mirror_bins = [int(b) for b in cfg["nonuniform_mirror_bins"]]
    n_draws, n_fisher = int(cfg["n_draws"]), int(cfg["n_fisher_draws"])
    if args.smoke:  # small test cells, writes scratch/smoke (v2 item 5, folder rule)
        bins, mirror_bins = [12, 40], [40]
        n_draws, n_fisher = 100, 50
    out_dir = args.out or ("scratch/smoke" if args.smoke else "reports/step2")
    cfg["n_draws"], cfg["n_fisher_draws"] = n_draws, n_fisher
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # Pinned checks gate every cell print (guide section B, LESSONS 1).
    for name, passed, detail in pinned_checks(cfg):
        print(f"pinned {name}: {'PASS' if passed else 'FAIL'} ({detail})", flush=True)
        if not passed:
            raise SystemExit("pinned check failed: fix the plumbing before running cells")
    # Noise-free checks gate the cell loop (guide section B, before plot
    # numbers). See noise_free_checks for the flux and tolerance design.
    checks = noise_free_checks(cfg)
    if not all(row["passed"] for row in checks):
        raise SystemExit("noise-free check failed: fix the model before running cells")
    pred = discrete_predictions(cfg)
    print("discrete COM predictions (noise-free, high SNR, +col axis):",
          {k: round(v, 4) for k, v in pred.items()}, flush=True)
    states = [(name, segs) for name, segs in cfg["trail_states"].items()]
    mirror_name = cfg["nonuniform_mirror"]["name"]
    states.append((mirror_name, cfg["nonuniform_mirror"]["segments"]))
    rows, counts_rows, samples = [], [], {}
    for si, (state, segments) in enumerate(states):
        cell_bins = mirror_bins if state == mirror_name else bins
        for hi, shape_name in enumerate(cfg["shapes"]):
            for bi, snr in enumerate(cell_bins):
                run_cell(cfg, state, segments, shape_name, snr, [si, hi, bi], rows,
                         counts_rows, shape_name == "moffat", samples,
                         weighted=(state == "rest"))  # matched point-star cells only (item 3)
    # Dual test (review item 3): the 10 test cells, fixed before the run
    # (list pinned 24 Oct from reports/step2/table.json cell order: the 6
    # cells holding the 8 bound-beating rows, plus the bin-80 pair cells and
    # their Moffat partners; states order [rest, uniform_5px, uniform_15px,
    # nonuniform_15px, mirror], shapes [gaussian, moffat], sorted bins per
    # state, mirror bins [12, 20, 40, 80]). Same seed indices as the normal
    # cells, so the offsets and noise streams are identical: paired.
    dual_cells = [  # (state, shape, bin, seed_idx)
        ("uniform_15px", "gaussian", 3, [2, 0, 0]), ("uniform_15px", "moffat", 3, [2, 1, 0]),
        ("nonuniform_15px", "gaussian", 3, [3, 0, 0]), ("nonuniform_15px", "moffat", 3, [3, 1, 0]),
        ("nonuniform_15px", "moffat", 5, [3, 1, 2]), ("uniform_15px", "moffat", 5, [2, 1, 2]),
        ("nonuniform_15px", "gaussian", 80, [3, 0, 6]), ("nonuniform_15px", "moffat", 80, [3, 1, 6]),
        ("nonuniform_15px_mirror", "gaussian", 80, [4, 0, 3]), ("nonuniform_15px_mirror", "moffat", 80, [4, 1, 3]),
    ]
    dual_result = None  # v2 item 1: the verdict lives in table.json too
    if not args.smoke:
        dual_result = dual_test(cfg, rows, dual_cells, rows)
    print("--- guide checks ---", flush=True)
    # Check 1: COM scatter vs SNR, log-log slope about -1. Range fixed
    # before the run: bins >= 12 (review item 5c).
    for state, _ in states:
        out1 = []
        for ax in ("row", "col"):
            pts = [(r["snr_bin"], r["scatter"]) for r in rows
                   if r["state"] == state and r["estimator"] == "com" and r["test"] == ""
                   and r["axis"] == ax and r["snr_bin"] >= 12]
            out1.append(f"{ax} slope {fmt(loglog_slope([p[0] for p in pts], [p[1] for p in pts]))}")
        print(f"check1 {state}: " + ", ".join(out1)
              + " (expect about -1; bins >= 12, fixed before the run)", flush=True)
    # Check 2 (v2 review items 1 and 2): the biased bound. The plain bound
    # 1/I holds for unbiased estimators; for a biased one,
    # var >= (1 + db/dtheta)^2 / I (my question was right, the v2 review was
    # wrong - LESSONS 6). db/dtheta comes from stats(): a degree-1 polyfit
    # of the per-draw error (estimate - truth) against the true offset
    # (drawn uniformly over +/-1 px); the slope is db/dtheta. Criteria fixed
    # before the run, replacing the v2 small-bias rule that cut away the 8
    # explained rows (LESSONS 9): check scatter >= |1 + slope| * CRB in every
    # row, all bins, test rows included; rows below that bound are bugs.
    # Report the slope per COM row. Expected outcome, measured BEFORE the
    # run from scratch/table_v2_old.json: zero bugs (the lowest non-COM
    # ratio there is 1.0297, and every beating COM row has slope about -1, so
    # its biased bound is about 0 and its scatter may sit below the plain
    # bound without being a bug).
    normal_rows = [r for r in rows if r["test"] == ""]
    print("check2 (biased bound, every row, all bins, fixed before the run):", flush=True)

    def biased_bound(r):  # |1 + slope| * CRB; None when slope or CRB is missing
        if r["bias_slope"] is None or r["crlb_px"] <= 0.0:
            return None
        return abs(1.0 + r["bias_slope"]) * r["crlb_px"]

    # A row is a bug only if its scatter sits below the bound by more than 3
    # standard errors of the scatter. The SE of a standard deviation is
    # scatter / sqrt(2(n-1)) for Gaussian errors: about 1.6% at n = 2000, 7%
    # at n = 100. (The scatter_se field of stats() is std / sqrt(n), the SE
    # of a mean, so it is not used here.) Without this margin, an estimator
    # AT the bound falls below it in half the rows by sampling noise alone
    # (Claude, 25 Oct).
    def scatter_sd_se(r):
        return r["scatter"] / math.sqrt(2.0 * max(r["n"] - 1, 1))

    bugs = [r for r in rows if biased_bound(r) is not None and biased_bound(r) > 1e-9
            and r["scatter"] + 3.0 * scatter_sd_se(r) < biased_bound(r)]
    print(f"  rows below the biased bound by more than 3 SE, all rows incl. test rows (BUGS): {len(bugs)} "
          f"(expected 0, measured before the run)", flush=True)
    for r in sorted(bugs, key=lambda r: (r["snr_bin"], r["state"], r["shape"], r["axis"])):
        print(f"    BUG {r['state']}/{r['shape']}/snr{r['snr_bin']}/{r['axis']} "
              f"{r['estimator']}[{r['test'] or 'normal'}]: scatter {fmt(r['scatter'], 4)} px "
              f"< bound {fmt(biased_bound(r), 4)} px, slope {fmt(r['bias_slope'], 3)}",
              flush=True)
    print("  slope db/dtheta report, per normal COM row (test-row slopes are in table.json):",
          flush=True)
    for r in sorted([r for r in normal_rows if r["estimator"] == "com"],
                    key=lambda r: (r["state"], r["shape"], r["snr_bin"], r["axis"])):
        print(f"    {r['state']}/{r['shape']}/snr{r['snr_bin']}/{r['axis']}: slope "
              f"{fmt(r['bias_slope'], 3)} +/- {fmt(r['bias_slope_se'], 3)}, scatter "
              f"{fmt(r['scatter'], 3)} px, biased bound {fmt(biased_bound(r), 3)} px, "
              f"plain-bound ratio {fmt(r['ratio_scatter_over_crlb'])}", flush=True)
    # The rows that beat the PLAIN bound (the v2 beater list, kept for the
    # record). Each one is explained now: a COM shrunk toward the patch
    # centre has db/dtheta about -1, so its biased bound is about 0 (the
    # review's answer to my LESSONS-6 question; the mechanism is tested in
    # dual_test).
    beaters = [r for r in normal_rows if r["ratio_scatter_over_crlb"] is not None
               and r["ratio_scatter_over_crlb"] < 1.0 and r["estimator"] == "com"]
    print(f"  rows beating the plain bound: {len(beaters)} (all COM, all at SNR 3-5, each "
          f"explained by the biased bound, slope about -1):", flush=True)
    for r in sorted(beaters, key=lambda r: (r["snr_bin"], r["state"], r["shape"], r["axis"])):
        print(f"    {r['state']}/{r['shape']}/snr{r['snr_bin']}/{r['axis']}: scatter/plain-bound "
              f"{fmt(r['ratio_scatter_over_crlb'])}, slope {fmt(r['bias_slope'], 3)}+/-{fmt(r['bias_slope_se'], 3)}, "
              f"biased bound {fmt(biased_bound(r), 3)} px, bias {fmt(r['bias'], 4)}+/-{fmt(r['bias_se'], 4)} px",
              flush=True)
    minr = min((r["ratio_scatter_over_biased_bound"] for r in rows
                if r["ratio_scatter_over_biased_bound"] is not None), default=None)
    print(f"  min scatter/biased-bound over all rows: {fmt(minr)} (plot rms_vs_snr.png, "
          f"written by src/plot_step2.py)", flush=True)
    # Check 3: matched fits close to the bound at bins 20-80 (expect about
    # 1). Unweighted vs weighted shown together (review 5b): at high SNR the
    # star's own shot noise varies pixel to pixel, so an unweighted fit
    # cannot reach the bound; the weighted fit can.
    for shape_name, est in (("gaussian", "gaussian_on_gaussian"), ("moffat", "moffat_on_moffat")):
        for est_name in (est, est + "_weighted"):
            parts = [f"snr{b} " + ",".join(
                f"{r['state']}:{r['axis']}={fmt(r['ratio_scatter_over_crlb'])}" for r in rows
                if r["shape"] == shape_name and r["estimator"] == est_name
                and r["test"] == "" and r["snr_bin"] in (20, 40, 80))
                for b in sorted(set(bins) & {20, 40, 80})]
            if any("snr" in p for p in parts):
                print(f"check3 {est_name} scatter/CRLB: " + " ".join(parts), flush=True)
    print("check3 note: unweighted rows match the previous run exactly (seed fixed); the "
          "weighted fit runs on the matched rest cells only (v2 item 3) with the standard "
          "call (LESSONS 13), so its numbers differ from the v2 ones (the v2 SLSQP machinery "
          "is gone, my Q1 and Q5 are moot). Moffat weighted: the v2 run left these rows "
          "empty; the standard call should run - if it still fails at bins 20-80, find the "
          "cause before quoting.", flush=True)
    # Check 4: nonuniform bias magnitude and the mirror sign flip (the
    # mechanism test; the guide says slow end, the discrete model says fast end).
    hi_bin = max(bins)
    for st in ("nonuniform_15px", mirror_name):
        for r in rows:
            if r["state"] == st and r["estimator"] == "com" and r["axis"] == "col" \
                    and r["snr_bin"] == hi_bin and r["shape"] == "gaussian" and r["test"] == "":
                print(f"check4 {st} COM col bias at snr{hi_bin}: bias {fmt(r['bias'], 4)} "
                      f"+/- {fmt(r['bias_se'], 4)} px, discrete prediction "
                      f"{pred[st]:+.4f} px (noise-free check above)", flush=True)
    print("check4: the sign flip nonuniform vs mirror IS the mechanism test; numbers settle it",
          flush=True)
    print("symmetry (review 11): the two broken symmetries are resolved by dual_test above "
          "(verdict printed there and stored in table.json under dual_test; the consequence "
          "for the project is in the log entry)", flush=True)
    print("real data: error vs SNR has no real-data counterpart (no ground truth on real "
          "frames); the tetra3 column comes in step 6. The SNR axis uses the MODEL noise "
          "(placeholder): step 1 measured pooled std 1.825 DN vs model 1.4753 DN, so real "
          "points sit about x1.24 right on the SNR axis; slopes and ratios are unaffected.",
          flush=True)
    nlines = len(Path(__file__).read_text().splitlines())
    print(f"lines in this file: {nlines} (rule: over 300, flagged for the log, split discussed)",
          flush=True)
    if args.smoke:
        print("--- smoke criteria (the full run is next) ---", flush=True)

        def cell_row(state, shape, est, ax, b, test=""):
            m = [r for r in rows if r["state"] == state and r["shape"] == shape and
                 r["estimator"] == est and r["axis"] == ax and r["snr_bin"] == b and r["test"] == test]
            return m[0] if m else None

        c12, c40 = cell_row("rest", "gaussian", "com", "col", 12), cell_row("rest", "gaussian", "com", "col", 40)
        ok1 = c12 is not None and c40 is not None and c12["scatter"] > c40["scatter"]
        print(f"smoke scatter decreases with SNR (COM rest/gaussian, noise): bin12 "
              f"{fmt(c12['scatter'], 4)} > bin40 {fmt(c40['scatter'], 4)}: {'PASS' if ok1 else 'FAIL'}",
              flush=True)
        f40 = cell_row("rest", "gaussian", "gaussian_on_gaussian", "col", 40)
        ok2 = f40 is not None and f40["ratio_scatter_over_crlb"] is not None \
            and 0.7 <= f40["ratio_scatter_over_crlb"] <= 1.5
        f40w = cell_row("rest", "gaussian", "gaussian_on_gaussian_weighted", "col", 40)
        print(f"smoke gaussian fit close to bound at bin 40 (range 0.7-1.5): "
              f"unweighted {fmt(None if f40 is None else f40['ratio_scatter_over_crlb'])}, "
              f"weighted {fmt(None if f40w is None else f40w['ratio_scatter_over_crlb'])}: "
              f"{'PASS' if ok2 else 'FAIL'}", flush=True)
        # Standard-call equivalence test (LESSONS 13): one cell runs the plain
        # fit and the weighted fit with sigma forced to 1 on identical data.
        # With sigma = 1 both calls minimise the same function, so the
        # positions must match within the solver tolerance (1e-3 px; the v2
        # hand-written machinery disagreed by 0.003 px on such data).
        run_cell(cfg, "rest", dict(states)["rest"], "gaussian", 40, [9, 9, 0], rows,
                 counts_rows, False, None, weighted=True, unity=True, label=" SMOKE unity")
        up = cell_row("rest", "gaussian", "gaussian_on_gaussian_weighted", "col", 40, " SMOKE unity")
        pp = cell_row("rest", "gaussian", "gaussian_on_gaussian", "col", 40, " SMOKE unity")
        ok3 = up is not None and pp is not None and abs(up["bias"] - pp["bias"]) < 1e-3
        print(f"smoke unity weights equal the plain fit (same cell, same data, bias "
              f"difference): {fmt(None if up is None or pp is None else abs(up['bias'] - pp['bias']), 6)} px "
              f"(tol 1e-3, solver tolerance): {'PASS' if ok3 else 'FAIL'}", flush=True)
        w40 = cell_row("rest", "moffat", "moffat_on_moffat_weighted", "col", 40)
        print(f"smoke weighted Moffat path runs (standard least_squares, matched rest "
              f"cells only, v2 item 3): bin40 ratio "
              f"{fmt(None if w40 is None else w40['ratio_scatter_over_crlb'])} (empty rows = "
              f"non-convergence: find the cause before quoting)", flush=True)
        est_keys = {"com", "gaussian_on_gaussian", "moffat_on_moffat", "gaussian_on_moffat"}
        worst = max((c[k] for c in counts_rows for k in est_keys if k in c), default=0)
        print(f"smoke dropped estimates sane (< 5% of draws): worst {worst} of {n_draws}: "
              f"{'PASS' if worst < 0.05 * n_draws else 'FAIL'}", flush=True)
    table = {"config_file": args.config, "seed": cfg["seed"], "smoke": args.smoke,
             "script": Path(__file__).name, "script_lines": nlines,
             "script_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                               time.gmtime(os.stat(__file__).st_mtime)),
             "n_draws": n_draws, "n_fisher_draws": n_fisher,
             "bins": bins, "mirror_bins": mirror_bins,
             "sigma_pix_dn": sigma_pix_dn(cfg),
             "s_px_gaussian": s_sigma_px(cfg["gaussian_params"]),
             "s_px_moffat": s_sigma_px(cfg["moffat_params"]),
             "flux_per_bin_dn": {str(b): flux_for_snr(b, cfg)
                                 for b in sorted(set(bins) | set(mirror_bins))},
             "noise_placeholder": "noise model is the step-1 placeholder; step 3 fits it; "
                                  "slopes and ratios are the result, absolute values provisional",
             "rows": rows, "checks": checks, "counts": counts_rows,
             "discrete_predictions": pred,
             "samples": {f"{s}|{sh}": g.tolist() for (s, sh), g in samples.items()},
             "dual_test": dual_result}
    with open(f"{out_dir}/table.json", "w") as f:
        json.dump(table, f, indent=1)
    if not args.smoke:  # provenance copy: the config only (review item 4; the
        shutil.copy(args.config, out_dir)  # script line count and mtime are in table.json)


if __name__ == "__main__":
    main()
