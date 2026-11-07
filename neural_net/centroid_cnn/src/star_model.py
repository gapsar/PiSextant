"""star_model.py — the renderer: PSF shape, normalization, noise, tone, quantize.

Split out of render_star.py on the step 1 (v2) review, item 4. The trail
renderer (path, sub-steps, light kernel, sampling) and the noise-and-
quantize step moved here from error_vs_snr.py on the step 2 review, item 4,
because step 5's generator needs them. Function bodies are unchanged by the
moves, and the pinned check numbers did not change (see WORKLOG). The two
pinned checks on this file's path live here (moved with the code they
test); error_vs_snr.py calls them in its pinned-checks gate.

This file does not read the config file: it takes the config dict as an
argument (review item 3 of step 1 v2). The units of every input and output
are in the function docstrings. Only installed packages are used (rule
4.1.13).
"""

import math

import numpy as np
from scipy.ndimage import map_coordinates  # trail sampling only (step 2 split)

PAD = 3  # px of margin so per-draw shifts stay inside the sampled field


def psf_unit_shape(u, v, p):
    """PSF shape function, peak 1, at offsets (u, v) from the PSF centre, px.

    Inputs: u, v: broadcastable coordinate arrays, px, along the rotated
    axes (angle_deg turns the frame; the major axis is at angle 0).
    p: PSF parameter dict: shape ("gaussian" or "moffat"), major_px
    (major-axis width; sigma for a Gaussian, alpha for a Moffat),
    axis_ratio (>= 1), angle_deg, beta (Moffat only).
    Output: array of the same broadcast shape, values in (0, 1].
    Not separable when the ellipse is rotated, so evaluate on the 2-D grid.
    """
    minor = p["major_px"] / p["axis_ratio"]
    if p["shape"] == "gaussian":
        q = (u * u) / (2.0 * p["major_px"] * p["major_px"]) + (v * v) / (2.0 * minor * minor)
        return np.exp(-q)
    q = (u * u) / (p["major_px"] * p["major_px"]) + (v * v) / (minor * minor)
    return (1.0 + q) ** (-p["beta"])


def psf_normalization(p):
    """Analytic integral over the infinite plane of the peak-1 shape function.

    Makes the input flux mean the total light of the PSF, so the flux check
    means something for every model (review item 2). Input: PSF parameter
    dict. Output: area factor, px^2. Moffat form needs beta > 1.
    """
    minor = p["major_px"] / p["axis_ratio"]
    if p["shape"] == "gaussian":
        return 2.0 * math.pi * p["major_px"] * minor
    return math.pi * p["major_px"] * minor / (p["beta"] - 1.0)


def add_noise(linear_dn, noise_params, rng):
    """Add photon shot noise and the read-noise floor (guide 4.1.10).

    Shot noise is Poisson in ELECTRONS for star light plus sky. The
    read noise is a Gaussian floor in DN. White noise only for step 1
    (correlated noise comes in the dataset steps).

    Inputs: linear_dn: array, DN, before noise. noise_params: noise dict
    from the config (placeholders for step 1, see its note).
    rng: numpy Generator for this check.
    Output: array, DN, before quantization.
    """
    gain = noise_params["electrons_per_dn"]
    counts_e = rng.poisson(linear_dn * gain)  # lambda is always > 0 here
    read = rng.normal(0.0, noise_params["read_noise_sigma_dn"], linear_dn.shape)
    return counts_e / gain + read


def tone_identity(linear_dn):
    """Apply the tone curve. Identity for step 1 (guide 4.1.9)."""
    return linear_dn


def quantize_clip(linear_dn, quantize, clip):
    """Round to integers and clip, as the 8-bit camera does (guide 4.1.10)."""
    if not quantize:
        return linear_dn
    lo, hi = clip
    return np.clip(np.round(linear_dn), lo, hi)


def render_star(pos_rc, flux, psf_params, background, noise_params, rng,
                patch_px, s, noise_on=True, quantize=True, clip=(0.0, 255.0)):
    """Render one star into one patch, in the config pipeline order.

    Order (guide 4.1.9): linear photons -> background -> noise -> tone
    curve (identity) -> quantize and clip. The PSF is evaluated on a fine
    grid (s samples per pixel side) and each s x s block is averaged:
    reshape to (patch, s, patch, s), mean over axes 1 and 3. This works
    for any PSF function and for trails later (review item 2).

    Inputs (review item 3; the function does not read the config):
        pos_rc: (row, col) star position, px, corner coords, patch-local;
            None means no star. flux: total PSF light, DN, on the infinite
            plane. psf_params: dict from the config psf_models.
        background: sky level, DN. noise_params: noise dict from the config.
        rng: numpy Generator for this check. patch_px, s: patch size and
        supersample per side. noise_on, quantize: bool. clip: (lo, hi) DN.
    Output: (float array before quantization, float array after
    quantization and clipping), both (patch_px, patch_px), DN.
    """
    if flux > 0.0:
        angle = math.radians(psf_params["angle_deg"])
        # Fine-grid coordinates inside each pixel: the middle of each slice,
        # so the block mean is the pixel integral (rule 4.1.2).
        idx = np.arange(patch_px, dtype=np.float64)
        fine = idx[:, None] + (np.arange(s)[None, :] + 0.5) / s
        row_off = fine.reshape(patch_px, s, 1, 1) - pos_rc[0]  # -> (patch, s, 1, 1)
        col_off = fine.reshape(1, 1, patch_px, s) - pos_rc[1]  # -> (1, 1, patch, s)
        u = col_off * math.cos(angle) + row_off * math.sin(angle)
        v = -col_off * math.sin(angle) + row_off * math.cos(angle)
        shape = psf_unit_shape(u, v, psf_params)  # broadcast -> (patch, s, patch, s)
        peak = flux / psf_normalization(psf_params)
        star = peak * shape.mean(axis=(1, 3))  # block mean = pixel integral
    else:
        star = np.zeros((patch_px, patch_px), dtype=np.float64)
    linear = star + background
    noisy = add_noise(linear, noise_params, rng) if noise_on else linear
    toned = tone_identity(noisy)
    return toned, quantize_clip(toned, quantize, clip)


# --- Trail renderer (moved from error_vs_snr.py, step 2 review item 4) ---
# Step 5's generator needs these functions, so they live here next to
# render_star. Bodies unchanged by the move (same habit as the 2nd split).
# Labels use the CONVENTIONS centre coords; trails run along +col
# (CONVENTIONS). Inputs take the config dict as an argument.

def label_at(segments):
    """Label position from the path start, px = mid-time = after half the
    equal time steps (even counts here; half-a-segment rule added for odd
    counts). Input: segment list, px moved per equal time step. Output: px
    from the start."""
    n = len(segments)
    whole, half = divmod(n, 2)
    return float(sum(segments[:whole]) + (0.5 * segments[whole] if half else 0.0))


def path_weights(segments, label_px):
    """Build the light kernel on the 0.1 px path grid: per sub-step, its
    centre position from the label (px, the trail axis is +col) and its
    light fraction (equal time steps share the light equally, guide 4.1.5;
    a zero-distance segment is one point position at its place). Inputs:
    segment list, px moved per equal time step; label position from the
    path start, px. Output: (positions, weights) arrays, weights sum to 1."""
    n = len(segments)
    positions, weights = [], []
    cum = 0.0
    for seg in segments:
        if seg == 0.0:  # rest: one point position at its place, weight 1/n
            positions.append(cum)
            weights.append(1.0 / n)
            continue
        m = max(int(round(seg * 10)), 1)  # sub-steps in this segment (min 1)
        for k in range(m):
            positions.append(cum + (k + 0.5) * 0.1)
            weights.append(1.0 / n / m)
        cum += seg
    return np.asarray(positions) - label_px, np.asarray(weights)


def light_weights(segments):
    """The light kernel weights (guide 4.1.5), per 0.1 px sub-step of the
    path, matching path_weights' sub-steps in order: equal time steps share
    the light equally; a zero-distance segment is one point at its place;
    a segment is split into 10 sub-steps per px (min 1). Input: segment
    list, px moved per equal time step. Output: weights array, sum 1."""
    n = len(segments)
    weights = []
    for seg in segments:
        if seg == 0.0:  # rest: one point position at its place, weight 1/n
            weights.append(1.0 / n)
            continue
        m = max(int(round(seg * 10)), 1)  # sub-steps in this segment (min 1)
        weights.extend([1.0 / n / m] * m)
    return np.asarray(weights)


def light_mean_bias(segments):
    """Discrete-model prediction: noise-free infinite-SNR COM minus label
    (the light mean minus the mid-time position), px, on the trail axis
    (+col). Input: segment list. Output: float."""
    pos, w = path_weights(segments, label_at(segments))
    return float((pos * w).sum())


def fine_grids(cfg):
    """Cell-centre px coordinates over the padded fine field, 2-D flat
    view: fine cell I on an axis covers sub-cell a of coarse pixel I // s,
    its centre at px I // s + (a + 0.5) / s, corner coords (render_star
    way). Inputs: config dict. Output: (row_px, col_px) broadcastable
    (G, G) arrays, G = (patch + 2*PAD) * s."""
    g = cfg["patch_px"] + 2 * PAD
    s = cfg["supersample_per_side"]
    px = (np.arange(g * s) + 0.5) / s  # cell-centre px: cell (i,a) = i + (a+0.5)/s, render_star way
    row_px = px[:, None] * np.ones((1, g * s))
    col_px = px[None, :] * np.ones((g * s, 1))
    return row_px, col_px


def eval_unit(params, row_px, col_px, centre_px, shift_px):
    """Peak-1 shape function evaluated at (grid - centre - shift), angle 0
    (CONVENTIONS: angle 0 puts the major axis along +col, so u is the col
    difference and v the row difference; same mapping as render_star).
    Inputs: PSF params, broadcast px coordinate grids, centre px, shift px
    (row, col). Output: (G, G) array, peak 1."""
    v = row_px - (centre_px[0] + shift_px[0])
    u = col_px - (centre_px[1] + shift_px[1])
    return psf_unit_shape(u, v, params)


def block_mean(fine2d, cfg):
    """Block-mean the flat fine field to pixels: reshape (g, s, g, s), mean
    over axes 1 and 3, then crop the pad (render_star habit, rule 4.1.2).
    Inputs: flat fine field (G, G), config dict. Output: (patch, patch)
    array, same units."""
    p, s = cfg["patch_px"], cfg["supersample_per_side"]
    g = p + 2 * PAD
    return fine2d.reshape(g, s, g, s).mean(axis=(1, 3))[PAD:PAD + p, PAD:PAD + p]


def sample_blocks(fine2d, cfg, offs):
    """Fine field sampled at every cell centre, bilinear: the scene moves
    by +off, so output cell p shows the source value at p - off (sign
    pinned by step-2 pinned check 2, not derived; on a smooth field the
    interpolation error is far under 0.01 px, negligible against the shot
    noise). Inputs: flat fine field (G, G) used as (g*s, g*s), config dict,
    offsets (N, 2) px (row, col). Output: (N, patch, patch) block-mean
    arrays, source units."""
    s = cfg["supersample_per_side"]
    g = cfg["patch_px"] + 2 * PAD
    G = g * s
    base = np.arange(G)  # index coords of the cell centres on each axis
    coords_row = base[None, :] - np.asarray(offs)[:, 0][:, None] * s  # (N, G)
    coords_col = base[None, :] - np.asarray(offs)[:, 1][:, None] * s  # (N, G)
    out = np.empty((len(offs), cfg["patch_px"], cfg["patch_px"]))
    for n in range(len(offs)):  # one call per draw; field smooth, order 1
        rc = coords_row[n][:, None] * np.ones((1, G))  # broadcast to output shape
        cc = coords_col[n][None, :] * np.ones((G, 1))
        out[n] = block_mean(map_coordinates(fine2d, [rc, cc], order=1, mode="nearest"), cfg)
    return out


def build_base(params, peak, positions, weights, cfg, centre_px):
    """Mean (noise-free, background-free) fine field at zero offset = the
    weighted direct sum of shifted peak-1 shape functions over the
    sub-steps (guide 4.1.5, the reference). Inputs: PSF params, peak DN,
    sub-step arrays (positions px from label along +col), config dict,
    centre px of the label on the grid. Output: (G, G) fine field, DN."""
    row_px, col_px = fine_grids(cfg)
    total = np.zeros((row_px.shape[0], col_px.shape[1]))
    for pos, w in zip(positions, weights):  # sub-step at (0, pos) px from label
        total += w * eval_unit(params, row_px, col_px, centre_px, (0.0, pos))
    return peak * total


def shape_peak(psf_params, flux):
    """Peak value of the peak-1 shape function for a given total light.
    Inputs: PSF params dict, flux DN. Output: peak, DN."""
    return flux / psf_normalization(psf_params)


def grid_centre(cfg):
    """Where the label sits on the PADDED fine grid: patch centre plus PAD,
    in fine_grids px labelling (block_mean crops the pad, so a shape put at
    this grid point shows at the patch centre). Input: config dict. Output:
    (row, col) px, field coords. Truth and stats use truth_grid instead (it
    stays in the estimator script)."""
    c = cfg["center_pixel_index"] + 0.5
    return (c + PAD, c + PAD)


def noise_and_quantize(mean_blocks, cfg, rng):
    """Pipeline order guide 4.1.9 with noise on: linear photons plus
    background (blocks arrive background-free; background added here) ->
    noise -> tone (identity) -> quantize and clip. Inputs: mean blocks (N,
    patch, patch) DN star light only, config dict, rng. Output: (float DN
    before quantization, quantized) arrays, (N, patch, patch). (Moved from
    error_vs_snr.py with the trail code, step 2 review item 4.)"""
    linear = mean_blocks + cfg["background_dn"]
    noisy = add_noise(linear, cfg["noise"], rng)
    toned = tone_identity(noisy)
    return toned, quantize_clip(toned, cfg["quantize"], tuple(cfg["clip_dn"]))


def check_zero_trail_equals_render_star(cfg):
    """Pinned check 1 of step 2 (moved with the trail code it tests): a 0 px
    trail (one sub-step, weight 1) through this file's trail path must give
    exactly render_star's patch, noise off, same position, to 1e-9 DN.
    Input: config dict. Output: (name, passed, detail) like the step-2
    pinned checks; both paths are noise-free, so no rng is used."""
    p = cfg["patch_px"]
    params = cfg["gaussian_params"]
    base = build_base(params, shape_peak(params, 1000.0), [0.0], [1.0], cfg, grid_centre(cfg))
    got = block_mean(base, cfg) + cfg["background_dn"]
    centre = (p // 2 + 0.5, p // 2 + 0.5)  # render_star patch-local coords
    ref = render_star(centre, 1000.0, params, cfg["background_dn"],
                      cfg["noise"], None, p, cfg["supersample_per_side"], noise_on=False)[0]
    d = float(np.max(np.abs(got - ref)))
    return ("check1_rest_equals_render_star", d < 1e-9, f"max diff {d:.2e} DN")


def check_zero_trail_weighted_kernel_equals_direct(cfg):
    """Pinned check 1b of step 2 (added at the step 2 review, item 4): a
    0 px trail run as ten identical positions carrying the light kernel
    weights (1/10 each, guide 4.1.5) must give the same noise-free patch as
    one position with weight 1, to 1e-9 DN. It checks the kernel weighting,
    not the sampling. Input: config dict. Output: (name, passed, detail)
    like the other pinned checks; both paths are noise-free, no rng used."""
    params = cfg["gaussian_params"]
    peak = shape_peak(params, 1000.0)
    gc = grid_centre(cfg)
    w = light_weights(cfg["trail_states"]["rest"])
    got = block_mean(build_base(params, peak, [0.0] * len(w), w, cfg, gc), cfg)
    ref = block_mean(build_base(params, peak, [0.0], [1.0], cfg, gc), cfg)
    d = float(np.max(np.abs(got - ref)))
    return ("check1b_weighted_kernel_equals_direct", d < 1e-9, f"max diff {d:.2e} DN")
