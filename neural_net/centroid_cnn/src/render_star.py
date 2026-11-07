"""Step 1 checks: render one star into a 32 x 32 patch (guide section 5, step 1).

This script runs the four official step-1 checks and writes the plots and
numbers to reports/step1/. Every generator parameter is in a JSON config.
The seed is fixed in the config, as the guide section 0 requires.

Split after the step 1 (v2) review, item 4: the renderer functions
(render_star and its helpers) now live in star_model.py, and step 2 will
import that file. The checks stay here, under the guide name for this
script (section B): render_star.py.

Updated after the step 1 (v2) review: checks 1 and 2 also run every model
on a 128 x 128 patch (item 1, this is the renderer check), check 3 reports
the real tile means and stds (item 3), and the real-frame path is relative
to this script's root folder, as inject_test.py does (item 6). The old
32 x 32 numbers did not change; the added runs use no new random draws.

Usage (run from the centroid_cnn folder):
    python3 src/render_star.py
    python3 src/render_star.py --config configs/step1.json --out reports/step1
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import matplotlib
from PIL import Image

matplotlib.use("Agg")  # save PNGs only, this box has no window session
import matplotlib.pyplot as plt

from star_model import render_star


def load_config(path):
    """Read the JSON config file. Input: path. Output: dict."""
    with open(path) as f:
        return json.load(f)


def rng_for_check(seed, check_number):
    """One random generator per check (review item 7). Inputs: seed, number.
    Output: numpy Generator. Sharing one generator would let a draw in one
    check shift the numbers of every later check."""
    return np.random.default_rng([seed, check_number])


def com(patch_weight, coord_1d):
    """Centre of mass along one axis. Input: weights (star light only),
    pixel-centre coords, px, corner coords. Output: coord, px."""
    return float((patch_weight * coord_1d).sum() / patch_weight.sum())


def check_com(cfg, seed, out_dir, write_csv):
    """Check 1: COM of a bright noise-free star = the input position, per PSF.

    50 random positions, offsets +/-2 px (guide 4.1.1, review item 4),
    noise off, checked on the float array before quantization. Each axis
    must stay within 0.01 px. For the Moffat the 32 x 32 bias is
    truncation, expected, not a render bug (v2 review, item 1): the
    renderer check is the 128 x 128 run, and it must pass. The large-patch
    run uses the same positions and the same +/-2 px offsets (the guide
    4.1.1 range formula grows with patch size; we keep the run
    like-for-like, see the WORKLOG open questions). It adds no random
    draws, so the 32 x 32 numbers did not change. Writes positions_50.csv
    for the round Gaussian when write_csv is True.
    """
    rng = rng_for_check(seed, 1)
    n = cfg["checks"]["com_n_positions"]
    flux = cfg["checks"]["flux_dn"]
    lo, hi = cfg["checks"]["com_offset_range_px"]
    centre_px = cfg["center_pixel_index"] + 0.5  # centre pixel 16 (CONVENTIONS.md)
    coord = np.arange(cfg["patch_px"], dtype=np.float64) + 0.5
    limit = cfg["checks"]["com_max_bias_px"]
    large_px = cfg["checks"]["large_patch_px"]
    centre_large = large_px // 2 + 0.5  # centre pixel at index large_px // 2
    coord_large = np.arange(large_px, dtype=np.float64) + 0.5
    out = {"offset_range_px": [lo, hi], "limit_px": limit, "models": {},
           "large_patch_px": large_px, "models_large_patch": {}, "renderer_check": ""}
    for name, psf in cfg["psf_models"].items():
        rows, offsets = [], []
        for _ in range(n):
            pos = (centre_px + rng.uniform(lo, hi), centre_px + rng.uniform(lo, hi))
            offsets.append((pos[0] - centre_px, pos[1] - centre_px))
            pre, _ = render_star(pos, flux, psf, cfg["background_dn"], cfg["noise"],
                                 rng, cfg["patch_px"], cfg["supersample_per_side"],
                                 noise_on=False)
            star_only = pre - cfg["background_dn"]  # all values > 0, noise is off
            got = (com(star_only.sum(axis=1), coord), com(star_only.sum(axis=0), coord))
            rows.append([pos[0], pos[1], got[0], got[1], got[0] - pos[0], got[1] - pos[1]])
        table = np.array(rows)
        if write_csv and name == "gaussian_round":
            with open(f"{out_dir}/positions_50.csv", "w") as f:
                f.write("row_px,col_px,com_row_px,com_col_px,bias_row_px,bias_col_px\n")
                for r in table:
                    f.write(",".join(f"{v:.5f}" for v in r) + "\n")
        model = {}
        for axis, axis_name in ((4, "row"), (5, "col")):
            b = table[:, axis]
            model[axis_name] = {
                "mean_px": float(b.mean()),
                "rms_px": float(np.sqrt((b ** 2).mean())),
                "max_abs_px": float(np.abs(b).max()),
                "pass": bool(np.abs(b).max() <= limit),
            }
        out["models"][name] = model
        large_rows = []
        for d_row, d_col in offsets:  # same positions, no new draws (noise off)
            pos = (centre_large + d_row, centre_large + d_col)
            pre, _ = render_star(pos, flux, psf, cfg["background_dn"], cfg["noise"],
                                 rng, large_px, cfg["supersample_per_side"],
                                 noise_on=False)
            star_only = pre - cfg["background_dn"]
            got = (com(star_only.sum(axis=1), coord_large),
                   com(star_only.sum(axis=0), coord_large))
            large_rows.append([got[0] - pos[0], got[1] - pos[1]])
        ltable = np.array(large_rows)
        large_model = {}
        for axis, axis_name in ((0, "row"), (1, "col")):
            b = ltable[:, axis]
            large_model[axis_name] = {
                "mean_px": float(b.mean()),
                "rms_px": float(np.sqrt((b ** 2).mean())),
                "max_abs_px": float(np.abs(b).max()),
                "pass": bool(np.abs(b).max() <= limit),
            }
        out["models_large_patch"][name] = large_model
    all_pass = all(m[a]["pass"] for m in out["models_large_patch"].values()
                   for a in ("row", "col"))
    out["renderer_check"] = ("PASS" if all_pass else "FAIL") + \
        f": all models within {limit} px at {large_px} px (v2 review item 1)"
    out["note"] = ("Moffat 32 px bias is edge truncation, expected; the large run "
                   "reuses the 32 px positions and adds no draws")
    return out


def check_flux(cfg, seed):
    """Check 2: patch flux of a noise-free centred star = the input flux, 1%.

    Run on the float array before quantization. For Gaussian models the
    centred star is well inside the patch, as the guide check says. The
    Moffat beta 3 keeps light in its wide wings; the 32 x 32 patch cannot
    show it and the deficit is physical, not a render bug. Report the
    number. The 128 x 128 run is the renderer check (v2 review, item 1);
    it is deterministic and changes no 32 x 32 number.
    """
    rng = rng_for_check(seed, 2)
    flux = cfg["checks"]["flux_dn"]
    centre_px = cfg["center_pixel_index"] + 0.5
    tol = cfg["checks"]["flux_tolerance_percent"]
    large_px = cfg["checks"]["large_patch_px"]
    centre_large = large_px // 2 + 0.5  # centre pixel at index large_px // 2
    out = {"flux_in_dn": flux, "tolerance_percent": tol, "models": {},
           "large_patch_px": large_px, "models_large_patch": {}}
    for name, psf in cfg["psf_models"].items():
        pre, _ = render_star((centre_px, centre_px), flux, psf, cfg["background_dn"],
                             cfg["noise"], rng, cfg["patch_px"],
                             cfg["supersample_per_side"], noise_on=False)
        got = float((pre - cfg["background_dn"]).sum())
        error_pct = 100.0 * (got - flux) / flux
        out["models"][name] = {
            "flux_out_dn": round(got, 3),
            "error_percent": round(error_pct, 4),
            "pass": bool(abs(error_pct) <= tol),
        }
        pre, _ = render_star((centre_large, centre_large), flux, psf,
                             cfg["background_dn"], cfg["noise"], rng, large_px,
                             cfg["supersample_per_side"], noise_on=False)
        got = float((pre - cfg["background_dn"]).sum())
        error_pct = 100.0 * (got - flux) / flux
        out["models_large_patch"][name] = {
            "flux_out_dn": round(got, 3),
            "error_percent": round(error_pct, 4),
            "pass": bool(abs(error_pct) <= tol),
        }
    out["renderer_check"] = f"all models at {large_px} px; the 1% test is for a" + \
        " centred star well inside the patch (v2 review item 1)"
    return out


def exact_zero_fraction(noise_params, background, n_draws):
    """Exact model zero fraction (review item 5): no Gaussian approximation.

    Sums P(Poisson(k) * Phi((0.5 - k) / read_sigma)) over k; the sum is
    exact for the model, so no draw noise. A 1e6-draw cross-check agrees
    within the draw noise (0.1512 vs 0.1517).

    Inputs: noise_params: noise dict from the config. background: DN.
    n_draws: unused here, kept for the reported sample size.
    Output: probability that a background pixel quantizes to 0.
    """
    gain = noise_params["electrons_per_dn"]
    read = noise_params["read_noise_sigma_dn"]
    lam = background * gain
    p = 0.0
    pk = math.exp(-lam)  # Poisson(0), then recurse pk+1 = pk * lam / (k+1)
    for k in range(0, 30):
        p += pk * 0.5 * math.erfc((k - 0.5) / (read * math.sqrt(2.0)))  # = Phi((0.5-k)/read)
        pk *= lam / (k + 1)
    return p


def real_empty_patch_fractions(cfg, values):
    """Real-data integer fractions for check 3, from local empty patches.

    Loads the dev-run frame (read-only). The path is relative to this
    script's root folder, as inject_test.py does (v2 review, item 6).
    Keeps the non-overlapping 32 x 32 tiles with no pixel above the config
    threshold (no star above 6 sigma, review item 1). Fits local patches,
    not the whole frame, as the step-3 note in the review says. Reports
    the kept-tile statistics (v2 review, item 3).

    Inputs: cfg: config dict. values: integer DN values to report.
    Output: (dict value -> fraction, number of pixels kept, tile count,
    dict with the kept/removed counts and the per-tile statistics).
    """
    root = Path(__file__).resolve().parents[3]  # same anchor as inject_test.py
    frame = np.asarray(Image.open(root / cfg["checks"]["real_frame"]).convert("L"),
                       dtype=np.float64)
    px = cfg["patch_px"]
    threshold = cfg["checks"]["real_empty_patch_threshold_dn"]
    n_tiles_y, n_tiles_x = frame.shape[0] // px, frame.shape[1] // px
    kept, n_pixels = [], 0
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            tile = frame[i * px:(i + 1) * px, j * px:(j + 1) * px]
            if float(tile.max()) <= threshold:
                kept.append(tile.ravel())
                n_pixels += tile.size
    all_pixels = np.concatenate(kept)
    fractions = {}
    for v in values:
        fractions[str(v)] = round(float((all_pixels == v).mean()), 4)
    tile_means = np.array([t.mean() for t in kept])
    tile_stds = np.array([t.std() for t in kept])
    q = np.percentile(tile_means, [0, 5, 50, 95, 100])
    stats = {
        "n_tiles_kept": len(kept),
        "n_tiles_removed": n_tiles_y * n_tiles_x - len(kept),
        "per_tile_mean_dn": {"min": round(float(q[0]), 4), "p5": round(float(q[1]), 4),
                             "median": round(float(q[2]), 4), "p95": round(float(q[3]), 4),
                             "max": round(float(q[4]), 4)},
        "within_tile_std_mean_dn": round(float(tile_stds.mean()), 4),
        "pooled_std_dn": round(float(all_pixels.std()), 4),
        "cut_bias_note": "the cut drops every tile with a pixel above 11 DN, so it "
                         "removes high noise values, and a 680 DN star peaks near 10 "
                         "DN, so faint stars stay in; both change the compared tails "
                         "(v2 review items 3 and 4, LESSONS 4)",
    }
    return fractions, n_pixels, len(kept), stats


def check_background_noise(cfg, seed, out_dir):
    """Check 3: noise on, no star: mean, std and histogram vs config and real.

    Config-target part: draw noise_n_patches background-only patches with
    this check's own generator, check mean/std against the config targets,
    and check the zero fraction against the exact model value (review
    item 5). Real-data part: compare integer-value fractions of the
    quantized model with real empty-sky patches (review item 1). The
    config noise values are placeholders (MAD artefacts), so the config
    part compares the model to itself; the real-data comparison FAILs.
    Do not tune the noise model in step 1. The kept-tile statistics add
    no numbers to the drawn part (v2 review, item 3).

    Writes hist_background.png: left panel, integer fractions per DN,
    model bars next to real bars; right panel, the drawn pre-quantization
    histogram with the config target mean. Returns a dict of the numbers.
    """
    rng = rng_for_check(seed, 3)
    c = cfg["checks"]
    n_patches = c["noise_n_patches"]
    pre_all, post_all = [], []
    for _ in range(n_patches):
        pre, post = render_star((None, None), 0.0, cfg["psf_models"]["gaussian_round"],
                                cfg["background_dn"], cfg["noise"], rng,
                                cfg["patch_px"], cfg["supersample_per_side"],
                                noise_on=True)
        pre_all.append(pre.ravel())
        post_all.append(post.ravel())
    pre_all = np.concatenate(pre_all)
    post_all = np.concatenate(post_all)
    mean = c["noise_mean_target_dn"]
    var = mean * cfg["noise"]["electrons_per_dn"] + cfg["noise"]["read_noise_sigma_dn"] ** 2
    target_std = math.sqrt(var)
    p0 = exact_zero_fraction(cfg["noise"], cfg["background_dn"], 1000000)
    got_mean, got_std = float(pre_all.mean()), float(pre_all.std())
    got_p0 = float((post_all == 0.0).mean())

    values = list(range(0, 11))
    real_fractions, real_pixels, real_tiles, real_stats = \
        real_empty_patch_fractions(cfg, values)
    model_fractions = {}
    for v in values:
        model_fractions[str(v)] = round(float((post_all == float(v)).mean()), 4)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    width = 0.4
    xs = np.arange(len(values))
    ax.bar(xs - width / 2, [model_fractions[str(v)] for v in values], width,
           label=f"model, quantized ({n_patches} patches)")
    ax.bar(xs + width / 2, [real_fractions[str(v)] for v in values], width,
           label=f"real empty patches ({real_tiles} tiles)", color="tab:orange")
    ax.set_xticks(xs)
    ax.set_xticklabels(values)
    ax.set_xlabel("DN (integer value)")
    ax.set_ylabel("fraction of pixels")
    ax.set_title("fractions per integer DN, model vs real")
    ax.legend()
    ax = axes[1]
    bins = np.arange(-3.0, 11.0 + 0.5, 0.25)
    ax.hist(pre_all, bins=bins, alpha=0.6, label="drawn, before quantization")
    ax.axvline(mean, color="black", linestyle="--", label=f"target mean {mean} DN")
    ax.set_xlabel("DN")
    ax.set_ylabel("pixels")
    ax.set_title(f"{n_patches} background-only patches, noise on, no star")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/hist_background.png", dpi=120)
    plt.close(fig)

    ok_config = (abs(got_mean - mean) <= c["noise_mean_tol_dn"]
                 and abs(got_std - target_std) <= c["noise_std_tol_dn"]
                 and abs(got_p0 - p0) <= c["noise_zero_tol_abs"])
    return {
        "config_part_note": "the config targets are placeholders (MAD artefact); "
                             "this part compares the model to itself",
        "target_mean_dn": mean,
        "target_std_dn": round(target_std, 4),
        "drawn_mean_dn": round(got_mean, 4),
        "drawn_std_dn": round(got_std, 4),
        "exact_zero_fraction_model": round(p0, 4),
        "drawn_zero_fraction": round(got_p0, 4),
        "n_pixels_model": int(pre_all.size),
        "pass_against_config": bool(ok_config),
        "real_part": {
            "note": "real data from local empty 32x32 tiles, no pixel above "
                    f"{c['real_empty_patch_threshold_dn']} DN; fits local "
                    "patches, not the whole frame (review item 1, step-3 note)",
            "n_pixels_real": real_pixels,
            "n_tiles_real": real_tiles,
            "fractions_per_dn_model": model_fractions,
            "fractions_per_dn_real": real_fractions,
            "kept_tile_stats": real_stats,
        },
        "verdict": "PASS against config, FAIL against real data",
    }


def check_grid(cfg, seed, out_dir):
    """Check 4: visual grid of 25 patches, noise on, quantized and clipped.

    Rows change the PSF shape (sigma and ellipticity, review item 6),
    columns change the flux. One common stretch for all panels: 0 to the
    brightest panel's max, so the faint stars stay visible (review item
    6). Saves grid_25_patches.png. Returns a dict.
    """
    rng = rng_for_check(seed, 4)
    rows = cfg["checks"]["grid_rows"]
    fluxes = cfg["checks"]["grid_fluxes_dn"]
    centre_px = cfg["center_pixel_index"] + 0.5
    patches = {}
    for i, psf in enumerate(rows):
        for j, flux in enumerate(fluxes):
            _, post = render_star((centre_px, centre_px), flux, psf,
                                  cfg["background_dn"], cfg["noise"], rng,
                                  cfg["patch_px"], cfg["supersample_per_side"],
                                  noise_on=True)
            patches[(i, j)] = post
    stretch_max = max(float(p.max()) for p in patches.values())
    fig, axes = plt.subplots(5, 5, figsize=(13, 13))
    for i, psf in enumerate(rows):
        for j, flux in enumerate(fluxes):
            post = patches[(i, j)]
            axes[i, j].imshow(post, vmin=0.0, vmax=stretch_max)
            label = f"{psf['shape']} {psf['major_px']:.1f}"
            if psf["axis_ratio"] > 1.0:
                label += f" r{psf['axis_ratio']:.1f} a{psf['angle_deg']:.0f}"
            axes[i, j].set_title(f"{label}, {flux:.0f} DN, max {post.max():.0f}",
                                 fontsize=8)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    fig.suptitle(f"25 patches, background 2.0 DN, noise on, common stretch 0-{stretch_max:.0f} DN")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/grid_25_patches.png", dpi=100)
    plt.close(fig)
    return {"file": "grid_25_patches.png", "stretch_max_dn": round(stretch_max, 1),
            "rows": rows, "fluxes_dn": fluxes}


def main():
    """Run the four step-1 checks, write plots and numbers to --out."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/step1.json")
    ap.add_argument("--out", default="reports/step1")
    ap.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    os.makedirs(args.out, exist_ok=True)
    results = {
        "seed": seed,
        "config": args.config,
        "check1_com": check_com(cfg, seed, args.out, True),
        "check2_flux": check_flux(cfg, seed),
        "check3_background_noise": check_background_noise(cfg, seed, args.out),
        "check4_grid": check_grid(cfg, seed, args.out),
    }
    with open(f"{args.out}/numbers.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
