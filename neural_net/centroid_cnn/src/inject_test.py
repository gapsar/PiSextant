#!/usr/bin/env python3
"""Step 0 injection test: find which pixel-centre convention tetra3 uses.

What it does: injects N_INJECT bright noise-free round Gaussian stars at
known positions in empty sky areas of one real frame, runs tetra3's
get_centroids_from_image, and reports (found - injected) per axis.

Coordinate definition used by this test (corner coordinates): pixel (i, j)
covers the square [i, i+1) x [j, j+1), so its centre is at (i + 0.5,
j + 0.5). A symmetric noise-free star's light centroid equals the injected
position exactly. If the mean bias is 0 on both axes, tetra3 reports in
this convention. If the mean bias is about -0.5 on both axes, tetra3
reports in the index convention (the centre of pixel (i, j) at (i, j)).

tetra3 comes from the read-only cedar-solve source tree. The package
installed in site-packages is the gRPC client only, it has no centroid
code. No cedar-detect test here: no server binary on this machine.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/gapsar/Documents/projects/neosextant/android/libraries/cedar-solve")
import tetra3  # noqa: E402  the source-tree package, not the installed one

SEED = 20261021        # fixed, per section 0 of the guide
N_INJECT = 20          # positions with random sub-pixel offsets, per step 0
SIGMA_PX = 3.3         # px, mid of the measured PSF range; the test needs a
                       # symmetric star, not the exact width of this camera
PEAK_DN = 110.0        # DN, nominal peak of the injected star: bright so noise
                       # and truncation jitter stay small. The spot covers ~200 px
                       # above the threshold, so the tetra3 call must raise the
                       # default max_area=100 filter or it would reject our stars.
SUP = 10               # supersamples per pixel axis (guide 4.1.2 asks >= 5)
CLEAN_R_PX = 25        # px, area checked for contamination around each injection
MIN_SEP_PX = 120       # px, separation between injections; tetra3 subtracts a
                       # local mean over a 25 px filter, so no two stars may
                       # share the filter window of each other
MARGIN_PX = 60.0       # px, keep injections away from the frame edges
MATCH_R_PX = 2.5       # px, max injection-to-found distance for pairing
MAX_CONTAMINATION = 6  # sigma above background: no real star, hot pixel or artefact
HALF_WIN = 24          # px, half the render window (48 x 48 holds the +/-15 px core)

def load_frame(path):
    """Read one 8-bit frame as float (row, col). Path in, 2-D array out."""
    return np.asarray(Image.open(path), dtype=np.float64)

def robust_background(img):
    """Background median and noise sigma (MAD-based) of the whole frame.
    Array in, (median, sigma) in DN out."""
    med = np.median(img)
    mad = np.median(np.abs(img - med))
    return med, 1.4826 * mad

def pick_positions(img, n, rng, bkg, sigma):
    """Pick n empty-sky positions with random sub-pixel offsets, by rejection
    sampling on a fixed rng so the test repeats exactly. Array in, list of
    (row, col) out in corner coords."""
    h, w = img.shape
    positions = []
    attempts = 0
    while len(positions) < n and attempts < 50000:
        attempts += 1
        row = rng.uniform(MARGIN_PX, h - MARGIN_PX)
        col = rng.uniform(MARGIN_PX, w - MARGIN_PX)
        i0, j0 = int(row), int(col)
        window = img[i0 - CLEAN_R_PX:i0 + CLEAN_R_PX, j0 - CLEAN_R_PX:j0 + CLEAN_R_PX]
        if window.max() > bkg + MAX_CONTAMINATION * sigma:
            continue
        if any(np.hypot(row - r, col - c) < MIN_SEP_PX for r, c in positions):
            continue
        positions.append((row, col))
    return positions

def pixel_profile(centre, sigma, sup):
    """Mean over each pixel area of a 1-D Gaussian centred at `centre`
    (window-local corner coords, peak 1). Centre in, 1-D array (length
    2 * HALF_WIN) out. Used to build the pixel-integrated 2-D patch."""
    edges = np.arange(2 * HALF_WIN)[:, None] + (np.arange(sup) + 0.5) / sup
    # pixel p covers [p, p + 1) in window coords; samples at pixel start + offsets
    return np.exp(-(edges - centre) ** 2 / (2 * sigma ** 2)).mean(axis=1)

def render_star(centre_row, centre_col, sigma, peak, sup):
    """Render one star in a 48 x 48 window, pixel area integrated, noise
    free. The window's top-left pixel is at frame pixel (floor(row) - 24,
    floor(col) - 24), so (centre_row, centre_col) = (24 + frac(row),
    24 + frac(col)). Corner coords in, 48 x 48 float patch out.

    The Gaussian is separable, so the patch is an outer product of two
    1-D pixel-area means. Its light centroid equals the input position to
    far below 0.01 px (checked in main)."""
    return peak * np.outer(pixel_profile(centre_row, sigma, sup),
                           pixel_profile(centre_col, sigma, sup))

def patch_centroid(patch):
    """Light centroid (row, col) of a patch, in window-local corner coords,
    peak-normalized patches from render_star not required (any scaling)."""
    rows = np.arange(patch.shape[0]) + 0.5  # pixel centres
    cols = np.arange(patch.shape[1]) + 0.5
    wsum = patch.sum()
    return ((patch * rows[:, None]).sum() / wsum,
            (patch * cols[None, :]).sum() / wsum)

def main():
    ap = argparse.ArgumentParser(description="Step 0 injection test")
    default_run = (Path(__file__).resolve().parents[3] /
                   "runs/2026-10-15_16-00-55/images/frame_06000.png")
    ap.add_argument("--frame", default=str(default_run))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "reports/step0"))
    ap.add_argument("--flat", action="store_true",
                    help="replace the frame by a constant level (its median): no noise, "
                         "so only the detector and the 8-bit rounding remain")
    args = ap.parse_args()

    frame = load_frame(args.frame)
    bkg, sigma_noise = robust_background(frame)
    if args.flat:
        frame = np.full_like(frame, np.round(bkg))
        bkg, sigma_noise = robust_background(frame)
    suffix = "_flat" if args.flat else ""
    print(f"frame {Path(args.frame).name}: {frame.shape[1]}x{frame.shape[0]} px, "
          f"background {bkg:.1f} DN, noise {sigma_noise:.2f} DN, max {frame.max():.0f} DN")

    rng = np.random.default_rng(SEED)
    positions = pick_positions(frame, N_INJECT, rng, bkg, sigma_noise)
    if len(positions) != N_INJECT:
        sys.exit(f"only found {len(positions)} clean positions, need {N_INJECT}")

    composite = frame.copy()
    check_rows = []
    for row, col in positions:
        # the render window top-left lands on an integer pixel, so the fractional
        # parts pass straight through: star centre = window-local (24 + fr, 24 + fc)
        cr, cc = 24 + (row - int(row)), 24 + (col - int(col))
        patch = render_star(cr, cc, SIGMA_PX, PEAK_DN, SUP)
        # self-check: the rendered patch's own light centroid must equal the target
        pc = patch_centroid(patch)
        check_rows.append((abs(pc[0] - cr), abs(pc[1] - cc)))
        i0, j0 = int(row) - HALF_WIN, int(col) - HALF_WIN
        composite[i0:i0 + 2 * HALF_WIN, j0:j0 + 2 * HALF_WIN] += patch

    worst = np.max(check_rows, axis=0)
    print(f"render self-check: worst |centroid - target| = {worst[0]:.4f}, {worst[1]:.4f} px")

    composite = np.round(composite)  # the real frame is 8-bit; the composite must be too
    assert composite.max() <= 255, "injection pushed a pixel past full well"
    image = Image.fromarray(composite.astype(np.uint8), mode="L")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / f"composite_frame{suffix}.png")  # for the eye check of step 0

    # sigma passed explicitly (cleaner threshold than the sigma=2 signature
    # default), max_area raised because our bright stars are ~200 px blobs
    # the code (not the docstring) returns (centroids, [sum, area, moments, ratio])
    # when return_moments=True: two top-level items, the second a list of four
    centroids, extra = tetra3.get_centroids_from_image(
        image, sigma=3, max_area=1000, return_moments=True)
    sums, areas, moments, axis_ratio = extra
    print(f"tetra3 found {len(centroids)} spots in the composite frame")

    # pair each injection with the nearest found centroid; record area to prove
    # none of the injections was rejected by the area limits
    rows_out = []
    for k, (row, col) in enumerate(positions):
        # centroids[:, 0] is y (row), centroids[:, 1] is x (col) - checked in the
        # source: region stats return (sum, y+0.5, x+0.5, ...) and get assembled
        # in that column order. This IS the thing this test measures, via the bias.
        d = np.hypot(centroids[:, 0] - row, centroids[:, 1] - col)
        kmin = int(np.argmin(d))
        if d[kmin] > MATCH_R_PX:
            rows_out.append((k, row, col, None, None, None, None, None, None))
            continue
        fy, fx = centroids[kmin]
        rows_out.append((k, row, col, fy, fx, fy - row, fx - col, areas[kmin], sums[kmin]))

    with open(out_dir / f"injection_results{suffix}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "inj_row", "inj_col", "found_row", "found_col",
                    "bias_row", "bias_col", "area_px", "sum_dn", "render_check"])
        for r, line in enumerate(rows_out):
            w.writerow(list(line[:9]) + [f"{check_rows[r][0]:.4f},{check_rows[r][1]:.4f}"])

    got = [r for r in rows_out if r[3] is not None]
    print(f"paired {len(got)} of {len(positions)} injections")
    for axis, name in ((5, "row"), (6, "col")):
        b = np.array([r[axis] for r in got])
        print(f"bias {name}: mean {b.mean():+.3f} px, rms {np.sqrt((b ** 2).mean()):.3f} px, "
              f"min {b.min():+.3f}, max {b.max():+.3f}")
    print("read the numbers: mean 0 on both axes confirms the corner convention this test")
    print("assumes; mean -0.5 on both axes means tetra3 reports pixel-centre coords.")

if __name__ == "__main__":
    main()
