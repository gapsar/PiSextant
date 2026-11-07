"""Step 3, part 5: flux distribution, vignetting and crowding (guide 5,
step 3; parameters 4.1.7, 4.1.11, 4.1.12 vignetting).

What it does, per development night (13, 15, 16 Oct):
1. Picks N_FRAMES frames spread over the night with a trail bound below
   TRAIL_PX (measure_psf.still_frames with a looser limit).
2. Linear light: the ISP tone curve (gamma + ce stretch) acts pixel by
   pixel, so each pixel is mapped back with its inverse (measure_psf.tone).
   Units: Y DN of the first gamma segment ("lin DN").
3. Detection, the SAME on every night (the logged centroids depend on the
   live detector settings, which changed between nights): Gaussian smoothing
   (sigma SMOOTH_PX), local maximum in a 7 x 7 box, at least DET_SIGMA
   noise units above the smoothed background; the noise is the std of a
   SYNTHETIC background of the night (part 1 model) smoothed the same way
   (a MAD of the real frame fails: on 13 Oct 58% of the pixels are 0, the
   MAD was too small and the first run detected noise, 852 per frame),
   scaled with radius by the part-1 ring variance c0 + c1 mu (the lens
   shading gain makes the noise grow toward the edges; with one noise for
   the whole frame the edges held hundreds of noise detections per cell);
   the background is local: the median of TILE x TILE px tiles; not on a fixed (hot)
   pixel; not within EDGE_PX of the border.
4. Photometry: sum of lin DN within AP_R px minus the annulus median
   (ANN_R1..ANN_R2 px) times the aperture area.
5. Outputs:
   - flux distribution: N(> F) per Mpx per night, and the flux quantiles;
   - crowding: other detections in a 32 x 32 patch around a detection, and
     the nearest-neighbour distance;
   - vignetting without a catalogue: the ship and the sky move the stars
     across the field, so every column of a horizontal band through the
     centre sees the same star population at about the same elevation. The
     flux at a fixed rank per unit area (bright end, complete everywhere),
     relative to the centre, is the light loss. Left and right are kept
     apart: a difference between them would point to an elevation effect,
     not to the lens. Compared with the prediction from the tuning file
     (luminance table, applied at strength 0.5).

RESULT 27 Oct (WORKLOG): the vignetting part is TOO NOISY on these data (the
bright end holds a few stars per bin, repeated over consecutive frames;
left and right centre bins differ by 40%). Kept to show the attempt; do not
use its ratios. Vignetting needs a catalogue match (known magnitudes).

Outputs in reports/step3/: flux.json, flux.png.

Run from the centroid_cnn folder:
    python3 src/measure_flux.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
from PIL import Image
from scipy.ndimage import gaussian_filter, maximum_filter

import measure_psf as MP
from measure_hot_pixels import read_night
from measure_background import render_background

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
N_FRAMES = 40
TRAIL_PX = 3.0
SMOOTH_PX = 1.5
DET_SIGMA = 5.0
EDGE_PX = 14
AP_R, ANN_R1, ANN_R2 = 6, 9, 13
PATCH = 32
BAND_ROWS = 250                         # band: |row - centre| < this
COL_EDGES = [0, 200, 400, 600, 800, 1000, 1152]   # |col - centre| bins
DENSITIES = [2, 5]                      # per Mpx: ranks used for vignetting (bright end)
H, W = 1296, 2304
TILE = 64
# part 1 ring fits (WORKLOG 25 Oct): var = c0 + c1 * mu per radius ring
RING_EDGES = [0, 200, 400, 600, 800, 1000, 1400]
RING_C0 = [1.23, 1.23, 1.35, 1.38, 1.41, 1.45]
RING_C1 = [0.72, 0.80, 0.94, 1.19, 1.48, 1.77]
_RR, _CC = np.mgrid[0:H, 0:W]
_RIDX = np.minimum(np.searchsorted(RING_EDGES, np.hypot(_RR - 648.0, _CC - 1152.0), side="right") - 1, 5)


def inverse_tone(y):
    """Y DN -> lin DN (inverse of measure_psf.tone, exact on the curve)."""
    x = np.interp(y, MP.G_OUT, MP.G_IN)
    return np.where(x > MP.CE_KNEE, MP.CE_KNEE + (x - MP.CE_KNEE) / MP.CE_GAIN, x)


def predicted_light(r):
    """Fraction of the centre flux left in Y at radius r, if the luminance
    table is the full inverse vignetting and the ISP applies it at 0.5."""
    algs = {list(a)[0]: list(a.values())[0] for a in MP.TUNING["algorithms"]}
    lut = np.array(algs["rpi.alsc"]["luminance_lut"]).reshape(12, 16)
    s = algs["rpi.alsc"]["luminance_strength"]
    yy, xx = np.mgrid[0:12, 0:16] + 0.5
    rc = np.hypot((yy / 12 - 0.5) * H, (xx / 16 - 0.5) * W)
    lr = np.interp(r, *zip(*sorted(zip(rc.ravel(), lut.ravel()))))
    l0 = lut[5:7, 7:9].mean()
    return ((1 + s * (lr - 1)) / lr) / ((1 + s * (l0 - 1)) / l0)


def smoothed_noise(night, rng):
    """Std of the night's synthetic background after the same smoothing."""
    m = MP.night_model(night)
    field = render_background(m["mu"], m["sigma"], m["s"], m["a"], m["b"], (512, 512), rng)
    return float(gaussian_filter(inverse_tone(field), SMOOTH_PX).std())


def noise_map(noise, mu):
    """Smoothed-noise map: the whole-frame value scaled by sqrt(ring var / mean var)."""
    var = np.array(RING_C0) + np.array(RING_C1) * mu
    mean_var = np.bincount(_RIDX.ravel(), minlength=6) @ var / _RIDX.size
    return noise * np.sqrt(var / mean_var)[_RIDX]


def detect(img, fixed, nmap):
    """Detections of one frame. Output: (n, 3) array row, col, flux (lin DN)."""
    lin = inverse_tone(img.astype(float))
    sm = gaussian_filter(lin, SMOOTH_PX)
    tiles = np.median(sm[:H // TILE * TILE, :W // TILE * TILE].reshape(H // TILE, TILE, W // TILE, TILE), axis=(1, 3))
    bkg = np.repeat(np.repeat(tiles, TILE, 0), TILE, 1)
    bkg = np.pad(bkg, ((0, H - bkg.shape[0]), (0, W - bkg.shape[1])), mode="edge")
    peaks = (sm == maximum_filter(sm, 7)) & (sm > bkg + DET_SIGMA * nmap) & ~fixed
    peaks[:EDGE_PX], peaks[-EDGE_PX:], peaks[:, :EDGE_PX], peaks[:, -EDGE_PX:] = False, False, False, False
    rr, cc = np.nonzero(peaks)
    dy, dx = np.mgrid[-ANN_R2:ANN_R2 + 1, -ANN_R2:ANN_R2 + 1]
    d = np.hypot(dy, dx)
    ap, ann = d <= AP_R, (d >= ANN_R1) & (d <= ANN_R2)
    out = []
    for r, c in zip(rr, cc):
        cut = lin[r - ANN_R2:r + ANN_R2 + 1, c - ANN_R2:c + ANN_R2 + 1]
        out.append((r, c, cut[ap].sum() - ap.sum() * np.median(cut[ann])))
    return np.array(out).reshape(-1, 3)


def main():
    MP.STILL_PX = TRAIL_PX
    det, n_frames = [], {}
    rng0 = np.random.default_rng(MP.SEED)
    for k, night in enumerate(MP.NIGHTS):
        run = MP.RUNS_DIR / night
        noise = smoothed_noise(night, rng0)
        nmap = noise_map(noise, MP.night_model(night)["mu"])
        frames = MP.still_frames(run)
        pick = [frames[i] for i in np.linspace(0, len(frames) - 1, min(N_FRAMES, len(frames))).astype(int)]
        med = np.median(read_night(run)[0], axis=0)
        fixed = maximum_filter(med - med.mean() >= MP.HOT_MED_DN, 3)
        for f, (name, t) in enumerate(pick):
            d = detect(np.asarray(Image.open(run / "images" / name)), fixed, nmap)
            det.append(np.column_stack([np.full(len(d), k), np.full(len(d), f), d]))
        n_frames[night] = len(pick)
        print(f"{night}: smoothed noise {noise:.3f} lin DN; frames with trail < {TRAIL_PX} px {len(frames)}, used {len(pick)}, "
              f"detections per frame {sum(len(x) for x in det if x[0, 0] == k) / len(pick):.0f}", flush=True)
    det = np.vstack(det)          # night, frame, row, col, flux
    np.save(HERE / "scratch" / "flux_detections.npy", det)
    mpx = H * W / 1e6
    out = {"settings": {"n_frames": N_FRAMES, "trail_px": TRAIL_PX, "smooth_px": SMOOTH_PX,
                        "det_sigma": DET_SIGMA, "aperture_px": AP_R, "annulus_px": [ANN_R1, ANN_R2]},
           "nights": {}}
    f_grid = np.array([50, 100, 200, 300, 500, 1000, 2000, 5000])

    # flux distribution and crowding per night
    for k, night in enumerate(MP.NIGHTS):
        d = det[det[:, 0] == k]
        nf = n_frames[night]
        n_gt = [(d[:, 4] > f).sum() / nf / mpx for f in f_grid]
        others, nn = [], []
        for f in range(nf):
            x = d[d[:, 1] == f][:, 2:4]
            if len(x) < 2:
                continue
            dr = np.abs(x[:, None, 0] - x[None, :, 0])
            dc = np.abs(x[:, None, 1] - x[None, :, 1])
            others.append(((dr < PATCH / 2) & (dc < PATCH / 2)).sum(axis=1) - 1)
            dist = np.hypot(dr, dc)
            np.fill_diagonal(dist, np.inf)
            nn.append(dist.min(axis=1))
        others, nn = np.concatenate(others), np.concatenate(nn)
        out["nights"][night] = {
            "frames": nf, "detections_per_frame": round(len(d) / nf, 1),
            "flux_lin_quantiles_p10_p50_p90_p99": np.round(np.percentile(d[:, 4], [10, 50, 90, 99]), 1).tolist(),
            "n_greater_than_flux_per_mpx": dict(zip(map(str, f_grid.tolist()), np.round(n_gt, 2).tolist())),
            "others_in_32px_patch_mean": round(float(others.mean()), 3),
            "others_in_32px_patch_fraction_0_1_2plus": np.round([np.mean(others == 0), np.mean(others == 1),
                                                                 np.mean(others >= 2)], 3).tolist(),
            "nearest_neighbour_px_p10_p50": np.round(np.percentile(nn, [10, 50]), 1).tolist(),
            "nearest_neighbour_lt_8px_fraction": round(float(np.mean(nn < 8)), 4),
        }
        o = out["nights"][night]
        print(f"  {night[:10]}: det/frame {o['detections_per_frame']}, flux p10/50/90/99 {o['flux_lin_quantiles_p10_p50_p90_p99']}, "
              f"N(>F)/Mpx {o['n_greater_than_flux_per_mpx']}, others/patch {o['others_in_32px_patch_mean']} "
              f"(0/1/2+ {o['others_in_32px_patch_fraction_0_1_2plus']}), nn p10/p50 {o['nearest_neighbour_px_p10_p50']}, "
              f"nn < 8 px {o['nearest_neighbour_lt_8px_fraction']}", flush=True)

    # vignetting: flux at a fixed surface-density rank, per |col| bin, left and right
    band = np.abs(det[:, 2] - MP.C_ROW) < BAND_ROWS
    total_frames = sum(n_frames.values())
    vig = []
    for side, sgn in [("left", -1), ("right", 1)]:
        for lo, hi in zip(COL_EDGES[:-1], COL_EDGES[1:]):
            dc = sgn * (det[:, 3] - MP.C_COL)
            m = band & (dc >= lo) & (dc < hi)
            area = 2 * BAND_ROWS * (hi - lo) * total_frames / 1e6
            fl = np.sort(det[m, 4])[::-1]
            row = {"side": side, "col_from_centre": [lo, hi], "n": int(m.sum())}
            for dens in DENSITIES:
                k = int(dens * area)
                row[f"flux_at_{dens}_per_mpx"] = float(fl[k]) if k < len(fl) else None
                # bootstrap SE over frames (resample frames)
            vig.append(row)
    for dens in DENSITIES:
        key = f"flux_at_{dens}_per_mpx"
        ref = np.mean([v[key] for v in vig if v["col_from_centre"][0] == 0])
        for v in vig:
            v[key.replace("flux", "ratio")] = round(v[key] / ref, 3) if v[key] else None
    rng = np.random.default_rng(MP.SEED)
    # SE of the ratio by bootstrap over (night, frame)
    keys = np.unique(det[:, 0] * 1000 + det[:, 1])
    boots = {(v["side"], v["col_from_centre"][0]): [] for v in vig}
    for _ in range(200):
        pick = rng.choice(keys, len(keys))
        cnt = {kk: c for kk, c in zip(*np.unique(pick, return_counts=True))}
        w = np.array([cnt.get(kk, 0) for kk in det[:, 0] * 1000 + det[:, 1]])
        refs = []
        vals = {}
        for v in vig:
            lo, hi = v["col_from_centre"]
            sgn = -1 if v["side"] == "left" else 1
            dc = sgn * (det[:, 3] - MP.C_COL)
            m = band & (dc >= lo) & (dc < hi) & (w > 0)
            fl = np.repeat(det[m, 4], w[m])
            fl = np.sort(fl)[::-1]
            k = int(DENSITIES[0] * 2 * BAND_ROWS * (hi - lo) * len(pick) / 1e6 * total_frames / len(keys))
            vals[(v["side"], lo)] = fl[k] if k < len(fl) else np.nan
        ref = np.nanmean([vals[(s, 0)] for s in ("left", "right")])
        for kk in vals:
            boots[kk].append(vals[kk] / ref)
    print(f"vignetting (band |row - centre| < {BAND_ROWS}): flux ratio to the centre at {DENSITIES} per Mpx, SE (first density), prediction")
    for v in vig:
        lo, hi = v["col_from_centre"]
        v["ratio_se"] = round(float(np.nanstd(boots[(v["side"], lo)])), 3)
        v["predicted"] = round(float(np.mean([predicted_light(r) for r in np.linspace(lo, hi, 5)])), 3)
        print(f"  {v['side']:5s} |col| {lo:4d}-{hi:4d}: n {v['n']:5d} ratio "
              f"{[v.get(f'ratio_at_{d}_per_mpx') for d in DENSITIES]} SE {v['ratio_se']} predicted {v['predicted']}", flush=True)
    out["vignetting"] = vig
    json.dump(out, open(HERE / "reports" / "step3" / "flux.json", "w"), indent=1)
    plot(det, n_frames, vig, f_grid, HERE / "reports" / "step3" / "flux.png")


def plot(det, n_frames, vig, f_grid, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    mpx = H * W / 1e6
    fg = np.logspace(1.5, 4, 40)
    for k, night in enumerate(MP.NIGHTS):
        d = det[det[:, 0] == k][:, 4]
        ax[0].loglog(fg, [(d > f).sum() / n_frames[night] / mpx for f in fg], label=night[:10])
    ax[0].set_xlabel("flux F (lin DN, aperture 6 px)")
    ax[0].set_ylabel("detections with flux > F per Mpx")
    ax[0].legend(fontsize=8)
    for side, mk in [("left", "o"), ("right", "s")]:
        v = [x for x in vig if x["side"] == side]
        mid = [np.mean(x["col_from_centre"]) for x in v]
        ax[1].errorbar(mid, [x[f"ratio_at_{DENSITIES[0]}_per_mpx"] for x in v], [2 * x["ratio_se"] for x in v],
                       fmt=mk, label=f"measured, {side}")
    v = [x for x in vig if x["side"] == "left"]
    ax[1].plot([np.mean(x["col_from_centre"]) for x in v], [x["predicted"] for x in v], "k--", label="tuning-file prediction")
    ax[1].set_xlabel("|column - centre| (px), central band")
    ax[1].set_ylabel("flux relative to the centre")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
