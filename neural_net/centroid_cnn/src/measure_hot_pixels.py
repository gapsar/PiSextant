"""Step 3, part 2: hot pixels (guide section 5, step 3; parameter 4.1.12).

What it does, per development night (13, 15, 16 Oct):
1. Reads N_FRAMES frames spread over the night and takes the per-pixel
   median. The ship moves and the sky turns, so a star does not stay on one
   pixel: the median removes the stars and keeps the fixed pattern.
2. A pixel is HOT when it is at least HOT_DN above the night's background
   in at least HOT_FRAC of the frames.
3. Tests the two findings of the prototypes (scratch/hot_pixels_proto*):
   - the count grows with the sensor temperature (dark current);
   - a hot pixel is a small blob, not one pixel: the ISP (demosaic,
     sharpening) spreads it. The script stacks the local excess around the
     isolated hot pixels, normalized to the centre: this is the ISP's
     impulse response.

Outputs in reports/step3/: hot_pixels.json, hot_pixels.png.

Run from the centroid_cnn folder:
    python3 src/measure_hot_pixels.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
from PIL import Image
from scipy.ndimage import label, median_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = Path(__file__).resolve().parents[3] / "runs"
NIGHTS = ["2026-10-13_16-14-01", "2026-10-15_16-00-55", "2026-10-16_16-05-50"]
N_FRAMES = 60      # frames per night, spread evenly over the run
HOT_DN = 8         # "lit": at least this many DN above the night background
HOT_FRAC = 0.5     # hot: lit in at least this fraction of the frames
LOCAL_PX = 7       # local background of the median image: median filter size
HALF = 3           # template half size: (2 * HALF + 1) px square
OUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "step3"


def read_night(run):
    """Frames and sensor temperatures of one night.
    Input: run folder. Output: (N, H, W) int16 stack, (N,) temperatures in C."""
    temps = {}
    for line in open(run / "capture.jsonl"):
        rec = json.loads(line)
        temps[Path(rec["image"]).name] = rec["camera_metadata"].get("SensorTemperature")
    files = sorted((run / "images").glob("frame_*.png"))
    pick = [files[i] for i in np.linspace(0, len(files) - 1, N_FRAMES).astype(int)]
    stack = np.stack([np.asarray(Image.open(f)) for f in pick]).astype(np.int16)
    t = np.array([temps.get(f.name) or np.nan for f in pick], float)
    return stack, t


def measure_night(name):
    """Hot pixel map, counts and template of one night. Output: dict."""
    stack, temp = read_night(RUNS_DIR / name)
    med = np.median(stack, axis=0)
    bkg = float(med.mean())
    lit = stack - bkg >= HOT_DN
    hot = lit.mean(axis=0) >= HOT_FRAC
    rows, cols = np.nonzero(hot)

    # temperature: lit hot pixels per frame, coldest vs warmest third
    n_lit = lit[:, rows, cols].sum(axis=1)
    order = np.argsort(temp)
    third = N_FRAMES // 3
    cold, warm = order[:third], order[-third:]

    # shape: clusters of hot pixels (8-connected), and the template of the
    # isolated ones (a cluster of one hot pixel, away from the edge)
    lab, n_clusters = label(hot, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    excess = med - median_filter(med, LOCAL_PX)
    h, w = med.shape
    cuts, peaks = [], []
    for r, c in zip(rows, cols):
        if sizes[lab[r, c] - 1] != 1 or not (HALF <= r < h - HALF and HALF <= c < w - HALF):
            continue
        cut = excess[r - HALF:r + HALF + 1, c - HALF:c + HALF + 1]
        peaks.append(float(cut[HALF, HALF]))
        cuts.append(cut / cut[HALF, HALF])
    template = np.mean(cuts, axis=0) if cuts else np.full((2 * HALF + 1,) * 2, np.nan)

    return {
        "night": name,
        "frames": N_FRAMES,
        "background_dn": round(bkg, 3),
        "temperature_c": {"min": float(np.nanmin(temp)), "median": float(np.nanmedian(temp)),
                          "max": float(np.nanmax(temp))},
        "hot_px": int(hot.sum()),
        "hot_per_mpx": round(hot.sum() / (h * w) * 1e6, 2),
        "clusters": int(n_clusters),
        "cluster_size_counts": np.bincount(sizes).tolist(),
        "lit_hot_per_frame_cold_third": {"t_c": round(float(np.nanmean(temp[cold])), 1),
                                         "n": round(float(n_lit[cold].mean()), 1)},
        "lit_hot_per_frame_warm_third": {"t_c": round(float(np.nanmean(temp[warm])), 1),
                                         "n": round(float(n_lit[warm].mean()), 1)},
        "isolated_used_for_template": len(cuts),
        "peak_excess_dn": {"p10": float(np.percentile(peaks, 10)), "median": float(np.median(peaks)),
                           "p90": float(np.percentile(peaks, 90))} if peaks else None,
        "template": np.round(template, 3).tolist(),
        "hot_list_row_col": [[int(r), int(c)] for r, c in zip(rows, cols)],
    }


def plot(results, out):
    """Count vs temperature, and the template of each night."""
    fig, ax = plt.subplots(1, 1 + len(results), figsize=(4 * (1 + len(results)), 3.6))
    for res in results:
        t = [res["lit_hot_per_frame_cold_third"]["t_c"], res["lit_hot_per_frame_warm_third"]["t_c"]]
        n = [res["lit_hot_per_frame_cold_third"]["n"], res["lit_hot_per_frame_warm_third"]["n"]]
        ax[0].semilogy(t, n, "o-", label=res["night"][:10])
    ax[0].set_xlabel("sensor temperature (C)")
    ax[0].set_ylabel("lit hot pixels per frame")
    ax[0].set_title("hot pixels vs temperature")
    ax[0].legend(fontsize=8)
    for a, res in zip(ax[1:], results):
        tpl = np.array(res["template"])
        im = a.imshow(tpl, cmap="RdBu_r", vmin=-0.4, vmax=0.4)
        for (i, j), v in np.ndenumerate(tpl):
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        a.set_title(f"{res['night'][:10]}: template, n={res['isolated_used_for_template']}", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    fig.colorbar(im, ax=ax[1:], shrink=0.8, label="excess / centre (centre = 1)")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [measure_night(n) for n in NIGHTS]
    hot_sets = [set(map(tuple, r["hot_list_row_col"])) for r in results]
    common = sorted(set.intersection(*hot_sets))
    summary = {
        "settings": {"n_frames": N_FRAMES, "hot_dn": HOT_DN, "hot_frac": HOT_FRAC, "local_px": LOCAL_PX},
        "nights": results,
        "hot_on_all_nights": [list(map(int, p)) for p in common],
        "template_mean_of_nights": np.round(np.mean([r["template"] for r in results], axis=0), 3).tolist(),
    }
    json.dump(summary, open(OUT_DIR / "hot_pixels.json", "w"), indent=1)
    plot(results, OUT_DIR / "hot_pixels.png")
    for r in results:
        print(f"{r['night'][:10]}: T median {r['temperature_c']['median']:.0f} C, hot {r['hot_px']} "
              f"({r['hot_per_mpx']}/Mpx) in {r['clusters']} clusters; lit per frame "
              f"{r['lit_hot_per_frame_cold_third']['n']} at {r['lit_hot_per_frame_cold_third']['t_c']} C, "
              f"{r['lit_hot_per_frame_warm_third']['n']} at {r['lit_hot_per_frame_warm_third']['t_c']} C; "
              f"peak excess {r['peak_excess_dn']}")
    print("hot on all 3 nights:", len(common))
    tpl = np.array(summary["template_mean_of_nights"])
    print("template (mean of nights), centre 3x3 + ring:")
    print(np.round(tpl, 2))


if __name__ == "__main__":
    main()
