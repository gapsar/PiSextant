"""Step 3 check, done with the step-5 generator: 50 real + 50 synthetic star
patches, mixed, for the user's eye test (guide step 3: "the user should not
be able to tell them apart easily").

Written by Claude, 29 Oct 2026. Run from the centroid_cnn folder:
    python src/eye_test_stars.py
Output: reports/step5/eye_test_stars.png (open THIS one only) and
reports/step5/eye_test_key.json (the answers: open after the test).

Real: random saved frames of 13/15/16 Oct inside the TRAINING time range
(configs/step5.json splits), own detection (smoothed image, blobs of >= 12 px,
not hot pixels), random detections, 32 x 32 patch with the centre pixel =
floor(light centre) (CONVENTIONS). Synthetic: generator with ALL groups =
this instrument, each with the flux of one real detection (aperture 6 px,
annulus 9-13 px, as in measure_flux.py), so brightness is not a cue.
Display: 8-bit Y, gray, the same stretch rule for every patch.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import generator as G  # noqa: E402

RUNS = HERE.parents[1] / "runs"
OUT = HERE / "reports" / "step5"
NIGHTS = ["2026-10-13_16-14-01", "2026-10-15_16-00-55", "2026-10-16_16-05-50"]


def real_patches(rng, n):
    lo, hi = G.CFG5["splits"]["train"]
    out = []
    yy, xx = np.mgrid[-16:16, -16:16]
    rad = np.hypot(yy + 0.5, xx + 0.5)
    while len(out) < n:
        night = NIGHTS[rng.integers(3)]
        cap = [json.loads(l) for l in open(RUNS / night / "capture.jsonl")]
        cap = [c for c in cap if c.get("image_saved")]
        t0, t1 = cap[0]["t_exposure_mid"], cap[-1]["t_exposure_mid"]
        cap = [c for c in cap if t0 + lo * (t1 - t0) <= c["t_exposure_mid"] <= t0 + hi * (t1 - t0)]
        c = cap[rng.integers(len(cap))]
        img = np.asarray(Image.open(RUNS / night / c["image"]), float)
        bkg = np.median(img)
        lab, k = ndimage.label(ndimage.gaussian_filter(img - bkg, 1.5) > 1.2)
        if k == 0:
            continue
        idx = np.arange(1, k + 1)
        size = ndimage.sum(np.ones_like(img), lab, idx)
        com = ndimage.center_of_mass(np.clip(img - bkg, 0, None), lab, idx)
        ok = [j for j in range(k) if size[j] >= 12 and 20 < com[j][0] < img.shape[0] - 20
              and 20 < com[j][1] < img.shape[1] - 20]
        for j in rng.permutation(ok)[:2]:
            r0, c0 = int(np.floor(com[j][0] + 0.5)), int(np.floor(com[j][1] + 0.5))   # corner coords
            p = img[r0 - 16:r0 + 16, c0 - 16:c0 + 16]
            ann = p[(rad >= 9) & (rad <= 13)]
            flux = float(np.sum(p[rad <= 6] - np.median(ann)))
            if flux > 20:
                out.append({"patch": p, "flux": flux, "night": night, "frame": c["image"]})
    return out[:n]


def main():
    rng = np.random.default_rng(20261029)
    OUT.mkdir(parents=True, exist_ok=True)
    real = real_patches(rng, 50)
    G.WINDOWS["check"] = G.load_split_windows("check", n_per_run=100)
    synth = []
    for i, r in enumerate(real):
        X, y, P, ex = G.make_sample("check", 5000 + i, inst_all=True, flux_lin=r["flux"])
        synth.append({"patch": ex["Y"], "flux": r["flux"]})
    items = [("real", r) for r in real] + [("synthetic", s) for s in synth]
    order = rng.permutation(len(items))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(10, 10, figsize=(13, 13))
    key = {}
    for k, (a, i) in enumerate(zip(ax.ravel(), order)):
        kind, it = items[i]
        p = it["patch"]
        a.imshow(p, cmap="gray", vmin=0, vmax=max(8, np.percentile(p, 99.5)), interpolation="nearest")
        a.set_title(str(k + 1), fontsize=7)
        a.axis("off")
        key[k + 1] = {"kind": kind, "flux_lin": round(it["flux"], 1),
                      **({"night": it["night"], "frame": it["frame"]} if kind == "real" else {})}
    fig.suptitle("Star eye test: which patches are REAL? (50 real, 50 synthetic)")
    fig.tight_layout()
    fig.savefig(OUT / "eye_test_stars.png", dpi=70)
    json.dump(key, open(OUT / "eye_test_key.json", "w"), indent=1)
    print("written", OUT / "eye_test_stars.png")


if __name__ == "__main__":
    main()
