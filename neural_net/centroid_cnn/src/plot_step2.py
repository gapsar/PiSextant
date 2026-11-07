"""Step 2 plots: reads reports/step2/table.json and writes the four PNGs
(v2 review item 5: moved out of src/error_vs_snr.py). Run it after the full
run, from the centroid_cnn folder. No new features: the same four plots as
the v2 run (scatter, bias, RMS, examples), normal rows only. The legends
now show the estimator and its fit_kind from the data (v2 review item 4), so
the reader can see which model each line used. The legend estimator list is
the v2 list unchanged, so the new weighted Moffat rows do not appear in the
plots.

Usage (run from the centroid_cnn folder):
    python3 src/plot_step2.py            # reads and writes reports/step2
    python3 src/plot_step2.py --table reports/step2/table.json --out reports/step2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # save PNGs only, this box has no window session
import matplotlib.pyplot as plt

FIT_SHORT = {  # fit_kind -> legend suffix (v2 review item 4)
    None: "no model",
    "matched, point-sampled": "matched",
    "PSF-shape mismatch": "shape-mismatch",
    "point-PSF on trail (mismatch)": "trail-mismatch",
}
ESTS = ["com", "gaussian_on_gaussian", "moffat_on_moffat", "gaussian_on_moffat",
        "gaussian_on_moffat_weighted"]  # the v2 list, unchanged


def load_json(path):
    """Read a JSON file. Input: path. Output: dict."""
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="reports/step2/table.json")
    ap.add_argument("--out", default="reports/step2")
    args = ap.parse_args()
    table = load_json(args.table)
    cfg = load_json(table["config_file"])  # the config path is in the table (review item 4)
    rows = table["rows"]
    pred = table["discrete_predictions"]
    note = table["noise_placeholder"]
    states = [(name, segs) for name, segs in cfg["trail_states"].items()]
    states.append((cfg["nonuniform_mirror"]["name"], cfg["nonuniform_mirror"]["segments"]))

    def rows_for(state, est):  # normal rows only, as in the v2 run
        return sorted([r for r in rows if r["state"] == state and r["estimator"] == est
                       and r["axis"] == "col" and r["test"] == ""], key=lambda r: r["snr_bin"])

    def label_for(rows_list, est):  # estimator + fit_kind (v2 item 4)
        return f"{est} [{FIT_SHORT[rows_list[0]['fit_kind']]}]"

    Path(args.out).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))  # scatter vs SNR, log-log
    for ai, (state_name, _segs) in enumerate(states):
        axp = axes[ai // 3][ai % 3]
        for est in ESTS:
            pts = rows_for(state_name, est)
            if pts:
                axp.plot([p["snr_bin"] for p in pts], [p["scatter"] for p in pts],
                         marker=".", label=label_for(pts, est))
        cr = rows_for(state_name, "com")
        if cr:
            axp.plot([p["snr_bin"] for p in cr], [p["crlb_px"] for p in cr],
                     "k--", lw=1, label="CRLB (col)")
        axp.set_title(state_name, fontsize=9)
        axp.set_xscale("log")
        axp.set_yscale("log")
        axp.legend(fontsize=7)
        axp.set_xlabel("SNR")
        axp.set_ylabel("scatter px")
    axes[1][2].axis("off")
    fig.suptitle(f"scatter vs SNR, trail axis +col (log-log); {note}")
    fig.savefig(f"{args.out}/scatter_vs_snr.png", dpi=110, bbox_inches="tight")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))  # bias vs SNR, linear y
    for ai, (state_name, _segs) in enumerate(states):
        axp = axes[ai // 3][ai % 3]
        for est in ESTS[:2]:
            pts = rows_for(state_name, est)
            if pts:
                axp.plot([p["snr_bin"] for p in pts], [p["bias"] for p in pts],
                         marker=".", label=label_for(pts, est))
        axp.axhline(pred[state_name], ls="--", color="k", lw=1,
                    label=f"prediction {pred[state_name]:+.3f}")
        axp.set_title(state_name, fontsize=9)
        axp.legend(fontsize=7)
        axp.set_xlabel("SNR")
        axp.set_ylabel("signed bias px")
    axes[1][2].axis("off")
    fig.suptitle(f"bias vs SNR, trail axis +col (linear, log x); {note}")
    fig.savefig(f"{args.out}/bias_vs_snr.png", dpi=110, bbox_inches="tight")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))  # RMS error vs SNR, log-log
    for ai, (state_name, _segs) in enumerate(states):
        axp = axes[ai // 3][ai % 3]
        for est in ESTS:
            pts = rows_for(state_name, est)
            if pts:
                axp.plot([p["snr_bin"] for p in pts], [p["rms"] for p in pts],
                         marker=".", label=label_for(pts, est))
        cr = rows_for(state_name, "com")
        if cr:
            axp.plot([p["snr_bin"] for p in cr], [p["crlb_px"] for p in cr],
                     "k--", lw=1, label="CRLB (col)")
        axp.set_title(state_name, fontsize=9)
        axp.set_xscale("log")
        axp.set_yscale("log")
        axp.legend(fontsize=7)
        axp.set_xlabel("SNR")
        axp.set_ylabel("RMS error px (bias^2+scatter^2)")
    axes[1][2].axis("off")
    fig.suptitle(f"RMS error vs SNR, trail axis +col (log-log), all estimators and the bound; {note}")
    fig.savefig(f"{args.out}/rms_vs_snr.png", dpi=110, bbox_inches="tight")

    combos = [(s, sh) for s, _ in states for sh in cfg["shapes"]]
    samples = {tuple(k.split("|")): np.asarray(v) for k, v in table["samples"].items()}
    fig, axes = plt.subplots(len(combos), 9, figsize=(11, 1.15 * len(combos)))
    for ri, (s, sh) in enumerate(combos):  # one row per state x shape, 9 draws
        grid = samples.get((s, sh))
        for ci in range(9):
            axp = axes[ri][ci]
            if grid is not None:
                axp.imshow(grid[ci], cmap="gray")
            axp.set_xticks([])
            axp.set_yticks([])
        axes[ri][0].set_ylabel(f"{s}\n{sh}", fontsize=6)
    fig.suptitle(f"example draws at the middle bin (SNR 20), noise on, 9 draws per row; {note}")
    fig.savefig(f"{args.out}/samples_examples.png", dpi=110, bbox_inches="tight")
    print(f"plot_step2: wrote 4 PNGs to {args.out} (from {args.table})", flush=True)


if __name__ == "__main__":
    main()
