"""Step 8: evaluation on synthetic TEST data (guide step 8). Nothing here is
used for a choice (the hybrid rule and all models were chosen on val).

Written by Claude, 30 Oct 2026. Usage (from the centroid_cnn folder, after
run_step8.sh made the test sets and the ablation):
    source ../../dev_env.sh; python src/evaluate.py
Output: reports/step8/eval.txt, eval.json.

Methods: cnn = checkpoints/step7_full_v3_e120.pt; hybrid = cnn + gyro_fit from
the CNN start when the ch1 peak >= threshold (configs/hybrid.json); cnn_nogyro
= step7_nogyro_v3_e120.pt (same recipe, channel 1 only); baselines centre,
gauss, tetra3, gyro_fit (step 6; v2 test from reports/step6/estimates.npz).

Predictions BEFORE the run (from the val results):
- v2 test: hybrid beats gyro_fit in total rms and failures in every bin, ties
  in median at SNR >= 50 and on trails, rms<2 slightly worse on trails 10+.
  Biases |b| < 0.03 px except SNR 300+ (small n).
- signed along-trail bias (axis = measured light offset, i.e. toward the fast
  end): light-centre methods (gauss, tetra3) clearly positive and growing with
  the offset; gyro_fit, cnn, hybrid about 0.
- ablation: the gyro channel gain is visible on trails > 3 px (median -5 to
  -10 %) and about 0 on still stars.
- gyro_large vs gyro_ref: cnn median +20-50 %, failures up, no collapse;
  gyro_fit degrades by a similar amount (it uses the same channel 2).
- unseen sets: psf_narrow and phone worst (pixel locking on undersampled
  stars), psf_wide and bits14 close to in-range; sigma calibration 60-80 %.
- this_instrument: better than the "all" average (narrower PSF than many
  random cases).
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
import baselines as B  # noqa: E402
import check_step5  # noqa: E402
import hybrid as H  # noqa: E402
import metrics  # noqa: E402
import train_cnn as T  # noqa: E402

OUT = HERE / "reports" / "step8"
SETS = ["s8_this_instrument", "s8_gyro_ref", "s8_gyro_large", "s8_psf_narrow", "s8_psf_wide", "s8_corr_strong",
        "s8_bits14", "s8_phone"]
NOGYRO = "step7_nogyro_v3_e120"
BASE = ("centre", "gauss", "tetra3", "gyro_fit")
RULE = json.load(open(HERE / "configs" / "hybrid.json"))


def run_methods(X, y, base=None):
    mu, sig = H.cnn(X)
    gf = H.refine(X, mu)
    hyb, use = H.apply(RULE, mu, sig, gf, H.peak(X, mu))
    if base is None:
        with Pool(16) as pool:
            res = pool.map(B.one, list(X), chunksize=50)
        base = {m: np.array([r[0][m] for r in res]) for m in BASE}
    errs = {m: base[m] - y for m in BASE}
    errs.update({"cnn": mu - y, "hybrid": hyb - y})
    if (HERE / "checkpoints" / f"{NOGYRO}.pt").exists():
        errs["cnn_nogyro"] = H.cnn(X, NOGYRO)[0] - y
    return errs, sig, use


def light_offset(X):
    """Measured light centre minus mid-time point (ch2 first moment), px."""
    ch2 = X[:, 1].astype(float)
    return np.stack([(ch2 * B.RR).sum((1, 2)), (ch2 * B.CC).sum((1, 2))], 1) - B.CEN


def sigma_cal(e, sig, m=None):
    z = np.abs(e) / sig
    z = z if m is None else z[m]
    return np.mean(z < 1), np.mean(z < 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L, J = [], {"rule": RULE}
    X, y, P = check_step5.load_split(HERE / "datasets" / "v2", "test")
    est = dict(np.load(HERE / "reports" / "step6" / "estimates.npz"))
    assert np.array_equal(est["y"], y), "estimates.npz is not in the test order"
    errs, sig, use = run_methods(X, y, est)
    inst = (P["psf_inst"] > 0) & (P["bg_inst"] > 0) & (P["tone_inst"] > 0) & (P["motion_inst"] > 0)
    subsets = {"this instrument (all 4)": inst, "bad (artefact)": P["bad"] > 0,
               "still, snr>=20": (P["centre_len_px"] < 1) & (P["snr"] >= 20),
               "trail>3, snr>=20": (P["centre_len_px"] > 3) & (P["snr"] >= 20)}
    tab, lines = metrics.table(errs, P, X, subsets)
    J["v2_test"] = tab
    L += [f"STEP 8 -- v2 TEST split (n {len(y)}); hybrid rule {json.dumps(RULE)}",
          f"hybrid refined fraction {use.mean():.3f}", ""] + lines

    # acceptance: hybrid (and cnn) vs gyro_fit per bin
    L += ["", "ACCEPTANCE vs gyro_fit (best classical): rms = all samples; bias worse = |b| larger by > 2 SE",
          f"{'group':24s} {'method':8s} {'rms':>12s} {'rms<2':>12s} {'fail':>12s} {'bias':>10s}"]
    for g, row in tab.items():
        if "gyro_fit" not in row:
            continue
        gref = row["gyro_fit"]
        for meth in ("cnn", "hybrid"):
            s = row[meth]
            n = s["n"]
            se = 2 * s["rms"] / np.sqrt(max(n, 1))
            bw = any(abs(s[k]) - abs(gref[k]) > se for k in ("bias_row", "bias_col"))
            L.append(f"{g:24s} {meth:8s} {s['rms']:5.3f}/{gref['rms']:5.3f}{'+' if s['rms'] < gref['rms'] else '-'} "
                     f"{s['rms_no_fail']:5.3f}/{gref['rms_no_fail']:5.3f}{'+' if s['rms_no_fail'] < gref['rms_no_fail'] else '-'} "
                     f"{s['fail']:5.3f}/{gref['fail']:5.3f}{'+' if s['fail'] <= gref['fail'] else '-'} "
                     f"{'WORSE' if bw else 'ok':>10s}")
    L.append("(+ = better than gyro_fit, - = worse; format method/gyro_fit)")
    for g in ("snr 2-5", "snr 5-10"):
        c, h = tab[g]["centre"], tab[g]["hybrid"]
        L.append(f"vs centre, {g}: hybrid rms {h['rms']:.3f} vs {c['rms']:.3f}, fail {h['fail']:.3f} vs {c['fail']:.3f}")

    # signed along-trail bias (toward the fast end)
    m1 = light_offset(X)
    a = np.linalg.norm(m1, axis=1)
    u = m1 / np.maximum(a, 1e-9)[:, None]
    L += ["", "SIGNED bias along the measured light offset (+ = toward the fast end), px; across in ()"]
    edges = [0.2, 0.5, 1.0, 2.0, 1e9]
    L.append(f"{'|offset| px':12s} {'n':>5s} " + " ".join(f"{m:>16s}" for m in errs))
    J["signed_bias"] = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a >= lo) & (a < hi)
        cells = []
        for meth, e in errs.items():
            al = np.mean(np.sum(e[m] * u[m], 1))
            ac = np.mean(e[m, 0] * u[m, 1] - e[m, 1] * u[m, 0])
            cells.append(f"{al:+7.3f} ({ac:+6.3f})")
            J["signed_bias"].setdefault(f"{lo:g}-{hi:g}", {})[meth] = [float(al), float(ac)]
        L.append(f"{lo:g}-{hi:g}".ljust(12) + f" {m.sum():5d} " + " ".join(f"{c:>16s}" for c in cells))

    # ablation
    if "cnn_nogyro" in errs:
        L += ["", "ABLATION (gyro channels): cnn vs cnn_nogyro, median / rms<2 / fail"]
        for g in ("still, snr>=20", "trail 0-1", "trail 1-3", "trail 3-10", "trail 10+", "trail>3, snr>=20", "all"):
            c, n = tab[g]["cnn"], tab[g]["cnn_nogyro"]
            L.append(f"  {g:20s} cnn {c['median']:.3f}/{c['rms_no_fail']:.3f}/{c['fail']:.3f}   "
                     f"nogyro {n['median']:.3f}/{n['rms_no_fail']:.3f}/{n['fail']:.3f}   "
                     f"median gain {100 * (1 - c['median'] / n['median']):+.1f} %")

    # sigma calibration, pixel locking
    L += ["", "SIGMA calibration (cnn): fraction |e|/sigma < 1 / < 2 per axis (Gaussian 0.683 / 0.954)"]
    f1, f2 = sigma_cal(errs["cnn"], sig)
    L.append(f"  all {f1:.3f} / {f2:.3f}")
    for lo, hi in ((2, 5), (5, 20), (20, 300), (300, 1e9)):
        m = (P["snr"] >= lo) & (P["snr"] < hi)
        f1, f2 = sigma_cal(errs["cnn"][m], sig[m])
        L.append(f"  snr {lo:g}-{hi:g}: {f1:.3f} / {f2:.3f}")
    L += ["", "PIXEL LOCKING (snr >= 20): mean error per axis in 5 bins of the fractional true position; span px"]
    frac = (B.CEN + y) % 1.0
    hs = P["snr"] >= 20
    for meth in ("gyro_fit", "cnn", "hybrid"):
        spans = []
        for ax in (0, 1):
            b = [errs[meth][hs & (frac[:, ax] >= f) & (frac[:, ax] < f + 0.2), ax].mean() for f in np.arange(0, 1, 0.2)]
            spans.append(max(b) - min(b))
        L.append(f"  {meth:10s} row span {spans[0]:.3f}  col span {spans[1]:.3f}")

    # test sets (robustness, generality)
    L += ["", "TEST SETS (5000 each, test windows): all samples; median / rms<2 / fail / rms / b_row / b_col; "
          "sigma cal (cnn) <1"]
    ref = tab["all"]
    L.append(f"  {'v2 test (reference)':22s} " + "  ".join(
        f"{m} {ref[m]['median']:.3f}/{ref[m]['rms_no_fail']:.3f}/{ref[m]['fail']:.3f}" for m in ("gyro_fit", "cnn", "hybrid")))
    J["sets"] = {}
    for s in SETS:
        root = HERE / "datasets" / s
        if not (root / "meta.json").exists():
            L.append(f"  {s}: MISSING (generation failed?)")
            continue
        Xs, ys, Ps = check_step5.load_split(root, "test")
        e2, sg2, use2 = run_methods(Xs, ys)
        u2 = metrics.trail_dirs(Xs)
        J["sets"][s] = {}
        L.append(f"  {s} (n {len(ys)}, hybrid refined {use2.mean():.3f}, sigma cal {sigma_cal(e2['cnn'], sg2)[0]:.3f}, "
                 f"snr p50 {np.median(Ps['snr']):.0f}, trail p50 {np.median(Ps['centre_len_px']):.1f} px)")
        for meth, e in e2.items():
            st = metrics.stats(e, u2)
            J["sets"][s][meth] = st
            L.append(f"      {meth:11s} {st['median']:.3f} / {st['rms_no_fail']:.3f} / {st['fail']:.3f} / "
                     f"{st['rms']:.3f} / {st['bias_row']:+.3f} / {st['bias_col']:+.3f}")

    # Pi budget
    ck = torch.load(HERE / "checkpoints" / f"{H.MODEL}.pt", map_location="cpu")
    net = T.Net(ck["in_ch"], arch=ck.get("arch", "a"))
    macs, npar = T.count_macs(net)
    L += ["", f"PI BUDGET: cnn {npar} parameters, {macs / 1e6:.2f} M multiply-adds per patch; hybrid adds gyro_fit "
          f"({RULE.get('gyro_fit_ms_per_star_laptop', float('nan')):.1f} ms per star on one laptop core) on "
          f"{use.mean() * 100:.1f} % of the stars (Pi timing: step 10)"]
    txt = "\n".join(L)
    (OUT / "eval.txt").write_text(txt + "\n")
    json.dump(J, open(OUT / "eval.json", "w"), indent=1, default=float)
    print(txt)


if __name__ == "__main__":
    main()
