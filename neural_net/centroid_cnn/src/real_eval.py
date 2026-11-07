"""Step 9: real data. Replace the production centroids by the CNN (and hybrid)
centroids and solve again (guide step 9).

Written by Claude, 30 Oct 2026. Run from the centroid_cnn folder:
    source ../../dev_env.sh
    python src/real_eval.py 2026-10-17_16-00-56 [--n 20] [--workers 8]
    python src/real_eval.py --summary 2026-10-17_16-00-56 2026-10-18_16-19-09
Output (this folder only; runs/ is read-only): reports/step9/<run>_frames.jsonl
(one line per saved image), reports/step9/summary.txt.

Per saved image:
- production centroids = cedar-detect with the run's own config (sigma 3.75,
  binning 2, hot-pixel removal), first max_centroids (95), as replay_solve.py.
- CNN input per centroid (CONVENTIONS): patch 32 x 32, centre pixel =
  floor(cedar centroid); ch1 = (Y - median of the outer 3-px frame) /
  sigma_frame; sigma_frame = median over 64 x 64 tiles of the std after 5-sigma
  clipping (the generator: std of an empty 64 x 64 patch). ch2/ch3 = dwell map
  and time ramp of the MEASURED path: gyro (camera frame, w = M_case w_imu)
  minus its mean over +-150 s, linear interpolation, 1 ms steps, over the 300 ms
  of the star's row; star mid time on the IMU clock = t_exposure_mid + 10 ms
  (nominal tau, CONVENTIONS) + row / 1296 x 18 ms (nominal readout);
  projection = motion.project_path with configs/step4.json camera.
- A star keeps its production centroid if its patch leaves the frame or its
  measured path goes > 14 px from the mid-time point (as the generator).
- Three solves with the same tetra3 settings and star order: production,
  cnn (checkpoints/step7_full_v3_e120.pt), hybrid (configs/hybrid.json).
- Fix of each solve as replay_solve.py (run calibration), error vs gps.jsonl.

Added 30 Oct (run 2, test runs 14 and 20 Oct): per star the expected light
offset m1 = first moment of ch2 - 16.5 (measured light centre minus mid-time
point); --tau (20 Oct had +64 ms), --max-min (20 Oct: first 75 min only),
--tag (output name). Summary: slope of the CNN - cedar shift on m1 (about -1
if cedar gives the light centre and the CNN the mid-time point), and the fix
error by roll phase (sign of the frame-mean m1).
Prediction run 2: slope -0.6 to -1.0 on stars with |m1| > 0.3 px on all
nights; 20 Oct with tau 10 ms: slope closer to 0 or noisy (wrong timing), with
tau 64 ms: clearer; fix error between roll phases: the prod spread is larger
than the cnn spread, but within the noise (SE about 1 nmi).

Predictions BEFORE the first run:
- the CNN - cedar mean offset per axis is either about 0 (cedar uses the corner
  convention, as the 16 Oct analysis says) or about +-0.5 px (centre
  convention). Its size tells which; it shifts all fixes, not their scatter.
- solve RMSE on paired frames: cnn lower than production by 5-15 % (the CNN
  removes the along-trail bias and halves the failures); hybrid about = cnn.
- solve rate: same within 1 %; false solves not more.
- fix scatter (robust std per axis): same or a few % lower; the fix offset
  changes by the convention shift only.
"""
import argparse
import json
import os
import sys
import time
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent.parent
PIPE = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(PIPE))
import baselines as B  # noqa: E402
import motion as M  # noqa: E402

OUT = HERE / "reports" / "step9"
CFG4 = json.load(open(HERE / "configs" / "step4.json"))
CAM = CFG4["camera"]
RULE = json.load(open(HERE / "configs" / "hybrid.json"))
TAU0, READOUT, OFFSET_S, MAX_EXT = 0.010, 0.018, 150.0, CFG4["patch"]["max_extent_px"]
G = {}          # per-process state (filled before the fork and in init_worker)


def jsonl(p):
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run(run):
    rd = PIPE / "runs" / run
    meta = json.loads((rd / "run_meta.json").read_text())
    t, w = [], []
    with open(rd / "imu.jsonl") as f:                 # stream (large file)
        for line in f:
            r = json.loads(line)
            if r.get("gyro") is not None:
                t.append(r["t"])
                w.append(r["gyro"])
    t = np.array(t)
    o = np.argsort(t)
    t, w = t[o], (np.array(w) @ np.array(meta["M_case"]).T)[o]
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(w, 0)])
    temps = [r for r in jsonl(rd / "temp.jsonl") if r.get("t") is not None and r.get("temp_raw") is not None]
    gps = [r for r in jsonl(rd / "gps.jsonl") if r.get("type") == "fix"]
    caps = [c for c in jsonl(rd / "capture.jsonl") if c.get("image_saved") and (rd / c["image"]).exists()]
    G.update(run=run, rd=rd, meta=meta, t=t, w=w, cs=cs, caps=caps,
             temp_t=np.array([r["t"] for r in temps]), temp_v=np.array([r["temp_raw"] for r in temps]),
             gps_t=np.array([r["t_gps"] for r in gps]), gps_lat=np.array([r["lat"] for r in gps]),
             gps_lon=np.array([r["lon"] for r in gps]))


def init_worker():
    import torch
    import tetra3
    from tetra3 import cedar_detect_client
    import replay_solve as RS
    import inspect
    import train_cnn as T
    torch.set_num_threads(1)
    cfg = G["meta"]["solver"]["config"]
    t3 = tetra3.Tetra3(load_database=RS.resolve_database(cfg, None))
    ck = torch.load(HERE / "checkpoints" / f"{RULE['model']}.pt", map_location="cpu")
    net = T.Net(ck["in_ch"], ck["cfg"]["ch1_k"], ck["cfg"]["ch2_scale"], ck.get("arch", "a"))
    net.load_state_dict(ck["state"])
    net.eval()
    G.update(RS=RS, torch=torch, t3=t3, net=net, cfg=cfg,
             names=set(inspect.signature(t3.solve_from_centroids).parameters),
             cedar=cedar_detect_client.CedarDetectClient(binary_path=os.environ["CEDAR_DETECT_BINARY"],
                                                         port=50100 + mp.current_process()._identity[0]))
    G["cedar"]._use_shmem = False       # one fixed shared-memory name: not safe with parallel workers


def sigma_frame(gray):
    h, w = gray.shape
    s = []
    for r in range(0, h - 63, 64):
        for c in range(0, w - 63, 64):
            v = gray[r:r + 64, c:c + 64].astype(float).ravel()
            for _ in range(3):
                m, sd = np.median(v), v.std()
                v = v[np.abs(v - m) < 5 * max(sd, 1e-6)]
            s.append(v.std())
    return float(np.median(s))


def measured_path(rc, t_star):
    dt = CFG4["path_dt_s"]
    tg = t_star + np.arange(-0.15, 0.15 + dt / 2, dt)
    t, w, cs = G["t"], G["w"], G["cs"]
    lo, hi = np.searchsorted(t, [t_star - OFFSET_S, t_star + OFFSET_S])
    off = (cs[hi] - cs[lo]) / max(hi - lo, 1)
    wg = np.stack([np.interp(tg, t, w[:, j]) for j in range(3)], -1) - off
    return M.project_path(M.integrate(tg, wg, t_star), rc, CAM)


def build_inputs(gray, cents, t_mid):
    sig = sigma_frame(gray)
    H, W = gray.shape
    X, idx, flags = [], [], []
    for i, (r, c) in enumerate(cents):
        r0, c0 = int(np.floor(r)) - 16, int(np.floor(c)) - 16
        if r0 < 0 or c0 < 0 or r0 + 32 > H or c0 + 32 > W:
            flags.append("edge")
            continue
        d = measured_path(np.array([r, c]), t_mid + G.get("tau", TAU0) + r / CAM["height"] * READOUT)
        if np.abs(d).max() > MAX_EXT:
            flags.append("long")
            continue
        p = gray[r0:r0 + 32, c0:c0 + 32].astype(float)
        frame = np.concatenate([p[:3].ravel(), p[-3:].ravel(), p[3:-3, :3].ravel(), p[3:-3, -3:].ravel()])
        ch2, ch3 = M.channels(d, CFG4)
        X.append(np.stack([(p - np.median(frame)) / sig, ch2, ch3]))
        m1 = [float((ch2 * B.RR).sum() - B.CEN), float((ch2 * B.CC).sum() - B.CEN)]
        idx.append((i, r0, c0, float(np.abs(d).max()), m1))
        flags.append("cnn")
    return np.array(X, np.float32), idx, flags, sig


def solve(cents, shape):
    s = G["meta"]["solver"]["config"]["solve"]
    kw = {"size": shape, "fov_estimate": s.get("fov_estimate"), "fov_max_error": s.get("fov_max_error"),
          "pattern_checking_stars": s.get("pattern_checking_stars", 8), "match_radius": s.get("match_radius", 0.01),
          "match_threshold": s.get("match_threshold", 1e-3), "solve_timeout": s.get("solve_timeout", 3000),
          "distortion": s.get("distortion", 0), "return_catalog": False}
    kw = {k: v for k, v in kw.items() if k in G["names"]}
    sol = G["t3"].solve_from_centroids(np.asarray(cents), **kw)
    return sol if sol is not None and sol.get("RA") is not None else None


def fix_of(sol, cap):
    from astropy.time import Time
    meta, RS = G["meta"], G["RS"]
    g = cap["imu_window"]["gravity_median"]
    if sol is None or g is None:
        return None
    temp = float(np.interp(cap["t_exposure_mid"], G["temp_t"], G["temp_v"])) if len(G["temp_t"]) else None
    f = RS.calculate_fix(sol["RA"], sol["Dec"], sol["Roll"], g, Time(cap["t_exposure_mid"], format="unix"),
                         np.array(meta["M_case"]), np.array(meta["R_cal"]),
                         np.array(meta.get("cal_b") or [0, 0, 0]),
                         np.array(meta["cal_c"]) if meta.get("cal_c") is not None else None,
                         temp, float(meta.get("cal_temp_ref_c") or 25.0))
    glat = float(np.interp(cap["t_exposure_mid"], G["gps_t"], G["gps_lat"]))
    glon = float(np.interp(cap["t_exposure_mid"], G["gps_t"], G["gps_lon"]))
    return {"lat": f["latitude"], "lon": f["longitude"], "err_n_nmi": (f["latitude"] - glat) * 60,
            "err_e_nmi": ((f["longitude"] - glon + 180) % 360 - 180) * 60 * np.cos(np.radians(glat))}


def one_frame(k):
    cap = G["caps"][k]
    torch = G["torch"]
    t0 = time.perf_counter()
    gray = np.asarray(Image.open(G["rd"] / cap["image"]))
    dc = G["cfg"]["detect"]
    cents = np.array(G["cedar"].extract_centroids(gray, sigma=float(dc["sigma"]),
                                                  use_binned=int(dc.get("binning", 1)) == 2,
                                                  detect_hot_pixels=True), float).reshape(-1, 2)
    cents = cents[:int(dc.get("max_centroids", 30))]
    rec = {"frame_id": cap["frame_id"], "t": cap["t_exposure_mid"], "n_centroids": len(cents)}
    if len(cents) == 0:
        rec["status"] = "no_stars"
        return rec
    X, idx, flags, sig = build_inputs(gray, cents, cap["t_exposure_mid"])
    cnn, hyb = cents.copy(), cents.copy()
    shift, refined = [], 0
    if len(X):
        with torch.no_grad():
            mu, ls = G["net"](torch.from_numpy(X))
        mu, sg = mu.numpy().astype(float), np.exp(ls.numpy())
        for j, (i, r0, c0, ext, m1) in enumerate(idx):
            cnn[i] = [r0 + 16.5 + mu[j, 0], c0 + 16.5 + mu[j, 1]]
            hyb[i] = cnn[i]
            rr, cc = (np.floor(mu[j] + 16.5)).astype(int).clip(2, 29)
            if X[j, 0, rr - 2:rr + 3, cc - 2:cc + 3].max() >= RULE["threshold"]:
                g = B.gyro_fit(X[j, 0].astype(float), X[j, 1].astype(float), mu[j] + 16.5)[0]
                hyb[i] = [r0 + g[0], c0 + g[1]]
                refined += 1
            shift.append([*(cnn[i] - cents[i]), float(np.exp(np.log(sg[j]).mean())), ext,
                          float(X[j, 0].max()), *m1])
    t_cnn = time.perf_counter() - t0
    rec.update({"sigma_frame": sig, "n_cnn": len(idx), "n_edge": flags.count("edge"), "n_long": flags.count("long"),
                "n_refined": refined, "shift": shift, "t_prep_s": round(t_cnn, 3)})
    for name, cc in (("prod", cents), ("cnn", cnn), ("hybrid", hyb)):
        sol = solve(cc, gray.shape)
        rec[name] = None if sol is None else {
            "RA": sol["RA"], "Dec": sol["Dec"], "Roll": sol["Roll"], "FOV": sol["FOV"], "RMSE": sol.get("RMSE"),
            "Matches": sol.get("Matches"), "distortion": sol.get("distortion"), "fix": fix_of(sol, cap)}
    rec["status"] = "done"
    return rec


def run(run_id, n, workers, tau=TAU0, max_min=0, tag=""):
    OUT.mkdir(parents=True, exist_ok=True)
    load_run(run_id)
    G["tau"] = tau
    if max_min:
        t0c = G["caps"][0]["t_exposure_mid"]
        G["caps"] = [c for c in G["caps"] if c["t_exposure_mid"] < t0c + 60 * max_min]
    caps = G["caps"]
    ks = list(range(len(caps)))[:n] if n else list(range(len(caps)))
    out = OUT / (f"{run_id}{tag}_frames.jsonl" if not n else f"{run_id}{tag}_smoke.jsonl")
    print(f"{run_id}: {len(ks)} saved images, {len(G['t'])} gyro samples, workers {workers}", flush=True)
    t0 = time.time()
    with Pool(workers, initializer=init_worker) as pool, open(out, "w") as f:
        for j, rec in enumerate(pool.imap(one_frame, ks, chunksize=4)):
            f.write(json.dumps(rec, default=float) + "\n")
            if (j + 1) % 100 == 0:
                print(f"  {j + 1}/{len(ks)}  {time.time() - t0:.0f} s", flush=True)
    print(f"done -> {out} in {time.time() - t0:.0f} s", flush=True)


def robust(x):
    x = np.asarray(x, float)
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else float("nan")


def summary(runs):
    L = []
    for run_id in runs:
        recs = [r for r in jsonl(OUT / f"{run_id}_frames.jsonl")]
        done = [r for r in recs if r["status"] == "done"]
        sh = np.array([s for r in done for s in r["shift"]])
        L += [f"=== {run_id}: {len(recs)} saved images, {len(done)} with stars",
              f"stars per frame: centroids median {np.median([r['n_centroids'] for r in done]):.0f}, CNN used "
              f"{np.sum([r['n_cnn'] for r in done]) / max(np.sum([r['n_centroids'] for r in done]), 1):.3f} "
              f"(edge {np.sum([r['n_edge'] for r in done])}, path > 14 px {np.sum([r['n_long'] for r in done])}), "
              f"hybrid refined {np.sum([r['n_refined'] for r in done])} stars; sigma_frame median "
              f"{np.median([r['sigma_frame'] for r in done]):.2f} DN",
              f"CNN - cedar shift, all used stars (n {len(sh)}): mean row {sh[:, 0].mean():+.3f} col {sh[:, 1].mean():+.3f} "
              f"px, median {np.median(sh[:, 0]):+.3f} {np.median(sh[:, 1]):+.3f}, robust std "
              f"{robust(sh[:, 0]):.3f} {robust(sh[:, 1]):.3f}"]
        for lo, hi in ((0, 10), (10, 30), (30, 100), (100, 1e9)):
            m = (sh[:, 4] >= lo) & (sh[:, 4] < hi)
            if m.sum() > 20:
                L.append(f"   ch1 peak {lo:g}-{hi:g} sigma: n {m.sum():6d}  mean shift {sh[m, 0].mean():+.3f} "
                         f"{sh[m, 1].mean():+.3f}  robust std {robust(sh[m, 0]):.3f} {robust(sh[m, 1]):.3f}  "
                         f"CNN sigma median {np.median(sh[m, 2]):.3f}")
        meths = ("prod", "cnn", "hybrid")
        for m in meths:
            sol = [r[m] for r in done if r.get(m)]
            L.append(f"{m:7s} solved {len(sol)}/{len(recs)} ({len(sol) / max(len(recs), 1):.3f})")
        paired = [r for r in done if all(r.get(m) for m in meths)]
        L.append(f"paired frames (all three solved): {len(paired)}")
        for m in meths:
            rm = np.array([r[m]["RMSE"] for r in paired], float)
            ma = np.array([r[m]["Matches"] for r in paired], float)
            ratio = rm / np.array([r["prod"]["RMSE"] for r in paired], float)
            L.append(f"  {m:7s} RMSE median {np.median(rm):6.2f} arcsec, mean {rm.mean():6.2f}; ratio to prod median "
                     f"{np.median(ratio):.3f}, frames lower than prod {np.mean(ratio < 1):.3f}; matches median "
                     f"{np.median(ma):.0f} mean {ma.mean():.1f}")
        L.append("  fix error vs GPS, nmi (paired frames with a fix): median N / E (offset), robust std N / E, "
                 "median |e|, frames > 300 nmi from the night median (false solves); 60 s means: median |e|")
        for m in meths:
            fx = [(r["t"], r[m]["fix"]) for r in paired if r[m]["fix"]]
            t = np.array([a for a, _ in fx])
            en = np.array([b["err_n_nmi"] for _, b in fx])
            ee = np.array([b["err_e_nmi"] for _, b in fx])
            mag = np.hypot(en - np.median(en), ee - np.median(ee))
            ok = mag < 300
            sm = []
            for tc in np.arange(t.min(), t.max(), 60.0):
                k = ok & (t >= tc) & (t < tc + 60)
                if k.sum() >= 5:
                    sm.append(np.hypot(en[k].mean(), ee[k].mean()))
            L.append(f"  {m:7s} offset {np.median(en[ok]):+6.2f} {np.median(ee[ok]):+6.2f}  robust std "
                     f"{robust(en[ok]):6.2f} {robust(ee[ok]):6.2f}  median |e| {np.median(np.hypot(en[ok], ee[ok])):6.2f}"
                     f"  false {np.sum(~ok)}  60 s: median |e| {np.median(sm):5.2f} (n {len(sm)})")
        if sh.shape[1] >= 7:
            L.append("  light-offset test (stars with |m1| > 0.3 px): CNN - cedar shift vs m1 = expected light centre "
                     "- mid-time (slope -1 = CNN at mid time, cedar at light centre)")
            mm = np.hypot(sh[:, 5], sh[:, 6]) > 0.3
            for ax, an in ((0, "row"), (1, "col")):
                x, yv = sh[mm, 5 + ax], sh[mm, ax]
                sl = np.polyfit(x, yv, 1)[0] if mm.sum() > 20 else float("nan")
                L.append(f"    {an}: n {mm.sum():6d}  slope {sl:+.3f}")
            u = sh[:, 5:7] / np.maximum(np.hypot(sh[:, 5], sh[:, 6]), 1e-9)[:, None]
            along = np.sum(sh[:, :2] * u, 1)
            for lo, hi in ((0.1, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 1e9)):
                k = (np.hypot(sh[:, 5], sh[:, 6]) >= lo) & (np.hypot(sh[:, 5], sh[:, 6]) < hi)
                if k.sum() > 20:
                    L.append(f"    |m1| {lo:g}-{hi:g} px: n {k.sum():6d}  mean |m1| {np.hypot(sh[k, 5], sh[k, 6]).mean():.3f}"
                             f"  mean shift along m1 {along[k].mean():+.3f}  (expected -|m1| if cedar = light centre)")
            L.append("  fix error by roll phase (sign of the frame-mean m1 row / col, frames with |mean m1| > 0.1 px): "
                     "mean N / E nmi")
            fm = {r["frame_id"]: np.mean([s[5:7] for s in r["shift"]], 0) for r in paired if r["shift"]}
            for m in meths:
                cells, means = [], []
                for sr in (-1, 1):
                    for sc in (-1, 1):
                        k = [r for r in paired if r[m]["fix"] and r["frame_id"] in fm
                             and np.hypot(*fm[r["frame_id"]]) > 0.1
                             and np.sign(fm[r["frame_id"]][0]) == sr and np.sign(fm[r["frame_id"]][1]) == sc]
                        en = np.array([r[m]["fix"]["err_n_nmi"] for r in k])
                        ee = np.array([r[m]["fix"]["err_e_nmi"] for r in k])
                        ok = np.hypot(en - np.median(en), ee - np.median(ee)) < 300 if len(k) else []
                        if len(k) > 10:
                            means.append((en[ok].mean(), ee[ok].mean()))
                            cells.append(f"({sr:+d},{sc:+d}) n {ok.sum():4d} {en[ok].mean():+6.2f} {ee[ok].mean():+6.2f}")
                if means:
                    a_ = np.array(means)
                    L.append(f"    {m:7s} " + " | ".join(cells) + f"  spread N {np.ptp(a_[:, 0]):.2f} E {np.ptp(a_[:, 1]):.2f}")
        L.append("")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n", type=int, default=0, help="smoke test: first n saved images")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--tau", type=float, default=TAU0, help="IMU time offset, s (nominal 0.010)")
    ap.add_argument("--max-min", type=float, default=0, help="only the first N minutes")
    ap.add_argument("--tag", default="", help="suffix of the output name")
    a = ap.parse_args()
    if a.summary:
        summary(a.runs)
        return
    for r in a.runs:
        run(r, a.n, a.workers, a.tau, a.max_min, a.tag)


if __name__ == "__main__":
    main()
