"""Error tables for steps 6 and 8 (guide step 8: rms, bias per axis, p95, in
bins of SNR, trail length, image region, gyro-error size, neighbour distance;
bias also along / across the trail).

Written by Claude, 29 Oct 2026. Used by src/baselines.py (and step 8).
Errors e = estimate - label, px, (row, col). A "failure" is |e| > FAIL_PX;
rms, bias and p95 are given over ALL samples and the failure rate apart, so a
method cannot hide failures.
"""
import numpy as np

FAIL_PX = 2.0

BINS = {
    "snr": ("snr", [2, 5, 10, 20, 50, 300, 1e9]),
    "trail": ("centre_len_px", [0, 1, 3, 10, 1e9]),
    "radius": ("r_px", [0, 600, 1100, 1e9]),
    "gyro_err": ("meas_err_rms_px", [0, 0.1, 0.3, 1.0, 1e9]),
    "neighbour": ("nearest_neigh_px", [0, 8, 16, 1e9]),
}


def trail_dirs(X):
    """Unit trail direction (row, col) per sample from channels 2 and 3
    (late minus early dwell centroid); 0 where shorter than 0.5 px."""
    ch2, ch3 = X[:, 1].astype(float), X[:, 2].astype(float)
    rr, cc = np.mgrid[0:32, 0:32] + 0.5
    early, late = ch2 * (1 - ch3), ch2 * ch3
    e = np.stack([(early * rr).sum((1, 2)), (early * cc).sum((1, 2))], 1) / np.maximum(early.sum((1, 2)), 1e-9)[:, None]
    l = np.stack([(late * rr).sum((1, 2)), (late * cc).sum((1, 2))], 1) / np.maximum(late.sum((1, 2)), 1e-9)[:, None]
    d = l - e
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return np.where(n > 0.5, d / np.maximum(n, 1e-9), 0.0)


def stats(e, u):
    """e (n, 2) errors, u (n, 2) trail directions (0 = no trail)."""
    r = np.linalg.norm(e, axis=1)
    t = np.linalg.norm(u, axis=1) > 0
    along = np.sum(e * u, 1)
    across = e[:, 0] * u[:, 1] - e[:, 1] * u[:, 0]
    return {"n": int(len(e)), "rms": float(np.sqrt(np.mean(r ** 2))), "bias_row": float(e[:, 0].mean()),
            "bias_col": float(e[:, 1].mean()), "p95": float(np.percentile(r, 95)),
            "median": float(np.median(r)), "fail": float(np.mean(r > FAIL_PX)),
            "rms_no_fail": float(np.sqrt(np.mean(r[r <= FAIL_PX] ** 2))) if (r <= FAIL_PX).any() else float("nan"),
            "bias_along": float(along[t].mean()) if t.sum() > 5 else float("nan"),
            "bias_across": float(across[t].mean()) if t.sum() > 5 else float("nan")}


def table(errors, P, X, subsets=None):
    """errors: {method: (n, 2)}; P: parameter arrays; X: inputs (for the trail
    direction). Output: dict and text lines."""
    u = trail_dirs(X)
    out, lines = {}, []
    groups = [("all", np.ones(len(X), bool))]
    for name, (key, edges) in BINS.items():
        for lo, hi in zip(edges[:-1], edges[1:]):
            groups.append((f"{name} {lo:g}-{hi:g}".replace("-1e+09", "+"), (P[key] >= lo) & (P[key] < hi)))
    for name, m in (subsets or {}).items():
        groups.append((name, m))
    head = f"{'group':22s} {'method':14s} {'n':>6s} {'rms':>7s} {'rms<2':>7s} {'median':>7s} {'p95':>7s} {'fail':>6s} " \
           f"{'b_row':>7s} {'b_col':>7s} {'b_along':>8s} {'b_across':>8s}"
    for g, m in groups:
        if m.sum() < 20:
            continue
        lines.append(head if not lines else "")
        for meth, e in errors.items():
            s = stats(e[m], u[m])
            out.setdefault(g, {})[meth] = s
            lines.append(f"{g:22s} {meth:14s} {s['n']:6d} {s['rms']:7.3f} {s['rms_no_fail']:7.3f} {s['median']:7.3f} "
                         f"{s['p95']:7.3f} {s['fail']:6.3f} {s['bias_row']:7.3f} {s['bias_col']:7.3f} "
                         f"{s['bias_along']:8.3f} {s['bias_across']:8.3f}")
    return out, lines
