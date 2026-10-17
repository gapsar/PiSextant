#!/usr/bin/env python3
"""Multi-heading position fix -- runs on the laptop.

Every error that is tied to the instrument (accelerometer offset, its
temperature drift, a board that slipped after a shock, calibration residual)
moves the fix in a direction that turns with the instrument. With the
instrument at several headings, that error can be separated from the
position. Per frame (complex numbers in a local plane, N + iE, in nmi):

    fix_i - displacement_i = P + exp(i * heading_i) * e0

  P            the position at the reference time (what we want)
  e0           the instrument-fixed error, as seen at heading 0 (north)
  heading_i    azimuth of the camera boresight, from the star solve itself
               (no compass needed), clockwise from north
  displacement the vessel's motion since the reference time: zero for a
               static instrument (berth, anchor); under way it comes from
               --dr-from-gps (the GPS track is used ONLY for the relative
               motion, like a log + compass would give, never for P)

Requirements for a good result:
  - headings spread by at least ~60 deg (180 deg apart is best, and headings
    on both sides of the circle make the result robust);
  - the instrument at the SAME tilt in every segment (flat on the same
    surface, only turned): runs at another tilt are excluded automatically;
  - no shock between segments.

Usage (static, berth/anchor):
    python multiheading_fix.py runs/<r1> runs/<r2> runs/<r3> ...
    ... --position=48.8566,2.3522      # optional: truth, for evaluation only
    ... --cal configs/r_cal_ism330dhcx.json  # optional: recompute every fix with
                                             # one calibration (default: the fixes
                                             # logged live by the capture)
Under way (testing against GPS):
    python multiheading_fix.py runs/<r1> ... --dr-from-gps

(write --position=-21.0,55.3 with '=' when the latitude is negative)
"""

import argparse
import glob
import json
import math
import os

import numpy as np

import calibrate_rcal as cr

BIN_S = 60.0           # frames are averaged per minute before the fit (they
                       # are not independent at shorter scales)
TILT_TOL_DEG = 5.0     # runs whose mean tilt differs more than this from the
                       # majority are excluded (the model assumes one tilt)
OUTLIER_MAD = 5.0      # per-run outlier cut on the fix, in median deviations
SAME_HEADING_DEG = 15.0      # runs this close in heading must agree ...
SAME_HEADING_TOL_NMI = 1.5   # ... to within this (two flat runs at one heading
                             # agreed to ~0.1-0.5 nmi on 2026-10-12)


# --- loading -----------------------------------------------------------------

def load_temps(run_dir):
    p = os.path.join(run_dir, "temp.jsonl")
    if not os.path.exists(p):
        return None
    recs = [r for r in cr.load_jsonl(p) if r.get("t") is not None and r.get("temp_raw") is not None]
    if not recs:
        return None
    return np.array([r["t"] for r in recs]), np.array([r["temp_raw"] for r in recs])


def load_cal(path):
    cal = json.load(open(path))
    c = np.array(cal["c"]) if cal.get("c") is not None else None
    return np.array(cal["A"]), np.array(cal["b"]), c, float(cal.get("temp_ref_c", 25.0))


def load_frames(run_dir, cal=None, skip_min=0.0):
    """Solved frames of one run: time, fix (lat, lon), camera boresight in
    ICRS, gravity direction in the IMU frame. With cal, the fix is recomputed
    with that calibration (same math as capture_run_wave.calculate_fix)."""
    log = cr.pick_solve_log(run_dir, None)
    meta = json.load(open(os.path.join(run_dir, "run_meta.json")))
    m_case = np.array(meta["M_case"], dtype=float)
    temps = load_temps(run_dir)
    rows = []
    for d in cr.load_jsonl(log):
        s = d.get("solution") or {}
        if d.get("type") != "frame" or s.get("RA") is None or not d.get("gravity_median"):
            continue
        if cal is None and not d.get("fix"):
            continue
        rows.append(d)
    if not rows:
        return None
    from astropy.time import Time
    t = Time([d["time_iso_mid"] for d in rows], format="iso", scale="utc")
    tu = t.unix
    keep = tu >= tu.min() + skip_min * 60.0
    rows = [d for d, k in zip(rows, keep) if k]
    tu = tu[keep]
    if not rows:
        return None
    era = Time(tu, format="unix").earth_rotation_angle("tio").deg
    c2i = cr.c2i_matrices(tu)
    out = []
    for d, ti, e, q in zip(rows, tu, era, c2i):
        s = d["solution"]
        att = cr.attitude_matrix(s["RA"], s["Dec"], s["Roll"])
        g = np.asarray(d["gravity_median"], dtype=float)
        if cal is None:
            lat, lon = d["fix"]["latitude"], d["fix"]["longitude"]
        else:
            A, b, c, t0 = cal
            v = A @ (m_case @ g) + b
            temp = d.get("imu_temp_c")
            if temp is None and temps is not None:
                temp = float(np.interp(ti, *temps))
            if c is not None and temp is not None:
                v = v + c * (temp - t0)
            lat, lon = cr.fix_latlon(att, v / np.linalg.norm(v), e, q)
        out.append({"t": ti, "lat": lat, "lon": lon, "era": e, "c2i": q,
                    "bore": att[0], "g": g / np.linalg.norm(g)})
    return out


def heading_deg(f):
    """Azimuth of the camera boresight (deg, clockwise from north) at the
    frame's own position: boresight ICRS -> CIRS -> local east/north/up."""
    v = f["c2i"] @ f["bore"]
    lst = math.radians(f["era"] + f["lon"])
    la = math.radians(f["lat"])
    up = np.array([math.cos(la) * math.cos(lst), math.cos(la) * math.sin(lst), math.sin(la)])
    east = np.array([-math.sin(lst), math.cos(lst), 0.0])
    north = np.cross(up, east)
    return math.degrees(math.atan2(v @ east, v @ north))


def load_gps(run_dir):
    p = os.path.join(run_dir, "gps.jsonl")
    if not os.path.exists(p):
        return None, None
    recs = cr.load_jsonl(p)
    note = next((r.get("source") for r in recs if r.get("type") == "meta" and r.get("source")), None)
    fixes = sorted((r for r in recs if r.get("type") == "fix" and r.get("lat") is not None),
                   key=lambda r: r["t_gps"])
    if not fixes:
        return None, note
    return (np.array([r["t_gps"] for r in fixes]), np.array([r["lat"] for r in fixes]),
            np.array([r["lon"] for r in fixes])), note


# --- the fit -------------------------------------------------------------------

def to_plane(lat, lon, lat0, lon0):
    """(lat, lon) -> N + iE in nmi around (lat0, lon0)."""
    return complex((lat - lat0) * 60.0,
                   ((lon - lon0 + 180.0) % 360.0 - 180.0) * 60.0 * math.cos(math.radians(lat0)))


def from_plane(z, lat0, lon0):
    return lat0 + z.real / 60.0, lon0 + z.imag / (60.0 * math.cos(math.radians(lat0)))


def fit(bins):
    """Least squares on the minute bins: y = P + exp(i h) e0. Returns P, e0,
    the 1-sigma radial uncertainty of P (nmi) and the residuals."""
    y = np.array([b["y"] for b in bins])
    h = np.radians([b["h"] for b in bins])
    w = np.sqrt([b["n"] for b in bins])                   # weight: frames per bin
    M = np.c_[np.ones(len(y)), np.exp(1j * h)]
    (P, e0), *_ = np.linalg.lstsq(M * w[:, None], y * w, rcond=None)
    res = y - M @ np.array([P, e0])
    # real-valued covariance of (Re P, Im P, Re e0, Im e0)
    J = np.zeros((2 * len(y), 4))
    J[0::2, 0] = 1; J[1::2, 1] = 1
    J[0::2, 2] = np.cos(h); J[0::2, 3] = -np.sin(h)
    J[1::2, 2] = np.sin(h); J[1::2, 3] = np.cos(h)
    W = np.repeat(w ** 2, 2)                              # same row order as J
    r = np.empty(2 * len(y)); r[0::2] = res.real; r[1::2] = res.imag
    dof = max(1, 2 * len(y) - 4)
    s2 = float(np.sum(W * r ** 2) / dof)
    try:
        cov = s2 * np.linalg.inv(J.T @ (J * W[:, None]))
        sigma_P = math.sqrt(cov[0, 0] + cov[1, 1])
    except np.linalg.LinAlgError:
        sigma_P = float("inf")
    return P, e0, sigma_P, res


def heading_coverage(hs):
    """Largest gap between the sorted headings -> covered arc = 360 - gap."""
    s = np.sort(np.mod(hs, 360.0))
    gaps = np.diff(np.r_[s, s[0] + 360.0])
    return 360.0 - gaps.max()


# --- main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (or a directory of them)")
    ap.add_argument("--cal", default=None,
                    help="recompute every fix with this calibration json")
    ap.add_argument("--position", default=None, metavar="LAT,LON",
                    help="known position, for EVALUATION only (never used in the fit)")
    ap.add_argument("--dr-from-gps", action="store_true",
                    help="under way: take the vessel's motion from each run's gps.jsonl "
                         "(relative displacement only)")
    ap.add_argument("--skip-min", type=float, default=0.0,
                    help="minutes dropped at the start of each run (warm-up)")
    args = ap.parse_args()

    cal = load_cal(args.cal) if args.cal else None
    runs = []
    for r in cr.expand_run_dirs(args.runs):
        fr = load_frames(r, cal, args.skip_min)
        if not fr or len(fr) < 20:
            print(f"{os.path.basename(r)}: fewer than 20 solved frames -- skipped")
            continue
        gps, note = (load_gps(r) if args.dr_from_gps else (None, None))
        if args.dr_from_gps and gps is None:
            print(f"{os.path.basename(r)}: --dr-from-gps but no gps.jsonl -- skipped")
            continue
        if note and "EXTRAPOLATED" in note.upper():
            print(f"{os.path.basename(r)}: NOTE gps.jsonl is extrapolated, not real GPS ({note})")
        runs.append({"name": os.path.basename(os.path.normpath(r)), "frames": fr, "gps": gps})
    if not runs:
        raise SystemExit("no usable run")

    # one tilt only: compare each run's mean gravity direction (IMU frame)
    ups = []
    for run in runs:
        u = np.mean([f["g"] for f in run["frames"]], axis=0)
        ups.append(u / np.linalg.norm(u))
    ref = np.median(np.array(ups), axis=0); ref /= np.linalg.norm(ref)
    kept = []
    for run, u in zip(runs, ups):
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(u @ ref)))))
        run["tilt"] = tilt
        if tilt > TILT_TOL_DEG:
            print(f"{run['name']}: tilted {tilt:.1f} deg from the other runs -- excluded "
                  f"(the model assumes one tilt)")
        else:
            kept.append(run)
    runs = kept
    if not runs:
        raise SystemExit("no run left at a common tilt")

    # reference point for the local plane, reference time for the motion
    all_f = [f for run in runs for f in run["frames"]]
    lat0 = float(np.median([f["lat"] for f in all_f]))
    lon0 = float(np.median([f["lon"] for f in all_f]))
    t_ref = min(f["t"] for f in all_f)
    gps_ref = None
    if args.dr_from_gps:
        g0 = runs[0]["gps"]
        gps_ref = (float(np.interp(t_ref, g0[0], g0[1])), float(np.interp(t_ref, g0[0], g0[2])))

    # per frame: fix minus vessel displacement, heading; outliers; minute bins
    bins = []
    for run in runs:
        ys, hs, ts = [], [], []
        for f in run["frames"]:
            y = to_plane(f["lat"], f["lon"], lat0, lon0)
            if args.dr_from_gps:
                g = run["gps"]
                if not g[0][0] <= f["t"] <= g[0][-1]:
                    continue
                pos = (float(np.interp(f["t"], g[0], g[1])), float(np.interp(f["t"], g[0], g[2])))
                y -= to_plane(pos[0], pos[1], lat0, lon0) - to_plane(gps_ref[0], gps_ref[1], lat0, lon0)
            ys.append(y); hs.append(heading_deg(f)); ts.append(f["t"])
        ys, hs, ts = np.array(ys), np.array(hs), np.array(ts)
        if len(ys) < 20:
            continue
        med = np.median(ys.real) + 1j * np.median(ys.imag)
        dev = np.abs(ys - med); mad = np.median(dev) + 1e-9
        ok = dev < OUTLIER_MAD * mad
        ys, hs, ts = ys[ok], hs[ok], ts[ok]
        run["y"] = ys.mean()
        run["h"] = math.degrees(np.angle(np.exp(1j * np.radians(hs)).mean()))
        run["n"] = len(ys)
        k = ((ts - ts.min()) // BIN_S).astype(int)
        for kk in np.unique(k):
            s = k == kk
            if s.sum() >= 5:
                bins.append({"run": run["name"], "y": ys[s].mean(), "n": int(s.sum()),
                             "h": math.degrees(np.angle(np.exp(1j * np.radians(hs[s])).mean()))})
    runs = [r for r in runs if "y" in r]
    if len(bins) < 3:
        raise SystemExit("not enough data after filtering")

    cover = heading_coverage(np.array([b["h"] for b in bins]))
    P, e0, sigma, res = fit(bins)
    lat_est, lon_est = from_plane(P, lat0, lon0)

    truth = None
    if args.position:
        truth = tuple(float(v) for v in args.position.split(","))
    elif args.dr_from_gps:
        truth = gps_ref

    print(f"\n{len(runs)} runs, {len(bins)} minute-bins, headings cover {cover:.0f} deg of the circle")
    if cover < 60:
        print("WARNING: headings span less than 60 deg -- the instrument error and the "
              "position can NOT be separated reliably. Turn the instrument further.")
    print("\n run                     heading  tilt  frames   single-run fix error   model residual")
    for run in runs:
        rb = [b for b in bins if b["run"] == run["name"]]
        m = P + np.exp(1j * math.radians(run["h"])) * e0
        single = ""
        if truth:
            zt = to_plane(*truth, lat0, lon0)
            single = f"{abs(run['y'] - zt):6.2f} nmi"
        print(f" {run['name']:24s} {run['h']:6.1f}  {run['tilt']:4.1f}  {run['n']:6d}   {single:>14s}        "
              f"{abs(run['y'] - m):.2f} nmi")
    # consistency: runs at (nearly) the same heading must show the same error;
    # if they disagree, something moved inside the instrument between them
    # (shock, handling) and the model's "one fixed error" no longer holds
    for i, a in enumerate(runs):
        for bb in runs[i + 1:]:
            dh = abs((a["h"] - bb["h"] + 180.0) % 360.0 - 180.0)
            if dh <= SAME_HEADING_DEG and abs(a["y"] - bb["y"]) > SAME_HEADING_TOL_NMI:
                print(f"WARNING: {a['name']} and {bb['name']} have the same heading "
                      f"({dh:.0f} deg apart) but differ by {abs(a['y'] - bb['y']):.2f} nmi: the "
                      f"instrument probably changed between them (shock, handling) -- keep "
                      f"only the runs after the last change")
    print(f"\ninstrument-fixed error: {abs(e0):.2f} nmi")
    when = " (at the first frame's time)" if args.dr_from_gps else ""
    print(f"POSITION{when}: {lat_est:.5f}, {lon_est:.5f}   +/- {sigma:.2f} nmi (1 sigma)")
    if truth:
        zt = to_plane(*truth, lat0, lon0)
        d = P - zt
        print(f"vs truth {truth[0]:.6f}, {truth[1]:.6f}: error {abs(d):.2f} nmi (N {d.real:+.2f}, E {d.imag:+.2f})")

    if len(runs) >= 3:
        print("\nstability (leaving each run out):")
        for run in runs:
            sub = [b for b in bins if b["run"] != run["name"]]
            if len(sub) < 3 or heading_coverage(np.array([b["h"] for b in sub])) < 30:
                print(f"   without {run['name']}: headings too narrow, not solvable")
                continue
            Pi, _, si, _ = fit(sub)
            msg = f"position moves {abs(Pi - P):.2f} nmi"
            if truth:
                msg += f", error vs truth {abs(Pi - to_plane(*truth, lat0, lon0)):.2f} nmi"
            print(f"   without {run['name']}: {msg}")


if __name__ == "__main__":
    main()
