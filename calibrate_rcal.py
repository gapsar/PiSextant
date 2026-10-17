#!/usr/bin/env python3
"""Camera-IMU alignment (rotation only) and shared helpers -- runs on the laptop.

The other calibration tools import its helpers (zenith model, solve-log
reading, rotation fit). For the calibration itself use calibrate_affine.py.

Every solved frame gives one vector pair: where the star solve + GPS position
+ time say the zenith is in the camera frame, versus where the IMU gravity
(rotated by the coarse M_case) says it is. The fixed rotation R_cal that best
aligns all pairs at once is Wahba's problem, solved in closed form by SVD --
that rotation IS the camera-IMU mounting misalignment behind the ~300 nmi
constant fix offset.

    python calibrate_rcal.py runs/<id> [runs/<id2> ...]

Each run folder needs capture.jsonl, gps.jsonl and a solve log (the live one
is used by default, --solve-log picks another by filename). False solves are
rejected by iterative trimming on the fit residuals. Output: the R_cal matrix
(printed + data/r_cal.json) and before/after fix errors per run.

One device pointing only constrains 2 of the 3 rotation degrees of freedom
(rotation about the gravity axis is invisible to gravity measurements): the
result is still exactly right for frames with similar tilt, but for a fully
constrained R_cal include runs with the device tilted differently. The
conditioning report at the end says how well constrained the fit is.
"""

import argparse
import glob
import json
import math
import os
import sys

import erfa
import numpy as np
from astropy.time import Time

EARTH_RADIUS_NMI = 3440.065

# Residual trimming: drop pairs beyond 3x the median residual (false solves
# sit degrees away, genuine pairs within IMU noise), floored so a very quiet
# run doesn't trim healthy pairs.
TRIM_FACTOR = 3.0
TRIM_FLOOR_DEG = 0.2


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def haversine_nmi(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_NMI * math.asin(math.sqrt(a))


def attitude_matrix(ra_deg, dec_deg, roll_deg):
    """ICRS -> camera rotation, identical to calculate_fix in capture_run_wave.py:
    rows are boresight, roll-rotated east, roll-rotated north."""
    ra, dec, roll = np.radians([ra_deg, dec_deg, roll_deg])
    cos_ra, sin_ra = np.cos(ra), np.sin(ra)
    cos_dec, sin_dec = np.cos(dec), np.sin(dec)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    boresight = np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])
    east = np.array([-sin_ra, cos_ra, 0.0])
    north = np.array([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec])
    return np.array([boresight,
                     cos_roll * east + sin_roll * north,
                     -sin_roll * east + cos_roll * north])


def c2i_matrices(t_unix):
    """ICRS -> CIRS (equator of date): frame bias + precession + nutation,
    IAU 2006/2000A (erfa.c2i06a). Ignoring it puts the zenith ~0.15 deg
    (~9 nmi) off.
    One 3x3 matrix per time (vectorized: one erfa call per run)."""
    tt = Time(np.atleast_1d(t_unix), format="unix").tt
    return erfa.c2i06a(tt.jd1, tt.jd2)


def zenith_icrs(lat_deg, lon_deg, era_deg, c2i):
    """Inverse of the fix math: the zenith unit vector in ICRS above a known
    position at a known time (local angle = longitude + ERA, in CIRS, then
    rotated back to ICRS with the c2i matrix of that time)."""
    lat = math.radians(lat_deg)
    lst = math.radians(lon_deg + era_deg)
    z_cirs = np.array([math.cos(lat) * math.cos(lst),
                       math.cos(lat) * math.sin(lst),
                       math.sin(lat)])
    return c2i.T @ z_cirs


def fix_latlon(r_icrs_to_cam, v_cam, era_deg, c2i):
    """Fix from attitude + camera-frame up vector (calculate_fix without the
    input handling), for the before/after comparison."""
    zenith = c2i @ (r_icrs_to_cam.T @ v_cam)
    lat = math.degrees(math.asin(min(1.0, max(-1.0, zenith[2]))))
    lst = math.degrees(math.atan2(zenith[1], zenith[0])) % 360
    lon = (lst - era_deg + 180) % 360 - 180
    return lat, lon


def fit_rotation(true_vecs, meas_vecs):
    """Rotation R minimizing sum |t_i - R m_i|^2 over unit vector pairs.

    Wahba's SVD solution when the measured directions actually spread; when
    they all sit in one tight cone the twist about the cone axis is not in
    the data and the SVD fills it with an arbitrary (often huge) rotation --
    there we return the minimal rotation aligning the mean directions
    instead: identical residuals, zero spurious twist. Returns
    (R, profile singular values, method name)."""
    b = true_vecs.T @ meas_vecs
    u, s, vt = np.linalg.svd(b)
    if s[1] > 1e-3 * s[0]:
        d = np.sign(np.linalg.det(u) * np.linalg.det(vt))
        return u @ np.diag([1.0, 1.0, d]) @ vt, s, "wahba"
    m_bar = meas_vecs.mean(axis=0)
    t_bar = true_vecs.mean(axis=0)
    m_bar, t_bar = m_bar / np.linalg.norm(m_bar), t_bar / np.linalg.norm(t_bar)
    cross = np.cross(m_bar, t_bar)
    sin, cos = np.linalg.norm(cross), float(m_bar @ t_bar)
    if sin < 1e-12:
        return np.eye(3), s, "mean-align"
    k = cross / sin
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    theta = math.atan2(sin, cos)
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * kx @ kx, \
        s, "mean-align"


def expand_run_dirs(paths):
    """Accept run folders directly or directories of run folders (runs/)."""
    run_dirs = []
    for p in paths:
        if os.path.exists(os.path.join(p, "capture.jsonl")):
            run_dirs.append(p)
        else:
            subs = sorted(d for d in glob.glob(os.path.join(p, "*"))
                          if os.path.exists(os.path.join(d, "capture.jsonl")))
            if not subs:
                print(f"  {p}: no run folders found -- skipped")
            run_dirs.extend(subs)
    return run_dirs


def pick_solve_log(run_dir, name_filter):
    paths = sorted(glob.glob(os.path.join(run_dir, "solves", "*.jsonl")))
    if name_filter:
        paths = [p for p in paths if name_filter in os.path.basename(p)]
    live = [p for p in paths if os.path.basename(p).startswith("live-")]
    return (live or paths or [None])[0]


def load_wave_gravity(run_dir, window_s):
    """De-rotated gravity averaging (the protocol's wave-compensation method,
    applied offline from imu.jsonl). Boat rotation is shared by camera and
    IMU so it is not an error, but wave *acceleration* leaks into the gravity
    estimate. Rotate every gravity sample into the world frame with the game
    quaternion, average over a window centered on the exposure (wave
    acceleration is near zero-mean over whole periods there), and rotate the
    average back with the attitude at the exposure itself. Returns a lookup
    gravity(t) -> vector in the IMU frame, or None off the ends of the log."""
    from scipy.spatial.transform import Rotation   # only needed with --wave-s

    recs = [r for r in load_jsonl(os.path.join(run_dir, "imu.jsonl"))
            if "quat_ijkw" in r]
    t = np.array([r["t"] for r in recs])
    rot = Rotation.from_quat([r["quat_ijkw"] for r in recs])   # body -> world
    h = rot.apply(np.array([r["gravity"] for r in recs]))      # world frame

    def gravity(t_mid, min_samples=20):
        lo, hi = np.searchsorted(t, (t_mid - window_s / 2, t_mid + window_s / 2))
        if hi - lo < min_samples:
            return None
        nearest = lo + int(np.argmin(np.abs(t[lo:hi] - t_mid)))
        return rot[nearest].apply(h[lo:hi].mean(axis=0), inverse=True)

    return gravity


def collect_pairs(run_dir, name_filter, wave_s=0.0, fixed_pos=None,
                  skip_min=0.0):
    """One record per usable frame: measured up (M_case @ gravity, unit) and
    true up (attitude @ zenith-from-GPS) in the camera frame, plus what the
    before/after fix comparison needs.

    fixed_pos=(lat, lon) replaces gps.jsonl with one constant position --
    only for runs captured while NOT moving (on land or at anchor).
    skip_min drops the first minutes of the run (thermal warm-up, or the
    ~45 min dynamic-calibration settle of unfrozen runs)."""
    log_path = pick_solve_log(run_dir, name_filter)
    if log_path is None:
        print(f"  {run_dir}: no solve log -- skipped")
        return []
    t_mid_by_id = {c["frame_id"]: c["t_exposure_mid"]
                   for c in load_jsonl(os.path.join(run_dir, "capture.jsonl"))}
    if fixed_pos is not None:
        t_all = [t for t in t_mid_by_id.values() if t is not None]
        if not t_all:
            print(f"  {run_dir}: empty capture.jsonl -- skipped")
            return []
        gps_t = np.array([min(t_all) - 60.0, max(t_all) + 60.0])
        gps_lat = np.array([fixed_pos[0]] * 2)
        gps_lon = np.array([fixed_pos[1]] * 2)
    else:
        gps_path = os.path.join(run_dir, "gps.jsonl")
        if not os.path.exists(gps_path):
            print(f"  {run_dir}: no gps.jsonl -- skipped "
                  f"(stationary run? pass --position LAT,LON)")
            return []
        gps = [r for r in load_jsonl(gps_path) if r.get("type") == "fix"]
        gps.sort(key=lambda r: r["t_gps"])
        gps_t = np.array([r["t_gps"] for r in gps])
        gps_lat = np.array([r["lat"] for r in gps])
        gps_lon = np.array([r["lon"] for r in gps])

    lines = load_jsonl(log_path)
    meta = lines[0] if lines and lines[0].get("type") == "meta" else {}
    m_case = np.array(meta.get("M_case"), dtype=float)
    logged_r_cal = np.array(meta.get("R_cal", np.eye(3).tolist()))
    if not np.allclose(logged_r_cal, np.eye(3)):
        print(f"  {run_dir}: NOTE -- run was recorded with a non-identity "
              f"R_cal; calibrating from raw gravity anyway (result replaces "
              f"it, it does not stack)")

    frames = [l for l in lines if l.get("type") == "frame"
              and l.get("status") == "success"
              and l.get("solution", {}).get("RA") is not None
              and l.get("gravity_used") is not None
              and t_mid_by_id.get(l.get("frame_id")) is not None]
    frames = [f for f in frames
              if gps_t[0] <= t_mid_by_id[f["frame_id"]] <= gps_t[-1]]
    if skip_min > 0 and frames:
        t_run0 = min(t for t in t_mid_by_id.values() if t is not None)
        n_before = len(frames)
        frames = [f for f in frames
                  if t_mid_by_id[f["frame_id"]] >= t_run0 + skip_min * 60.0]
        if n_before != len(frames):
            print(f"  {run_dir}: skipped the first {skip_min:g} min "
                  f"({n_before - len(frames)} frames)")
    if not frames:
        print(f"  {run_dir}: no solved frames inside the GPS log -- skipped")
        return []

    # One astropy call for the whole run (per-frame ERA is the slow part)
    t_mid = np.array([t_mid_by_id[f["frame_id"]] for f in frames])
    era = Time(t_mid, format="unix").earth_rotation_angle("tio").deg
    c2i = c2i_matrices(t_mid)

    wave = load_wave_gravity(run_dir, wave_s) if wave_s > 0 else None
    n_fallback = 0

    run_id = os.path.basename(run_dir.rstrip("/"))
    pairs = []
    for f, t, e, q in zip(frames, t_mid, era, c2i):
        sol = f["solution"]
        att = attitude_matrix(sol["RA"], sol["Dec"], sol["Roll"])
        gravity = f["gravity_used"]
        if wave is not None:
            g_w = wave(t)
            if g_w is None:
                n_fallback += 1
            else:
                gravity = g_w
        m = m_case @ np.asarray(gravity, dtype=float)
        norm = np.linalg.norm(m)
        if norm < 1e-6:
            continue
        lat = float(np.interp(t, gps_t, gps_lat))
        lon = float(np.interp(t, gps_t, gps_lon))
        pairs.append({
            "run": run_id, "frame_id": f["frame_id"],
            "meas": m / norm,                       # up per IMU, camera frame
            "true": att @ zenith_icrs(lat, lon, e, q), # up per stars + GPS
            "c2i": q,
            "att": att, "era": float(e), "gps": (lat, lon),
        })
    wave_note = (f", wave-averaged gravity {wave_s:g} s"
                 + (f" ({n_fallback} frames fell back to the exposure median)"
                    if n_fallback else "")) if wave else ""
    print(f"  {run_id}: {len(pairs)} pairs from "
          f"{os.path.basename(log_path)}{wave_note}")
    return pairs


def fix_errors_nmi(pairs, r_cal):
    return np.array([haversine_nmi(*p["gps"],
                                   *fix_latlon(p["att"], r_cal @ p["meas"], p["era"], p["c2i"]))
                     for p in pairs])


def mean_fix_error_nmi(pairs, r_cal):
    """Error of the average fix position -- the headline per-run number."""
    fixes = [fix_latlon(p["att"], r_cal @ p["meas"], p["era"], p["c2i"]) for p in pairs]
    gps = [p["gps"] for p in pairs]
    return haversine_nmi(float(np.mean([g[0] for g in gps])),
                         float(np.mean([g[1] for g in gps])),
                         float(np.mean([f[0] for f in fixes])),
                         float(np.mean([f[1] for f in fixes])))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+",
                        help="runs/<run_id> folders, or a directory of them "
                             "(one R_cal is fitted from all runs jointly -- "
                             "only combine runs of the same mounting session)")
    parser.add_argument("--solve-log", default=None,
                        help="substring to pick the solve log by filename "
                             "(default: the live log)")
    parser.add_argument("--wave-s", type=float, default=0.0,
                        help="window (seconds) for de-rotated gravity "
                             "averaging from imu.jsonl -- use at sea, ~3-5 "
                             "wave periods (default 0: the per-exposure "
                             "gravity median as recorded)")
    parser.add_argument("--position", default=None, metavar="LAT,LON",
                        help="fixed truth position in decimal degrees for "
                             "runs captured while NOT moving (land/anchor); "
                             "replaces gps.jsonl")
    parser.add_argument("--skip-min", type=float, default=0.0,
                        help="minutes to drop at the start of each run "
                             "(thermal warm-up ~10-15; unfrozen "
                             "dynamic-calibration settle ~45)")
    parser.add_argument("--out", default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "data", "r_cal.json"))
    args = parser.parse_args()

    fixed_pos = None
    if args.position is not None:
        try:
            fixed_pos = tuple(float(v) for v in args.position.split(","))
        except ValueError:
            sys.exit("--position must be 'LAT,LON' in decimal degrees")
        print(f"Truth: fixed position ({fixed_pos[0]:+.5f}, {fixed_pos[1]:+.5f})"
              f" -- assumes the device did not move during the runs.")

    print("Collecting vector pairs:")
    run_dirs = expand_run_dirs(args.run_dirs)
    pairs = []
    for run_dir in run_dirs:
        pairs.extend(collect_pairs(run_dir, args.solve_log, args.wave_s,
                                   fixed_pos, args.skip_min))
    if len(pairs) < 10:
        sys.exit(f"Only {len(pairs)} pairs -- not enough to calibrate.")

    meas = np.array([p["meas"] for p in pairs])
    true = np.array([p["true"] for p in pairs])

    # Iterative trimming: fit, drop everything beyond 3x the median residual
    # (recomputed over ALL pairs each round so nothing is dropped for good),
    # refit until the kept set stabilizes.
    keep = np.ones(len(pairs), dtype=bool)
    for _ in range(10):
        r_cal, svals, method = fit_rotation(true[keep], meas[keep])
        cos = np.clip(np.einsum("ij,ij->i", true, meas @ r_cal.T), -1.0, 1.0)
        residuals = np.degrees(np.arccos(cos))
        cut = max(TRIM_FACTOR * float(np.median(residuals[keep])), TRIM_FLOOR_DEG)
        new_keep = residuals <= cut
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    n_used, n_dropped = int(keep.sum()), int((~keep).sum())
    res_kept = residuals[keep]

    # The correction itself
    angle = math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(r_cal) - 1) / 2))))
    axis = np.array([r_cal[2, 1] - r_cal[1, 2],
                     r_cal[0, 2] - r_cal[2, 0],
                     r_cal[1, 0] - r_cal[0, 1]])
    axis = axis / n if (n := np.linalg.norm(axis)) > 1e-9 else axis

    print(f"\nR_cal from {n_used} pairs ({n_dropped} trimmed as outliers), "
          f"method: {method}")
    for row in r_cal:
        print("   [{:+.8f}, {:+.8f}, {:+.8f}]".format(*row))
    print(f"  rotation: {angle:.3f} deg about camera-frame axis "
          "[{:+.3f}, {:+.3f}, {:+.3f}]".format(*axis))
    print(f"  fit residuals (deg): median {np.median(res_kept):.3f}  "
          f"p90 {np.percentile(res_kept, 90):.3f}  max {res_kept.max():.3f}")

    # Conditioning: how much the measured directions actually spread. With
    # everything in one tight cone the rotation about that cone axis comes
    # from noise, not data -- fine for same-tilt use, see the docstring.
    mean_dir = meas[keep].mean(axis=0)
    mean_dir /= np.linalg.norm(mean_dir)
    spread = np.degrees(np.arccos(np.clip(meas[keep] @ mean_dir, -1.0, 1.0)))
    print(f"  conditioning: gravity direction spread p90 "
          f"{np.percentile(spread, 90):.2f} deg (max {spread.max():.2f}), "
          f"profile singular values {svals[0]:.0f} / {svals[1]:.1f} / {svals[2]:.1f}")
    if method == "mean-align":
        print("  NOTE: single-pointing data -- the twist about the gravity "
              "axis is not observable and was set to zero (smallest rotation "
              "consistent with the data). Exact for frames with similar tilt; "
              "add runs with different device tilts to pin down all 3 axes.")

    # Before/after on the calibration data itself
    print("\nFix error vs GPS, identity -> R_cal (kept pairs):")
    identity = np.eye(3)
    for run_id in dict.fromkeys(p["run"] for p in pairs):
        rp = [p for p, k in zip(pairs, keep) if k and p["run"] == run_id]
        if not rp:
            print(f"  {run_id}: ALL pairs trimmed -- this run does not fit the "
                  f"calibration of the others (device moved between runs?); "
                  f"it did not contribute to R_cal")
            continue
        before = fix_errors_nmi(rp, identity)
        after = fix_errors_nmi(rp, r_cal)
        print(f"  {run_id}: median {np.median(before):7.2f} -> {np.median(after):6.2f} nmi   "
              f"mean-fix-position {mean_fix_error_nmi(rp, identity):7.2f} -> "
              f"{mean_fix_error_nmi(rp, r_cal):6.2f} nmi   ({len(rp)} fixes)")

    out = {
        "r_cal": r_cal.tolist(),
        "method": method,
        "created": Time.now().iso,
        "runs": [os.path.basename(d.rstrip("/")) for d in run_dirs],
        "n_pairs_used": n_used,
        "n_pairs_trimmed": n_dropped,
        "rotation_deg": angle,
        "rotation_axis_camera": axis.tolist(),
        "residual_median_deg": float(np.median(res_kept)),
        "gravity_spread_p90_deg": float(np.percentile(spread, 90)),
        "wave_s": args.wave_s,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
