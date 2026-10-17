#!/usr/bin/env python3
"""Offline re-solve of a run folder -- runs on the laptop.

Re-runs cedar-detect + tetra3 on every saved frame of a run, recomputes the
fixes from the gravity data already in capture.jsonl, and writes a new solve
log next to the live one:

    runs/<run_id>/solves/replay-<name>.jsonl     (same schema as live)

The fixes use the same model as the live capture (affine accelerometer
calibration with its temperature term, precession-nutation). By default the
calibration recorded in the run is used, so only the detect/solve changes
show up; --cal applies another calibration on purpose. Compare the logs with
sea_eval.py (--solve-log) or multiheading_fix.py.

Usage:
    python replay_solve.py runs/<run_id>                       # replay the run's own config
    python replay_solve.py runs/<run_id> --config configs/candidate.json
    python replay_solve.py runs/<run_id> --config ... --name candidate2
    python replay_solve.py runs/<run_id> --cal configs/r_cal_ism330dhcx.json

Needs cedar-solve installed (import tetra3) with a cedar-detect-server
executable built for this computer (set CEDAR_DETECT_BINARY or --cedar-binary
if it is not in the tetra3 package's bin/ folder).
"""

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import erfa
import numpy as np
from PIL import Image
from astropy.time import Time

import os
import tetra3
from tetra3 import cedar_detect_client

CEDAR_BINARY = os.environ.get(
    "CEDAR_DETECT_BINARY",
    str(Path(tetra3.__file__).resolve().parent / "bin" / "cedar-detect-server"))
SCRIPT_DIR = Path(__file__).resolve().parent


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def sanitize(obj):
    """Convert numpy types so records can be dumped straight to JSON."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# --- Fix computation (same math as capture_run_wave.py) ----------------------

def calculate_fix(ra_deg, dec_deg, roll_deg, gravity_xyz, obs_time, m_case,
                  cal_a, cal_b, cal_c=None, temp_c=None, temp_ref_c=25.0):
    """Single-image celestial fix: camera attitude (star solve) plus the
    gravity vector (IMU, rotated into the camera frame) give the zenith in
    ICRS, hence latitude, and via the Earth rotation angle, longitude."""
    era_deg = obs_time.earth_rotation_angle("tio").deg

    v = cal_a @ (m_case @ np.asarray(gravity_xyz, dtype=float)) + cal_b
    if cal_c is not None and temp_c is not None:
        v = v + cal_c * (temp_c - temp_ref_c)
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        raise ValueError("gravity vector magnitude too close to zero")
    v_cam = v / norm

    ra, dec, roll = np.radians([ra_deg, dec_deg, roll_deg])
    cos_ra, sin_ra = np.cos(ra), np.sin(ra)
    cos_dec, sin_dec = np.cos(dec), np.sin(dec)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)

    boresight = np.array([cos_dec * cos_ra, cos_dec * sin_ra, sin_dec])
    east = np.array([-sin_ra, cos_ra, 0.0])
    north = np.array([-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec])

    r_icrs_to_cam = np.array([boresight,
                              cos_roll * east + sin_roll * north,
                              -sin_roll * east + cos_roll * north])
    zenith_icrs = r_icrs_to_cam.T @ v_cam
    tt = obs_time.tt                      # ICRS -> CIRS: precession-nutation
    zenith = erfa.c2i06a(tt.jd1, tt.jd2) @ zenith_icrs

    latitude = np.degrees(np.arcsin(np.clip(zenith[2], -1.0, 1.0)))
    lst_deg = np.degrees(np.arctan2(zenith[1], zenith[0])) % 360
    longitude = (lst_deg - era_deg + 180) % 360 - 180

    return {"latitude": float(latitude), "longitude": float(longitude),
            "lst_deg": float(lst_deg), "era_deg": float(era_deg)}


# --- Setup --------------------------------------------------------------------

def resolve_database(cfg, override):
    """The config records the database path as it was on the Pi; on the laptop
    the same file lives in pipeline/data/. Try, in order: the --database flag,
    the config's path, data/<same filename> next to this script."""
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    cfg_path = Path(cfg["database"]["path"]).expanduser()
    candidates += [cfg_path, SCRIPT_DIR / "data" / cfg_path.name]
    for c in candidates:
        if c.with_suffix(".npz").exists():
            return c.with_suffix("")   # tetra3 wants the path without extension
    sys.exit(f"Database not found (tried: {', '.join(str(c) for c in candidates)}). "
             f"Use --database, or build it with build_database.py.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="runs/<run_id> folder (needs images/)")
    parser.add_argument("--config", default=None,
                        help="config JSON (default: the config recorded in run_meta.json)")
    parser.add_argument("--name", default=None,
                        help="output name (default: the config's own name) "
                             "-> solves/replay-<name>.jsonl")
    parser.add_argument("--database", default=None,
                        help="database .npz (default: resolved from the config)")
    parser.add_argument("--cal", default=None,
                        help="calibration json (A, b, c, temp_ref_c) from "
                             "calibrate_affine.py / rezero.py (default: the "
                             "calibration recorded in the run)")
    parser.add_argument("--cedar-binary", default=CEDAR_BINARY,
                        help="cedar-detect-server executable for this computer")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    meta = json.loads((run_dir / "run_meta.json").read_text())
    if args.config:
        cfg = json.loads(Path(args.config).read_text())
    else:
        cfg = meta["solver"]["config"]

    name = args.name or cfg.get("name", "unnamed")
    out_path = run_dir / "solves" / f"replay-{name}.jsonl"
    if out_path.exists() and not args.overwrite:
        sys.exit(f"{out_path} already exists -- use --name or --overwrite.")

    # Fixes are recomputed with the same mounting matrix and calibration the
    # run used, so only the detect/solve changes show up in the comparison --
    # unless --cal supplies a new calibration on purpose.
    m_case = np.array(meta["M_case"])
    if args.cal:
        cal = json.loads(Path(args.cal).read_text())
        cal_a, cal_b = np.array(cal["A"]), np.array(cal["b"])
        cal_c = np.array(cal["c"]) if cal.get("c") is not None else None
        temp_ref = float(cal.get("temp_ref_c", 25.0))
    else:
        cal_a = np.array(meta["R_cal"])
        cal_b = np.array(meta.get("cal_b") or [0.0, 0.0, 0.0])
        cal_c = np.array(meta["cal_c"]) if meta.get("cal_c") is not None else None
        temp_ref = float(meta.get("cal_temp_ref_c") or 25.0)
    temps = [r for r in load_jsonl(run_dir / "temp.jsonl")
             if r.get("t") is not None and r.get("temp_raw") is not None] \
        if (run_dir / "temp.jsonl").exists() else []
    temp_t = np.array([r["t"] for r in temps])
    temp_v = np.array([r["temp_raw"] for r in temps])

    frames = load_jsonl(run_dir / "capture.jsonl")
    db = resolve_database(cfg, args.database)
    print(f"Replaying {len(frames)} frames of {run_dir.name} "
          f"with config '{name}', database {db.with_suffix('.npz')}")

    t3 = tetra3.Tetra3(load_database=db)
    solve_kw_names = set(inspect.signature(t3.solve_from_centroids).parameters)
    cedar = cedar_detect_client.CedarDetectClient(binary_path=str(args.cedar_binary))
    detect_cfg = cfg["detect"]
    solve_cfg = cfg["solve"]

    out = open(out_path, "w")
    out.write(json.dumps(sanitize({
        "type": "meta", "mode": "replay", "config": cfg,
        "M_case": m_case.tolist(), "R_cal": cal_a.tolist(),
        "cal_b": cal_b.tolist(),
        "cal_c": cal_c.tolist() if cal_c is not None else None,
        "cal_temp_ref_c": temp_ref, "cal_file": args.cal,
        "database": str(db.with_suffix(".npz")),
        "replayed_run": run_dir.name,
        "created": Time.now().iso,
    })) + "\n")

    n_solved = n_fixes = n_missing = 0
    try:
        for i, cap in enumerate(frames):
            frame_id = cap["frame_id"]
            image_path = run_dir / cap["image"]
            gravity = cap["imu_window"]["gravity_median"]
            n_imu = cap["imu_window"]["n_samples"]
            obs_time = Time(cap["t_exposure_mid"], format="unix")

            if not image_path.exists():   # PNG queue drained on a hard kill
                n_missing += 1
                out.write(json.dumps({
                    "type": "frame", "frame_id": frame_id, "image": cap["image"],
                    "time_iso_mid": obs_time.iso, "status": "file_missing",
                    "n_centroids": 0, "centroids": [], "solution": None,
                    "gravity_used": gravity, "n_imu_samples": n_imu,
                    "fix": None, "fix_error": None, "timings_ms": {},
                }) + "\n")
                continue

            try:
                gray = np.asarray(Image.open(image_path))
            except OSError as e:   # truncated PNG from a dirty shutdown
                n_missing += 1
                out.write(json.dumps({
                    "type": "frame", "frame_id": frame_id, "image": cap["image"],
                    "time_iso_mid": obs_time.iso, "status": "file_missing",
                    "error": str(e), "n_centroids": 0, "centroids": [],
                    "solution": None, "gravity_used": gravity,
                    "n_imu_samples": n_imu, "fix": None, "fix_error": None,
                    "timings_ms": {},
                }) + "\n")
                continue

            t0 = time.perf_counter()
            centroids = cedar.extract_centroids(
                gray,
                sigma=float(detect_cfg["sigma"]),
                use_binned=(int(detect_cfg.get("binning", 1)) == 2),
                detect_hot_pixels=True,
            )
            t_detect_ms = (time.perf_counter() - t0) * 1000

            solution = None
            t_solve_ms = 0.0
            if len(centroids) == 0:
                status = "no_stars"
            else:
                kw = {
                    "size": gray.shape,
                    "fov_estimate": solve_cfg.get("fov_estimate"),
                    "fov_max_error": solve_cfg.get("fov_max_error"),
                    "pattern_checking_stars": solve_cfg.get("pattern_checking_stars", 8),
                    "match_radius": solve_cfg.get("match_radius", 0.01),
                    "match_threshold": solve_cfg.get("match_threshold", 1e-3),
                    "solve_timeout": solve_cfg.get("solve_timeout", 3000),
                    "distortion": solve_cfg.get("distortion", 0),
                    "return_catalog": False,
                }
                kw = {k: v for k, v in kw.items() if k in solve_kw_names}
                max_centroids = int(detect_cfg.get("max_centroids", 30))
                t0 = time.perf_counter()
                solution = t3.solve_from_centroids(centroids[:max_centroids], **kw)
                t_solve_ms = (time.perf_counter() - t0) * 1000
                solved = solution is not None and solution.get("RA") is not None
                status = "success" if solved else "no_solution"

            fix, fix_error = None, None
            if status == "success" and gravity is not None:
                try:
                    temp = (float(np.interp(cap["t_exposure_mid"], temp_t, temp_v))
                            if len(temp_t) else None)
                    fix = calculate_fix(solution["RA"], solution["Dec"],
                                        solution["Roll"], gravity, obs_time,
                                        m_case, cal_a, cal_b, cal_c, temp, temp_ref)
                    n_fixes += 1
                except Exception as e:
                    fix_error = str(e)
            if status == "success":
                n_solved += 1

            out.write(json.dumps(sanitize({
                "type": "frame",
                "frame_id": frame_id,
                "image": cap["image"],
                "time_iso_mid": obs_time.iso,
                "status": status,
                "n_centroids": len(centroids),
                "centroids": centroids,
                "solution": solution,
                "gravity_used": gravity,
                "n_imu_samples": n_imu,
                "fix": fix,
                "fix_error": fix_error,
                "timings_ms": {"detect": round(t_detect_ms, 1),
                               "solve": round(t_solve_ms, 1)},
            }), default=str) + "\n")

            if (i + 1) % 100 == 0 or i + 1 == len(frames):
                print(f"  {i + 1}/{len(frames)} frames, "
                      f"{n_solved} solved ({n_solved / (i + 1):.1%})")
    finally:
        out.close()
        del cedar   # shuts down its cedar-detect-server subprocess

    print(f"Done: {n_solved} solved, {n_fixes} fixes"
          + (f", {n_missing} missing images" if n_missing else "")
          + f" -> {out_path}")


if __name__ == "__main__":
    main()
