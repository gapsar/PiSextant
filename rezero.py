#!/usr/bin/env python3
"""Re-zero the calibration after a shock or transport -- runs on the laptop.

A shock can make a board slip slightly: the camera-IMU geometry turns by a
small angle, while the sensor's own properties (gains, offset, temperature
term) stay the same. This tool keeps the existing calibration and measures
only that small rotation, from short runs at a KNOWN position:

    run 1: enclosure flat, ~10 min
    run 2: one side lifted ~30 deg, ~10 min (needed for the 3rd rotation
           component: with the enclosure flat only, a rotation about the
           vertical is invisible)

At sea, give no --position: the truth then comes from each run's gps.jsonl
(ship track). One run with the ship near level is enough for a fixed mount:
the rotation about the vertical stays invisible, but it also does not
change the fix while the ship stays near level.

It writes a new calibration file (A, b and c all rotated) to copy to the Pi
as configs/r_cal_ism330dhcx.json, and prints the error per run before/after.

    python rezero.py runs/<flat> runs/<tilted> --cal configs/r_cal_ism330dhcx.json \\
        --position=48.8566,2.3522 --out data/cal_rezero_<date>.json

(write --position=-21.0,55.3 with '=' when the latitude is negative)

    python rezero.py runs/<sea run with gps.jsonl> --cal configs/r_cal_ism330dhcx.json \
        --out data/cal_rezero_<date>.json
"""

import argparse
import json
import os

import numpy as np

import calibrate_affine as ca
import calibrate_rcal as cr


def fix_errors(pairs, A, b, c, t0):
    """Fix error (north, east) in nmi per frame; the truth of each frame is
    its true zenith p["true"] (fixed position or GPS track)."""
    v = ca.model_vec(pairs, A, b, c, t0)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    out = []
    for p, u in zip(pairs, v):
        la, lo = cr.fix_latlon(p["att"], u, p["era"], p["c2i"])
        la0, lo0 = cr.fix_latlon(p["att"], p["true"], p["era"], p["c2i"])
        out.append(((la - la0) * 60,
                    ((lo - lo0 + 180) % 360 - 180) * 60 * np.cos(np.radians(la0))))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders at the known position")
    ap.add_argument("--cal", required=True, help="current calibration json")
    ap.add_argument("--position", metavar="LAT,LON",
                    help="known fixed position; without it, use each run's gps.jsonl")
    ap.add_argument("--out", required=True, help="new calibration json")
    args = ap.parse_args()
    pos = tuple(float(x) for x in args.position.split(",")) if args.position else None

    cal = json.load(open(args.cal))
    A, b = np.array(cal["A"]), np.array(cal["b"])
    c = np.array(cal["c"]) if cal.get("c") is not None else None
    t0 = float(cal.get("temp_ref_c", 25.0))

    data = {}
    for r in args.runs:
        pairs = ca.trim(ca.collect_raw(r, None, 0.0, pos))
        if c is not None:
            pairs = [p for p in pairs if p["temp"] is not None]
        if len(pairs) < 20:
            print(f"{r}: only {len(pairs)} usable frames -- skipped")
            continue
        data[r] = pairs
    if not data:
        raise SystemExit("no usable run")

    pool = [p for pp in data.values() for p in pp]
    v = ca.model_vec(pool, A, b, c, t0)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    t = np.array([p["true"] for p in pool])
    R, _, _ = cr.fit_rotation(t, v)
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))

    ups = [ca.mean_dir(pp) for pp in data.values()]
    spread = max(np.degrees(np.arccos(np.clip(u1 @ u2, -1, 1)))
                 for u1 in ups for u2 in ups)
    print(f"Rotation measured: {angle:.3f} deg ({angle * 60:.1f} nmi)")
    if spread < 10 and pos is not None:
        print(f"WARNING: the runs differ by only {spread:.1f} deg of tilt -- the rotation "
              "about the vertical is not measured. Add a run with one side lifted ~30 deg.")

    A2, b2 = R @ A, R @ b
    c2 = R @ c if c is not None else None
    print("\n run                          before (nmi)   after (nmi)")
    for r, pairs in data.items():
        e0 = fix_errors(pairs, A, b, c, t0)
        e1 = fix_errors(pairs, A2, b2, c2, t0)
        print(f" {os.path.basename(r):28s} {np.hypot(*e0.mean(0)):9.2f}   {np.hypot(*e1.mean(0)):9.2f}")

    out = dict(cal)
    out.update({"A": A2.tolist(), "b": b2.tolist(),
                "c": c2.tolist() if c2 is not None else None,
                "rezero_runs": [os.path.basename(r) for r in data],
                "rezero_deg": angle,
                "note": f"{cal.get('note', '')}; re-zeroed on {', '.join(os.path.basename(r) for r in data)}"})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out} -- copy it to the Pi as configs/r_cal_ism330dhcx.json")


if __name__ == "__main__":
    main()
