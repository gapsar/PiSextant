#!/usr/bin/env python3
"""Build the tetra3/cedar-solve pattern database -- runs on the laptop.

Builds the database for YOUR camera's field of view (--max-fov, the
horizontal FOV in degrees; slightly larger than the real one is fine). The
other defaults (magnitude 6.5, pattern_max_error 0.005, 22 pattern stars and
75 verification stars per FOV) are the reference build's; its database is
examples/reference-build/database.npz (68.315 deg).

Output is  data/database.npz  next to this script (--save-as to change).
Takes a few minutes. Needs cedar-solve installed (import tetra3) and the
Hipparcos main catalog file hip_main.dat (see the README).

Usage:
    python build_database.py --catalog hip_main.dat --max-fov 68.3
    python build_database.py --catalog hip_main.dat --max-fov 50 --save-as data/fov50.npz
"""

import argparse
import time
from pathlib import Path



def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-fov", type=float, required=True,
                        help="camera horizontal FOV in degrees")
    parser.add_argument("--max-magnitude", type=float, default=6.5,
                        help="faintest stars to include (default: 6.5)")
    parser.add_argument("--pattern-max-error", type=float, default=0.005)
    parser.add_argument("--pattern-stars-per-fov", type=int, default=22,
                        help="pattern star density (cedar-solve lattice_field_oversampling)")
    parser.add_argument("--verification-stars-per-fov", type=int, default=75)
    parser.add_argument("--catalog", required=True,
                        help="path to hip_main.dat (or tyc_main.dat)")
    parser.add_argument("--save-as", default=None,
                        help="output .npz path (default: data/database.npz next to this script)")
    args = parser.parse_args()

    save_as = Path(args.save_as) if args.save_as else Path(__file__).parent / "data" / "database.npz"
    save_as.parent.mkdir(parents=True, exist_ok=True)

    from tetra3 import Tetra3
    t3 = Tetra3(load_database=None)

    print(f"Building: fov {args.max_fov} deg, mag {args.max_magnitude}, "
          f"{args.pattern_stars_per_fov} pattern / {args.verification_stars_per_fov} "
          f"verification stars per fov, pattern_max_error {args.pattern_max_error}")
    t0 = time.time()
    t3.generate_database(
        max_fov=args.max_fov,
        star_catalog=args.catalog,
        star_max_magnitude=args.max_magnitude,
        pattern_max_error=args.pattern_max_error,
        pattern_stars_per_fov=args.pattern_stars_per_fov,
        verification_stars_per_fov=args.verification_stars_per_fov,
        save_as=str(save_as),
    )
    print(f"Done in {time.time() - t0:.0f} s -> {save_as}")

    props = t3.database_properties
    for key in ("max_fov", "star_max_magnitude", "pattern_max_error",
                "pattern_stars_per_fov", "verification_stars_per_fov",
                "num_patterns", "epoch_proper_motion"):
        print(f"  {key}: {props.get(key)}")


if __name__ == "__main__":
    main()
