#!/usr/bin/env python3
"""Discipline the Pi's clock from the GPS (or NTP), then write the DS3231 RTC.

Runs on the Pi. The Pi Zero has no built-in RTC, so with no network it boots
with a stale clock; a DS3231 keeps time across power-off, but it still has to be
*set* from a good reference at least once, and topped up so it does not drift.
This script is that step. In the pipeline, clock error is a longitude error
(~0.25 nmi per second), so getting to sub-second before a run matters.

Order of preference for the time reference:

  1. GPS  -- read a few NMEA fixes from the serial receiver (reusing the exact
     parsing from nmea_logger.py), take the median system-minus-GPS offset, and
     step the system clock by it. GPS time is effectively exact; the residual
     is serial/read latency, which the median over several fixes keeps small.
  2. NTP  -- if no GPS is available but the system clock is already NTP-synced
     (timedatectl reports NTPSynchronized=yes), trust it as-is.

Either way it then writes the disciplined system time into the DS3231 with
`hwclock -w`, so the next cold, network-less boot starts accurate.

Typical use (needs root to set the clock and write the RTC):

    sudo python set_clock_from_gps.py --port /dev/ttyAMA0 --baud 9600

Dry run first to just see the offset without touching anything:

    python set_clock_from_gps.py --port /dev/ttyAMA0 --baud 9600 --dry-run

--max-step-s guards against a wild GPS/parse glitch stepping the clock by years;
raise it the first time you set a badly-wrong clock (that first step can be huge
and legitimate -- pass e.g. --max-step-s 1e12 for the initial set).
"""

import argparse
import statistics
import subprocess
import sys
import time

# Reuse the validated NMEA parsing so there is one source of truth for it.
from nmea_logger import checksum_ok, fix_from_sentence


def timedatectl_show():
    """Parse `timedatectl show` into a dict (empty on failure)."""
    try:
        out = subprocess.run(["timedatectl", "show"], capture_output=True,
                             text=True, timeout=5).stdout
        return dict(line.split("=", 1)
                    for line in out.strip().splitlines() if "=" in line)
    except Exception:
        return {}


def collect_gps_offsets(port_name, baud, n_fixes, timeout_s):
    """Read the serial GPS until n_fixes valid fixes (or timeout); return the
    list of (system_time - gps_time) offsets, one per fix.

    The offset is sampled right after the sentence is read, so it is dominated
    by serial + read latency (tens of ms at typical baud), which the caller
    reduces further by taking the median across fixes."""
    try:
        import serial
    except ImportError:
        raise SystemExit("pyserial is missing: pip install pyserial")

    try:
        port = serial.Serial(port_name, baud, timeout=2)
    except Exception as e:
        raise SystemExit(f"Cannot open GPS on {port_name} @ {baud}: {e}")

    offsets = []
    state = {}
    t_deadline = time.time() + timeout_s
    print(f"Reading GPS on {port_name} @ {baud} "
          f"(want {n_fixes} fixes, up to {timeout_s:.0f} s)...")
    try:
        while len(offsets) < n_fixes and time.time() < t_deadline:
            line = port.readline().decode("ascii", "ignore").strip()
            if not line or not checksum_ok(line):
                continue
            try:
                fix = fix_from_sentence(line, state)
            except (ValueError, IndexError):
                continue
            if fix is None or fix.get("t_gps") is None:
                continue
            offset = time.time() - fix["t_gps"]
            offsets.append(offset)
            print(f"  fix {len(offsets)}/{n_fixes}: {fix['iso_gps']}  "
                  f"system-minus-GPS {offset:+.3f} s")
    finally:
        port.close()
    return offsets


def step_system_clock(target_unix):
    """Set the system clock to target_unix (UTC). Needs root."""
    # date -u -s @<epoch> is the most portable; timedatectl refuses while NTP
    # is active, so we stop automatic sync first if it is on.
    iso = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(target_unix))
    frac = target_unix - int(target_unix)
    subprocess.run(["date", "-u", "-s", f"{iso}.{int(frac*1e6):06d}"],
                   check=True, capture_output=True, text=True)


def write_rtc():
    """System clock -> DS3231 (hwclock -w). Needs root and a bound RTC."""
    subprocess.run(["hwclock", "-w"], check=True, capture_output=True, text=True)


def read_rtc():
    """DS3231 -> printable string (hwclock -r), or an error note."""
    try:
        out = subprocess.run(["hwclock", "-r"], check=True,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip()
    except Exception as e:
        return f"(could not read RTC: {e})"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyAMA0",
                        help="serial GPS device (default /dev/ttyAMA0)")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--n-fixes", type=int, default=5,
                        help="GPS fixes to median over (default 5)")
    parser.add_argument("--gps-timeout-s", type=float, default=30.0)
    parser.add_argument("--max-step-s", type=float, default=86400.0,
                        help="refuse to step the clock by more than this "
                             "(safety; raise for the very first set)")
    parser.add_argument("--no-rtc-write", action="store_true",
                        help="set the system clock but do not write the RTC")
    parser.add_argument("--dry-run", action="store_true",
                        help="measure and print only; change nothing")
    args = parser.parse_args()

    # --- 1. Try GPS ---------------------------------------------------------
    offsets = collect_gps_offsets(args.port, args.baud, args.n_fixes,
                                  args.gps_timeout_s)

    source = None
    if offsets:
        offset = statistics.median(offsets)
        spread = (max(offsets) - min(offsets)) if len(offsets) > 1 else 0.0
        print(f"\nGPS: median system-minus-GPS offset {offset:+.3f} s "
              f"across {len(offsets)} fixes (spread {spread:.3f} s).")
        target = time.time() - offset   # what the system clock should read now
        source = "gps"
    else:
        # --- 2. Fall back to NTP -------------------------------------------
        td = timedatectl_show()
        if td.get("NTPSynchronized") == "yes":
            print("\nNo GPS fixes, but the system clock is NTP-synchronized "
                  "(timedatectl NTPSynchronized=yes) -- trusting it as-is.")
            offset, target, source = 0.0, time.time(), "ntp"
        else:
            sys.exit("\nNo GPS fixes and no NTP sync -- cannot set the clock. "
                     "Check the GPS wiring/--port/--baud, or bring the Pi online "
                     "briefly so NTP can sync, then rerun.")

    # --- 3. Apply -----------------------------------------------------------
    if source == "gps" and abs(offset) > args.max_step_s:
        sys.exit(f"Refusing to step the clock by {offset:+.1f} s "
                 f"(> --max-step-s {args.max_step_s:g}). If the clock really is "
                 f"this far off (e.g. first-ever set), rerun with a larger "
                 f"--max-step-s.")

    print(f"RTC currently reads: {read_rtc()}")

    if args.dry_run:
        print("\n--dry-run: nothing changed. Would have set the system clock "
              f"from {source.upper()} and "
              + ("skipped the RTC write." if args.no_rtc_write
                 else "written it to the DS3231."))
        return

    try:
        if source == "gps":
            step_system_clock(target)
            print(f"System clock set from GPS (stepped {offset:+.3f} s).")
        else:
            print("System clock left as NTP had it.")
        if not args.no_rtc_write:
            write_rtc()
            print(f"RTC written. DS3231 now reads: {read_rtc()}")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").strip()
        if "not permitted" in err.lower() or "must be superuser" in err.lower():
            sys.exit("Permission denied -- run with sudo.")
        sys.exit(f"Command failed: {' '.join(e.cmd)}\n{err}")
    except FileNotFoundError as e:
        sys.exit(f"Missing tool: {e}. Install util-linux (hwclock) / coreutils.")


if __name__ == "__main__":
    main()
