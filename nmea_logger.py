#!/usr/bin/env python3
"""GPS ground-truth logger -- runs on the laptop.

Reads NMEA sentences from a serial GPS, and writes two files per session:

    gps_<start_time>.jsonl   parsed fixes, with both GPS time and laptop time
                             so clock offsets between machines can be
                             diagnosed later
    gps_<start_time>.nmea    every raw sentence, untouched, just in case

A fix record is emitted per RMC sentence when the receiver sends them, or --
for receivers that don't (ours sends GGA/VTG/ZDA/DTM) -- per GGA sentence,
taking the date from ZDA and speed/course from VTG.

Start it before the Pi run begins, stop it (Ctrl-C) after the run ends, then
copy the .jsonl into the run folder as  runs/<run_id>/gps.jsonl  for the
analyzer.

Usage:
    python nmea_logger.py --port /dev/ttyUSB0 --baud 4800
"""

import argparse
import json
import time
from datetime import datetime, timedelta, timezone


def checksum_ok(sentence):
    """NMEA checksum: XOR of everything between '$' and '*'."""
    if not sentence.startswith("$") or "*" not in sentence:
        return False
    body, _, given = sentence[1:].partition("*")
    calc = 0
    for char in body:
        calc ^= ord(char)
    try:
        return calc == int(given[:2], 16)
    except ValueError:
        return False


def dm_to_deg(dm, hemisphere):
    """NMEA ddmm.mmmm (or dddmm.mmmm) to signed decimal degrees."""
    if not dm:
        return None
    value = float(dm)
    degrees = int(value / 100)
    minutes = value - degrees * 100
    result = degrees + minutes / 60.0
    return -result if hemisphere in ("S", "W") else result


def gps_time(hhmmss, year, month, day):
    """Combine NMEA hhmmss.ss time-of-day with a date into a UTC datetime."""
    seconds = float(hhmmss[4:])
    return datetime(year, month, day, int(hhmmss[0:2]), int(hhmmss[2:4]),
                    int(seconds), int((seconds % 1) * 1e6), tzinfo=timezone.utc)


def parse_rmc(fields):
    """$xxRMC: time, status, lat, NS, lon, EW, sog(kn), cog, date."""
    if len(fields) < 10 or fields[2] != "A":   # A = valid fix
        return None
    hhmmss, ddmmyy = fields[1], fields[9]
    t = gps_time(hhmmss, 2000 + int(ddmmyy[4:6]), int(ddmmyy[2:4]), int(ddmmyy[0:2]))
    return {
        "t_gps": t.timestamp(),
        "iso_gps": t.isoformat(),
        "lat": dm_to_deg(fields[3], fields[4]),
        "lon": dm_to_deg(fields[5], fields[6]),
        "sog_kn": float(fields[7]) if fields[7] else None,
        "cog_deg": float(fields[8]) if fields[8] else None,
    }


def parse_gga(fields):
    """$xxGGA: time, lat, NS, lon, EW, quality, n_sats, hdop, alt."""
    if len(fields) < 10:
        return None
    return {
        "hhmmss": fields[1],
        "lat": dm_to_deg(fields[2], fields[3]),
        "lon": dm_to_deg(fields[4], fields[5]),
        "quality": int(fields[6]) if fields[6] else 0,
        "n_sats": int(fields[7]) if fields[7] else None,
        "hdop": float(fields[8]) if fields[8] else None,
        "alt_m": float(fields[9]) if fields[9] else None,
    }


def parse_zda(fields):
    """$xxZDA: time, day, month, year -- the date source when there is no RMC."""
    if len(fields) < 5 or not (fields[2] and fields[3] and fields[4]):
        return None
    return int(fields[4]), int(fields[3]), int(fields[2])   # (year, month, day)


def parse_vtg(fields):
    """$xxVTG: cog true, T, cog magnetic, M, sog kn, N, sog km/h, K."""
    if len(fields) < 6:
        return None
    return {
        "cog_deg": float(fields[1]) if fields[1] else None,
        "sog_kn": float(fields[5]) if fields[5] else None,
    }


def fix_from_sentence(line, state):
    """Feed one checksum-valid sentence; return a fix record or None.

    state is a dict carried between calls: last GGA extras, last VTG
    speed/course, last ZDA date, and the timestamp of the last emitted fix
    (so RMC and GGA in the same cycle don't produce duplicate records).
    """
    fields = line.split("*")[0].split(",")
    sentence_type = fields[0][3:6]   # skip the talker id ($GP/$GN/...)

    if sentence_type == "ZDA":
        parsed = parse_zda(fields)
        if parsed:
            state["zda_ymd"] = parsed
        return None

    if sentence_type == "VTG":
        parsed = parse_vtg(fields)
        if parsed:
            state["vtg"] = parsed
        return None

    if sentence_type == "GGA":
        gga = parse_gga(fields)
        if gga is None:
            return None
        state["gga"] = {k: gga[k] for k in ("quality", "n_sats", "hdop", "alt_m")}
        if gga["quality"] == 0 or gga["lat"] is None:
            return None
        ymd = state.get("zda_ymd")
        if ymd is None:   # no ZDA seen yet: laptop UTC date (fine away from midnight)
            now = datetime.now(timezone.utc)
            ymd = (now.year, now.month, now.day)
        t = gps_time(gga["hhmmss"], *ymd)
        # ZDA arrives after the GGA it timestamps, so right after midnight UTC
        # the date is one day stale -- snap whole days using the laptop clock
        offset = t.timestamp() - time.time()
        if abs(offset) > 12 * 3600:
            t -= timedelta(days=round(offset / 86400))
        fix = {
            "t_gps": t.timestamp(),
            "iso_gps": t.isoformat(),
            "lat": gga["lat"],
            "lon": gga["lon"],
            "sog_kn": state.get("vtg", {}).get("sog_kn"),
            "cog_deg": state.get("vtg", {}).get("cog_deg"),
            **state["gga"],
        }
    elif sentence_type == "RMC":
        rmc = parse_rmc(fields)
        if rmc is None or rmc["lat"] is None:
            return None
        fix = {**rmc, **state.get("gga", {})}
    else:
        return None

    if fix["t_gps"] == state.get("last_emit_t"):   # RMC + GGA of the same second
        return None
    state["last_emit_t"] = fix["t_gps"]
    return fix


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=4800)
    parser.add_argument("--out-dir", default="gps_logs")
    parser.add_argument("--forward", default=None, metavar="URL",
                        help="also POST each fix to the Pi's web UI, e.g. "
                             "http://192.168.1.50:8080 (local network only; "
                             "logging to disk continues unchanged)")
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        raise SystemExit("pyserial is missing: pip install pyserial")

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    jsonl_path = os.path.join(args.out_dir, f"gps_{stamp}.jsonl")
    raw_path = os.path.join(args.out_dir, f"gps_{stamp}.nmea")

    port = serial.Serial(args.port, args.baud, timeout=2)
    jsonl = open(jsonl_path, "a")
    raw = open(raw_path, "a")
    jsonl.write(json.dumps({"type": "meta", "port": args.port, "baud": args.baud,
                            "started_iso": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(f"Logging {args.port} @ {args.baud} -> {jsonl_path}")
    print("Ctrl-C to stop. Copy the .jsonl into the run folder as gps.jsonl.")

    n_fixes = 0
    n_bad = 0
    state = {}
    offsets = []   # laptop clock minus GPS clock, for diagnostics
    try:
        while True:
            line = port.readline().decode("ascii", "ignore").strip()
            if not line or not checksum_ok(line):
                continue
            raw.write(line + "\n")
            try:
                fix = fix_from_sentence(line, state)
            except (ValueError, IndexError):
                # corrupted field that still passed the checksum (seen live:
                # a stray NUL byte inside a ZDA date) -- never kill the log
                n_bad += 1
                continue
            if fix is None:
                continue
            t_laptop = time.time()
            record = {"type": "fix", "t_laptop": round(t_laptop, 3), **fix}
            jsonl.write(json.dumps(record) + "\n")
            jsonl.flush()
            if args.forward:
                # fire-and-forget to the web UI; never let it stall logging
                try:
                    import urllib.request
                    req = urllib.request.Request(
                        args.forward.rstrip("/") + "/api/gps",
                        data=json.dumps(record).encode(),
                        headers={"Content-Type": "application/json"})
                    urllib.request.urlopen(req, timeout=2).read()
                except Exception:
                    pass
            n_fixes += 1
            offsets.append(t_laptop - fix["t_gps"])
            print(f"{fix['iso_gps']}  lat {fix['lat']:+.5f}  lon {fix['lon']:+.5f}  "
                  f"sog {fix['sog_kn'] or 0:.1f} kn  sats {fix.get('n_sats') or '?'}")
    except KeyboardInterrupt:
        pass
    finally:
        port.close()
        jsonl.close()
        raw.close()
        print(f"\nStopped: {n_fixes} fixes -> {jsonl_path}")
        if n_bad:
            print(f"Skipped {n_bad} corrupted sentences that passed the checksum.")
        if offsets:
            print(f"Laptop clock minus GPS clock: mean {sum(offsets) / len(offsets):+.2f} s "
                  f"(includes serial latency; large values mean the laptop clock is off)")


if __name__ == "__main__":
    main()
