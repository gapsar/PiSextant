#!/usr/bin/env python3
"""PiSextant web UI -- runs on the Pi, pure stdlib, no pip installs.

Serves a single-page UI (Leaflet map, vendored locally) that can:
  - start/stop solve_worker.py and capture_run_wave.py in their venvs,
  - stream the live run to the browser (fixes, smoothed fix, gravity,
    solve results, temperature) by TAILING the run's log files -- the
    capture pipeline itself is never touched,
  - receive GPS truth from the laptop (nmea_logger.py --forward), or a
    manually entered fixed position,
  - export KML traces (GPS track, smoothed fixes, raw fixes),
  - serve OpenStreetMap tiles from a local cache, fetch missing ones
    when online, and bulk-download a chosen region for offline use.

Usage:
    python3 server.py [--port 8080] [--pipeline-dir ..]

Autostart: see systemd/pisextant-webui.service in the repo.
"""

import argparse
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(HERE)
STATIC_DIR = os.path.join(HERE, "static")
TILES_DIR = os.path.join(HERE, "tiles")
CONFIG_PATH = os.path.join(HERE, "webui_config.json")

DEFAULT_CONFIG = {
    "camera_python": "python3",
    "tetra3_python": "python3",
    "solver_config": "configs/solver.json",
    "tile_url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "tile_user_agent": "PiSextant-webui/1.0 (celestial navigation; "
                       "set your contact in webui_config.json)",
    "max_tiles_per_job": 20000,
}


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            cfg.update(json.load(open(CONFIG_PATH)))
        except Exception as e:
            print(f"webui_config.json unreadable ({e}), using defaults")
    for key in ("camera_python", "tetra3_python"):   # allow ~/venv/bin/python
        cfg[key] = os.path.expanduser(cfg[key])
    return cfg


def haversine_nmi(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) *
         math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 3440.065 * math.asin(math.sqrt(a))


def load_tol(path, limit=None):
    """JSON records of a .jsonl file; with limit, stop after that many
    (run files can be hundreds of MB on a 512 MB Pi)."""
    out = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                if limit is not None and len(out) >= limit:
                    break
    except OSError:
        pass
    return out


# --- Event broadcast (SSE) --------------------------------------------------

class Broadcaster:
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()

    def subscribe(self):
        q = queue.Queue(maxsize=500)
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def push(self, event_type, data):
        msg = {"type": event_type, **data}
        with self.lock:
            clients = list(self.clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                pass  # slow client: drop, SSE clients recover via /api/run_data


BUS = Broadcaster()


# --- Live state -------------------------------------------------------------

class State:
    """Everything the UI shows, rebuilt cheaply from tailed files."""

    def __init__(self):
        self.lock = threading.Lock()
        self.gps = []          # (t, lat, lon) from forwarded NMEA
        self.fixed_position = None
        self.fixes = []        # (t, lat, lon)
        self.smoothed = []     # (t, lat, lon)
        self.temps = []        # (t, temp_raw)
        self.watched_run = None
        self.run_t0 = None     # first exposure time of the watched run
        self.run_t_last = None

    def add_gps(self, t, lat, lon):
        with self.lock:
            self.gps.append((t, lat, lon))
            if len(self.gps) > 50000:
                self.gps = self.gps[-40000:]

    def truth(self):
        """Current truth (lat, lon) or None: the fixed position, else the
        last forwarded GPS fix if it is fresh (received < 60 s ago)."""
        if self.fixed_position:
            return tuple(self.fixed_position)
        if self.gps and time.time() - self.gps[-1][0] < 60:
            return self.gps[-1][1], self.gps[-1][2]
        return None

    def reset_run_series(self):
        with self.lock:
            self.fixes = []
            self.smoothed = []
            self.temps = []
            self.run_t0 = None
            self.run_t_last = None


STATE = State()


# --- Run tailer -------------------------------------------------------------

class RunTailer(threading.Thread):
    """Follows the active run folder: new lines in the solve log, imu.jsonl
    and temp.jsonl become SSE events. Never writes anything."""

    def __init__(self, runs_dir):
        super().__init__(daemon=True)
        self.runs_dir = runs_dir
        self.current = None
        self.switch_to = "latest"
        self.stop_event = threading.Event()

    def newest_run(self):
        try:
            runs = [d for d in os.listdir(self.runs_dir)
                    if os.path.isdir(os.path.join(self.runs_dir, d))]
            return max(runs) if runs else None
        except OSError:
            return None

    def pick_solve_log(self, run_dir):
        d = os.path.join(run_dir, "solves")
        try:
            logs = sorted(f for f in os.listdir(d) if f.endswith(".jsonl"))
        except OSError:
            return None
        live = [f for f in logs if f.startswith("live-")]
        return os.path.join(d, (live or logs)[-1]) if logs else None

    def read_lines(self, f, limit=500):
        """Complete JSON records appended since the last read. A line still
        being written (no trailing newline yet) is left for the next pass,
        so a long record caught mid-write is never lost."""
        out = []
        for _ in range(limit):
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.endswith("\n"):
                f.seek(pos)
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return out

    def switch_run(self, name):
        """Load the history of a newly watched run. The solve log and
        temp.jsonl are read through the same handles that are then followed
        live, so nothing written in between is skipped; imu.jsonl is only
        followed live (it is display-only and large). The history is read in
        blocks of 500 lines: a 3 h solve log is ~500 MB, far too big to hold
        in memory at once."""
        files = {}
        STATE.reset_run_series()
        STATE.watched_run = name
        BUS.push("run", {"run": name})
        if not name:
            return files
        run_dir = os.path.join(self.runs_dir, name)
        log = self.pick_solve_log(run_dir)
        if log:
            f = open(log)
            while True:
                pos = f.tell()
                for rec in self.read_lines(f):
                    self.handle_solve(rec, live=False)
                if f.tell() == pos:
                    break
            files[log] = (f, self.handle_solve_line)
        temp_path = os.path.join(run_dir, "temp.jsonl")
        if os.path.exists(temp_path):
            f = open(temp_path)
            for rec in self.read_lines(f, limit=10**7):
                if rec.get("t") is not None:
                    STATE.temps.append((rec["t"], rec.get("temp_raw")))
            files[temp_path] = (f, self.handle_temp_line)
        # run start: first frame's exposure time from capture.jsonl
        for rec in load_tol(os.path.join(run_dir, "capture.jsonl"), limit=1):
            STATE.run_t0 = rec.get("t_exposure_mid")
        if STATE.temps:
            STATE.run_t_last = STATE.temps[-1][0]
        BUS.push("history_ready", {"run": name})
        return files

    def run(self):
        files = {}   # path -> (open file positioned at its tail, handler)
        imu_decim = 0
        while not self.stop_event.is_set():
            target = self.switch_to
            name = self.newest_run() if target == "latest" else target
            if name != self.current:
                for f, _handler in files.values():
                    f.close()
                self.current = name
                try:
                    files = self.switch_run(name)
                except OSError:
                    files = {}
            if not name:
                time.sleep(2)
                continue
            run_dir = os.path.join(self.runs_dir, name)
            # files that appear after the switch (a capture that just started)
            # are read from their start: every line in them is new
            for path, handler in (
                    (self.pick_solve_log(run_dir), self.handle_solve_line),
                    (os.path.join(run_dir, "temp.jsonl"), self.handle_temp_line),
                    (os.path.join(run_dir, "imu.jsonl"), self.handle_imu_line)):
                if path and path not in files and os.path.exists(path):
                    try:
                        f = open(path)
                    except OSError:
                        continue
                    if handler == self.handle_imu_line:
                        f.seek(0, 2)
                    files[path] = (f, handler)
            for f, handler in list(files.values()):
                for rec in self.read_lines(f):
                    if handler == self.handle_imu_line:
                        imu_decim += 1
                        if imu_decim % 5:     # ~5 Hz to the browser
                            continue
                    handler(rec)
            time.sleep(0.5)

    def handle_solve_line(self, rec):
        self.handle_solve(rec, live=True)

    def handle_solve(self, rec, live):
        if rec.get("type") != "frame":
            return
        t = None
        fix = rec.get("fix")
        sm = rec.get("fix_smoothed")
        if fix:
            STATE.fixes.append((rec.get("time_iso_mid"),
                                fix["latitude"], fix["longitude"]))
        if sm:
            STATE.smoothed.append((rec.get("time_iso_mid"),
                                   sm["latitude"], sm["longitude"]))
        if live:
            truth = STATE.truth()
            err = err_sm = None
            if truth:
                if fix:
                    err = round(haversine_nmi(truth[0], truth[1],
                                              fix["latitude"],
                                              fix["longitude"]), 2)
                if sm:
                    err_sm = round(haversine_nmi(truth[0], truth[1],
                                                 sm["latitude"],
                                                 sm["longitude"]), 2)
            BUS.push("solve", {
                "frame_id": rec.get("frame_id"),
                "status": rec.get("status"),
                "n_centroids": rec.get("n_centroids"),
                "fix": ([fix["latitude"], fix["longitude"]] if fix else None),
                "smoothed": ([sm["latitude"], sm["longitude"]] if sm else None),
                "smoothed_n": (sm or {}).get("n_fixes"),
                "wave": (rec.get("wave") or {}).get("applied"),
                "err_nmi": err, "err_smoothed_nmi": err_sm,
            })

    def handle_imu_line(self, rec):
        g = rec.get("gravity")
        if g:
            if rec.get("t") is not None:
                STATE.run_t_last = rec["t"]
            BUS.push("gravity", {"t": rec.get("t"), "g": g})

    def handle_temp_line(self, rec):
        if rec.get("t") is not None:
            STATE.temps.append((rec["t"], rec.get("temp_raw")))
            STATE.run_t_last = rec["t"]
        BUS.push("temp", {"t": rec.get("t"), "temp_raw": rec.get("temp_raw")})


# --- Managed processes ------------------------------------------------------

class Proc:
    def __init__(self, name):
        self.name = name
        self.p = None
        self.log = []          # ring buffer of last lines
        self.lock = threading.Lock()

    def running(self):
        return self.p is not None and self.p.poll() is None

    def start(self, cmd, cwd):
        if self.running():
            return False, "already running"
        try:
            self.p = subprocess.Popen(
                cmd, cwd=cwd, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
                stdin=subprocess.DEVNULL)
        except Exception as e:
            return False, str(e)
        threading.Thread(target=self._pump, daemon=True).start()
        BUS.push("proc", {"name": self.name, "running": True})
        return True, "started"

    def _pump(self):
        for line in self.p.stdout:
            with self.lock:
                self.log.append(line.rstrip()[:400])
                if len(self.log) > 200:
                    self.log = self.log[-150:]
            BUS.push("proc_log", {"name": self.name, "line": line.rstrip()[:400]})
        BUS.push("proc", {"name": self.name, "running": False,
                          "returncode": self.p.poll()})

    def stop(self, timeout_s=15):
        """SIGTERM, then SIGKILL after timeout_s. Both scripts finalize their
        logs on SIGTERM; the capture may first need to write the frames still
        queued (slow PNG encoding on the Zero), hence its longer timeout."""
        if not self.running():
            return False, "not running"
        self.p.terminate()
        try:
            self.p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.p.kill()
            return True, f"killed after {timeout_s} s (logs may be incomplete)"
        return True, "stopped"


PROCS = {"solver": Proc("solver"), "capture": Proc("capture")}


# --- Tile cache + downloader ------------------------------------------------

class TileJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.done = 0
        self.errors = 0
        self.active = False
        self.message = ""

    def status(self):
        with self.lock:
            return {"active": self.active, "total": self.total,
                    "done": self.done, "errors": self.errors,
                    "message": self.message}


TILE_JOB = TileJob()


def tile_path(z, x, y):
    return os.path.join(TILES_DIR, str(z), str(x), f"{y}.png")


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def _valid_tile(data, ctype):
    """A real map tile is a PNG/JPEG of reasonable size. OSM's 'Access
    blocked' response comes back as HTTP 200 with an image, so status
    alone doesn't catch it -- but it is tiny. Reject anything that isn't
    a plausible image, so a blocked/error body is never cached."""
    if not data or len(data) < 500:          # OSM block/error tiles are tiny
        return False
    if ctype and not ctype.startswith("image/"):
        return False
    return data.startswith(PNG_MAGIC) or data.startswith(JPEG_MAGIC)


def fetch_tile(cfg, z, x, y):
    url = cfg["tile_url"].format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={
        "User-Agent": cfg["tile_user_agent"]})
    resp = urllib.request.urlopen(req, timeout=20)
    data = resp.read()
    ctype = resp.headers.get("Content-Type", "")
    if not _valid_tile(data, ctype):
        # blocked / rate-limited / error page dressed as 200 -- do NOT cache
        raise ValueError(f"not a map tile ({ctype or 'no content-type'}, "
                         f"{len(data)} bytes) -- source blocked? see "
                         f"webui_config.json")
    path = tile_path(z, x, y)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


def purge_bad_tiles():
    """Delete every cached file that isn't a valid image (e.g. OSM's
    'Access blocked' tiles saved before the fetch check existed)."""
    removed = kept = 0
    for root, _dirs, filenames in os.walk(TILES_DIR):
        for fn in filenames:
            if not fn.endswith(".png"):
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, "rb") as f:
                    head = f.read(16)
                size = os.path.getsize(p)
            except OSError:
                continue
            if head.startswith(PNG_MAGIC) and size >= 500:
                kept += 1
            else:
                try:
                    os.remove(p)
                    removed += 1
                except OSError:
                    pass
    return removed, kept


def bbox_tiles(zmin, zmax, lat1, lon1, lat2, lon2):
    """All (z,x,y) covering the bbox for each zoom level."""
    out = []
    lat1, lat2 = max(lat1, lat2), min(lat1, lat2)   # lat1 north
    for z in range(zmin, zmax + 1):
        n = 2 ** z
        def xt(lon): return int((lon + 180.0) / 360.0 * n)
        def yt(lat):
            r = math.radians(lat)
            return int((1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0 * n)
        x1, x2 = sorted((xt(lon1), xt(lon2)))
        y1, y2 = sorted((yt(lat1), yt(lat2)))
        for x in range(max(0, x1), min(n - 1, x2) + 1):
            for y in range(max(0, y1), min(n - 1, y2) + 1):
                out.append((z, x, y))
    return out


def run_tile_job(cfg, tiles):
    with TILE_JOB.lock:
        TILE_JOB.total = len(tiles)
        TILE_JOB.done = 0
        TILE_JOB.errors = 0
        TILE_JOB.active = True
        TILE_JOB.message = ""
    consecutive = 0
    for (z, x, y) in tiles:
        if os.path.exists(tile_path(z, x, y)):
            with TILE_JOB.lock:
                TILE_JOB.done += 1
            continue
        try:
            fetch_tile(cfg, z, x, y)
            consecutive = 0
        except Exception as e:
            consecutive += 1
            with TILE_JOB.lock:
                TILE_JOB.errors += 1
                TILE_JOB.message = str(e)[:200]
            if consecutive >= 15:
                # the server is refusing us (blocked / offline): stop
                # hammering it instead of burning through the whole job
                with TILE_JOB.lock:
                    TILE_JOB.message = (
                        f"aborted after {consecutive} consecutive errors "
                        f"(last: {str(e)[:120]}) -- blocked or offline? See "
                        f"the tile-source notes in webui_config.json")
                    TILE_JOB.active = False
                return
        with TILE_JOB.lock:
            TILE_JOB.done += 1
        time.sleep(0.7)   # be polite to volunteer-run tile servers
    with TILE_JOB.lock:
        TILE_JOB.active = False


# --- KML --------------------------------------------------------------------

def kml_doc(name, coords, color):
    """coords: [(lat, lon)] -> a single LineString KML."""
    coord_str = "\n".join(f"{lon:.6f},{lat:.6f},0" for (_, lat, lon) in coords)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document>
<name>{name}</name>
<Style id="s"><LineStyle><color>{color}</color><width>3</width></LineStyle></Style>
<Placemark><name>{name}</name><styleUrl>#s</styleUrl>
<LineString><tessellate>1</tessellate><coordinates>
{coord_str}
</coordinates></LineString></Placemark>
</Document></kml>"""


# --- Session checklist -----------------------------------------------------

def build_checklist(meta, ntp, clock):
    """Green-light checks for the ACTIVE run (ISM330DHCX): clock synced,
    affine calibration loaded, sensor temperature stable (the ISM's offsets
    move with temperature, ~10 min after power-on)."""
    if not meta:
        return {"items": []}
    if clock == "rtc":
        items = [{"name": "clock from RTC (no NTP)", "ok": "warn",
                  "detail": "good to ~1 s/week (1 s = 0.25 nmi of longitude); "
                            "run 'sudo hwclock -w' once when NTP is synced"}]
    else:
        items = [{"name": "clock NTP-synced", "ok": ntp,
                  "detail": "longitude needs true time"}]

    cal = meta.get("cal_kind")
    items.append({"name": "calibration loaded",
                  "ok": cal == "affine",
                  "detail": f"cal_kind={cal}" + ("" if cal == "affine" else
                            " -- copy the affine json to "
                            "configs/r_cal_ism330dhcx.json")})

    temps = STATE.temps
    plateau = None
    detail = "no temperature data yet"
    if temps and temps[-1][0] - temps[0][0] >= 600:
        t_last = temps[-1][0]
        recent = [v for (t, v) in temps if t >= t_last - 600 and v is not None]
        if len(recent) >= 5:
            plateau = (max(recent) - min(recent)) <= 0.5
            detail = (f"{'flat' if plateau else 'still moving'}: "
                      f"{min(recent):.1f}..{max(recent):.1f} C over last 10 min")
    else:
        plateau = False
        detail = "warming up (<10 min of data)"
    items.append({"name": "temperature stable", "ok": plateau, "detail": detail})

    wave = meta.get("wave_s")
    if wave:
        items.append({"name": "wave compensation off", "ok": False,
                      "detail": f"wave_s={wave}: gyro offset uncorrected, "
                                f"fixes will be wrong"})
    return {"items": items}


# --- Utilities --------------------------------------------------------------

def clock_source(ntp):
    """'ntp' when synced; 'rtc' when not synced but the DS3231 is there (it
    set the time at boot); 'none' when neither (the time can be wrong)."""
    if ntp:
        return "ntp"
    return "rtc" if os.path.exists("/dev/rtc0") else "none"


def clock_sync_status():
    try:
        out = subprocess.run(["timedatectl", "show", "--property",
                              "NTPSynchronized", "--value"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() == "yes"
    except Exception:
        return None


def disk_free_gb(path):
    try:
        u = shutil.disk_usage(path)
        return round(u.free / 1e9, 2)
    except Exception:
        return None


def decimate(series, n_max=2000):
    if len(series) <= n_max:
        return series
    step = len(series) / n_max
    return [series[int(i * step)] for i in range(n_max)]


# --- HTTP handler -----------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "PiSextantUI/1.0"
    cfg = None
    runs_dir = None
    tailer = None

    def log_message(self, fmt, *args):
        pass   # quiet

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_body_json(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    # ---- GET ----
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            return self.serve_static("index.html")
        if path.startswith("/static/"):
            return self.serve_static(path[len("/static/"):])
        if path.startswith("/tiles/"):
            return self.serve_tile(path)
        if path == "/api/stream":
            return self.serve_sse()
        if path == "/api/status":
            return self.serve_status()
        if path == "/api/runs":
            return self.serve_runs()
        if path == "/api/run_data":
            return self.serve_run_data()
        if path == "/api/tiles/status":
            return self.send_json(TILE_JOB.status())
        if path.startswith("/api/kml/"):
            return self.serve_kml(path.rsplit("/", 1)[-1])
        self.send_json({"error": "not found"}, 404)

    def serve_static(self, rel):
        rel = os.path.normpath(rel)
        if rel.startswith(".."):
            return self.send_json({"error": "bad path"}, 400)
        path = os.path.join(STATIC_DIR, rel)
        if not os.path.isfile(path):
            return self.send_json({"error": "not found"}, 404)
        ctype = {"html": "text/html", "js": "application/javascript",
                 "css": "text/css", "png": "image/png",
                 "svg": "image/svg+xml"}.get(rel.rsplit(".", 1)[-1],
                                             "application/octet-stream")
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_tile(self, path):
        m = re.match(r"^/tiles/(\d+)/(\d+)/(\d+)\.png$", path)
        if not m:
            return self.send_json({"error": "bad tile"}, 400)
        z, x, y = map(int, m.groups())
        p = tile_path(z, x, y)
        data = None
        if os.path.exists(p):
            with open(p, "rb") as f:
                data = f.read()
        else:
            try:
                data = fetch_tile(self.cfg, z, x, y)   # online: fetch+cache
            except Exception:
                pass
        if data is None:
            return self.send_json({"error": "tile unavailable offline"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(data)

    def serve_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        q = BUS.subscribe()
        try:
            while True:
                try:
                    msg = q.get(timeout=15)
                    payload = f"data: {json.dumps(msg)}\n\n"
                except queue.Empty:
                    payload = ": keepalive\n\n"
                self.wfile.write(payload.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            BUS.unsubscribe(q)

    def serve_status(self):
        run = STATE.watched_run
        meta = {}
        if run:
            mp = os.path.join(self.runs_dir, run, "run_meta.json")
            try:
                m = json.load(open(mp))
                meta = {"cal_kind": m.get("cal_kind"),
                        "wave_s": m.get("wave_compensation", {}).get("window_s"),
                        "smooth_s": m.get("smoothed_fix", {}).get("window_s")}
            except Exception:
                pass
        ntp = clock_sync_status()
        clock = clock_source(ntp)
        self.send_json({
            "run": run, "run_meta": meta,
            "procs": {k: v.running() for k, v in PROCS.items()},
            "ntp_synchronized": ntp,
            "clock_source": clock,
            "disk_free_gb": disk_free_gb(self.runs_dir),
            "gps_points": len(STATE.gps),
            "fixed_position": STATE.fixed_position,
            "tile_job": TILE_JOB.status(),
            "checklist": build_checklist(meta, ntp, clock),
        })

    def serve_runs(self):
        out = []
        try:
            for d in sorted(os.listdir(self.runs_dir), reverse=True)[:50]:
                p = os.path.join(self.runs_dir, d)
                if os.path.isdir(p):
                    out.append({"run": d,
                                "has_meta": os.path.exists(
                                    os.path.join(p, "run_meta.json"))})
        except OSError:
            pass
        self.send_json({"runs": out})

    def serve_run_data(self):
        with STATE.lock:
            payload = {
                "run": STATE.watched_run,
                "fixes": decimate(STATE.fixes),
                "smoothed": decimate(STATE.smoothed),
                "gps": decimate(STATE.gps),
                "temps": decimate(STATE.temps, 600),
                "fixed_position": STATE.fixed_position,
            }
        self.send_json(payload)

    def serve_kml(self, which):
        with STATE.lock:
            series = {"gps": (STATE.gps, "ff00a5ff"),
                      "smoothed": (STATE.smoothed, "ffff7800"),
                      "fixes": (STATE.fixes, "ff4080ff")}.get(which)
        if series is None:
            return self.send_json({"error": "unknown kml"}, 404)
        coords, color = series
        if not coords:
            return self.send_json({"error": f"no {which} data yet"}, 404)
        body = kml_doc(f"pisextant-{which}", coords, color).encode()
        self.send_response(200)
        self.send_header("Content-Type",
                         "application/vnd.google-earth.kml+xml")
        self.send_header("Content-Disposition",
                         f"attachment; filename=pisextant-{which}.kml")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- POST ----
    def do_POST(self):
        path = self.path.split("?")[0]
        body = self.read_body_json()
        if path == "/api/gps":
            return self.post_gps(body)
        if path == "/api/position":
            return self.post_position(body)
        if path == "/api/watch":
            self.tailer.switch_to = body.get("run") or "latest"
            return self.send_json({"ok": True})
        if path == "/api/proc/start":
            return self.post_proc_start(body)
        if path == "/api/proc/stop":
            name = body.get("name")
            if name not in PROCS:
                return self.send_json({"ok": False, "msg": "bad name"}, 400)
            ok, msg = PROCS[name].stop(timeout_s=90 if name == "capture" else 15)
            return self.send_json({"ok": ok, "msg": msg})
        if path == "/api/tiles/download":
            return self.post_tiles(body)
        if path == "/api/tiles/purge":
            removed, kept = purge_bad_tiles()
            return self.send_json({"ok": True, "removed": removed, "kept": kept})
        self.send_json({"error": "not found"}, 404)

    def post_gps(self, body):
        try:
            t = float(body.get("t_gps") or time.time())
            lat, lon = float(body["lat"]), float(body["lon"])
        except (KeyError, TypeError, ValueError):
            return self.send_json({"ok": False, "msg": "need lat/lon"}, 400)
        STATE.add_gps(t, lat, lon)
        BUS.push("gps", {"t": t, "lat": lat, "lon": lon,
                         "sog_kn": body.get("sog_kn"),
                         "cog_deg": body.get("cog_deg")})
        self.send_json({"ok": True})

    def post_position(self, body):
        if body.get("clear"):
            STATE.fixed_position = None
        else:
            try:
                STATE.fixed_position = [float(body["lat"]), float(body["lon"])]
            except (KeyError, TypeError, ValueError):
                return self.send_json({"ok": False, "msg": "need lat/lon"}, 400)
        BUS.push("fixed_position", {"position": STATE.fixed_position})
        self.send_json({"ok": True})

    def post_proc_start(self, body):
        name = body.get("name")
        cfg = self.cfg
        cmd = []
        if name == "solver":
            cmd = [cfg["tetra3_python"], "solve_worker.py",
                   body.get("config") or cfg["solver_config"]]
            ok, msg = PROCS["solver"].start(cmd, PIPELINE_DIR)
        elif name == "capture":
            cmd = [cfg["camera_python"], "capture_run_wave.py",
                   "--duration-min", str(body.get("duration_min", 15))]
            for flag, key in (("--wave-s", "wave_s"),
                              ("--smooth-s", "smooth_s"),
                              ("--interval-s", "interval_s"),
                              ("--png-every", "png_every")):
                if body.get(key) is not None:
                    cmd += [flag, str(body[key])]
            ok, msg = PROCS["capture"].start(cmd, PIPELINE_DIR)
            if ok:
                self.tailer.switch_to = "latest"
        else:
            ok, msg = False, "bad name"
        self.send_json({"ok": ok, "msg": msg, "cmd": " ".join(cmd) if ok else None})

    def post_tiles(self, body):
        try:
            zmin, zmax = int(body["zmin"]), int(body["zmax"])
            lat1, lon1 = float(body["lat1"]), float(body["lon1"])
            lat2, lon2 = float(body["lat2"]), float(body["lon2"])
        except (KeyError, TypeError, ValueError):
            return self.send_json({"ok": False, "msg": "bad bbox"}, 400)
        tiles = bbox_tiles(zmin, zmax, lat1, lon1, lat2, lon2)
        if not body.get("confirm"):
            return self.send_json({"ok": True, "estimate": len(tiles),
                                   "confirm_required": True})
        if TILE_JOB.status()["active"]:
            return self.send_json({"ok": False, "msg": "a job is running"})
        if len(tiles) > self.cfg["max_tiles_per_job"]:
            return self.send_json(
                {"ok": False,
                 "msg": f"{len(tiles)} tiles exceeds the "
                        f"{self.cfg['max_tiles_per_job']} cap; raise zmin or "
                        f"shrink the box"})
        threading.Thread(target=run_tile_job, args=(self.cfg, tiles),
                         daemon=True).start()
        self.send_json({"ok": True, "started": len(tiles)})


def main():
    global PIPELINE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--pipeline-dir", default=PIPELINE_DIR)
    args = ap.parse_args()

    PIPELINE_DIR = os.path.abspath(args.pipeline_dir)
    runs_dir = os.path.join(PIPELINE_DIR, "runs")
    os.makedirs(TILES_DIR, exist_ok=True)

    cfg = load_config()
    tailer = RunTailer(runs_dir)
    tailer.start()

    Handler.cfg = cfg
    Handler.runs_dir = runs_dir
    Handler.tailer = tailer
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"PiSextant UI on http://0.0.0.0:{args.port}  "
          f"(pipeline: {PIPELINE_DIR})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
