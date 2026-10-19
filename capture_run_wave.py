#!/usr/bin/env python3
"""Capture script -- runs on the Pi, capture venv. Captures frames, reads the
IMU, gets each frame solved by solve_worker.py, computes a position fix and
logs everything into a self-contained run folder.

Main features:

1. Smoothed fix. Each frame also gets "fix_smoothed", the trailing --smooth-s
   (default 60 s) mean of the instant fixes: wave motion and solver noise
   average out, leaving the calibration floor. It is the position to
   navigate by.

2. Optional wave compensation (--wave-s, OFF by default). Each fix then uses
   gravity averaged over a trailing window, every IMU sample carried to the
   exposure's body frame with the gyro-integrated attitude, so wave
   acceleration is near zero-mean over the window. Off by default because
   the gyro's zero-rate offset is not corrected yet: on the reference IMU
   (~0.1-0.4 deg/s, drifting with temperature) a 20 s window faked 1-2 deg
   of rotation (~76 nmi vs ~13 nmi without). "gravity_wave_gyro" is still
   logged per frame whenever --wave-s > 0.

3. Temperature log: temp.jsonl, one record every ~5 s (degrees C).

Logs, per run folder: run_meta.json, images/, imu.jsonl (t, gravity = raw
accel, gyro), temp.jsonl, capture.jsonl, solves/live-<config>.jsonl. The
run is finalized cleanly on Ctrl-C and on SIGTERM (the webui's stop button).

Configuration files, all in configs/ (see configs/README.md; the reference
build's files are in examples/reference-build/):

    instrument.json   camera settings and IMU driver (required)
    m_case.json       camera-to-IMU mounting rotation (fit_mcase.py)
    accel_cal.json    accelerometer calibration (calibrate_affine.py,
                      rezero.py); without it the fixes are uncalibrated

Start solve_worker.py (solver venv) first, then:

    python capture_run_wave.py --duration-min 15
"""

import argparse
import json
import os
import pickle
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque

import erfa
import numpy as np
from PIL import Image
from astropy.time import Time
from picamera2 import Picamera2

import imu_backends

VERSION = "2.1"   # sensors and mounting from configs/ (2026-10-20)

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(REPO_DIR, "configs")
SOCKET_PATH = "/tmp/pisextant_solver.sock"
SHM_DIR = "/dev/shm"

# Wave compensation window. Default OFF: the gyro's zero-rate offset is not
# corrected yet (see the docstring). Pass --wave-s 20 to try it.
WAVE_DEFAULT_S = 0.0
SMOOTH_WINDOW_S = 60.0  # trailing fix-position mean; wave-band error is gone
                        # (decorrelated by ~40 s at sea)
SMOOTH_GATE_DEG = 5.0   # fixes this far from the window median are false solves

TEMP_LOG_EVERY_S = 5.0             # one temp.jsonl record every ~5 s


def load_instrument(path):
    """configs/instrument.json: the "camera" and "imu" sections."""
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Write it for your hardware (see "
                 "configs/README.md), or copy "
                 "examples/reference-build/instrument.json for the reference build.")
    cfg = json.load(open(path))
    for key in ("camera", "imu"):
        if key not in cfg:
            sys.exit(f'{path}: the "{key}" section is missing')
    return cfg


def camera_controls(cam):
    """libcamera controls from the "camera" section. Everything is fixed, so
    frames are comparable across the whole run (auto exposure or white
    balance would change the grayscale response frame to frame)."""
    ctrl = {
        "ExposureTime": int(cam.get("exposure_us", 300000)),
        "AnalogueGain": float(cam.get("analogue_gain", 8.0)),
        "AwbEnable": False,
        "ColourGains": (1.0, 1.0),   # exact values don't matter, consistency does
    }
    if cam.get("lens_position") is not None:   # only for autofocus cameras
        from libcamera import controls
        ctrl["AfMode"] = controls.AfModeEnum.Manual
        ctrl["LensPosition"] = float(cam["lens_position"])   # dioptres, 0 = infinity
    return ctrl


INSTRUMENT_PATH = os.path.join(CONFIG_DIR, "instrument.json")

# Camera-to-IMU mounting rotation (rows: camera boresight, left, up, in IMU
# axes), written by fit_mcase.py. The accelerometer calibration is fit on
# top of this matrix: change one, refit the other. Without the file the
# identity is used and the fixes are wrong until you calibrate.
M_CASE_PATH = os.path.join(CONFIG_DIR, "m_case.json")


def load_m_case(path):
    """(M_CASE, source): configs/m_case.json if present, else the identity."""
    if os.path.exists(path):
        return np.array(json.load(open(path))["M_case"], dtype=float), path
    return np.eye(3), "identity (configs/m_case.json missing -- NOT calibrated)"


M_CASE, M_CASE_SOURCE = load_m_case(M_CASE_PATH)

# Accelerometer calibration. Two formats:
#   {"A": 3x3, "b": [3], "c": [3], "temp_ref_c": T0}
#                          affine model true_up ~ normalize(A@m + b + c(T-T0)),
#                          from calibrate_affine.py / rezero.py
#   {"r_cal": 3x3}         rotation only, valid at the orientation it was fit
# Without the file the run uses M_CASE alone (identity calibration).
R_CAL_PATH = os.path.join(CONFIG_DIR, "accel_cal.json")


def load_cal(path):
    """(A, b, kind, c, temp_ref): c is the optional temperature term (offset
    change per deg C around temp_ref, from calibrate_affine.py --temp), else
    None."""
    if not os.path.exists(path):
        return np.eye(3), np.zeros(3), "identity", None, 25.0
    cal = json.load(open(path))
    if "A" in cal:
        c = np.array(cal["c"]) if cal.get("c") is not None else None
        return (np.array(cal["A"]), np.array(cal["b"]), "affine", c,
                float(cal.get("temp_ref_c", 25.0)))
    return np.array(cal["r_cal"]), np.zeros(3), "rotation", None, 25.0


CAL_A, CAL_B, CAL_KIND, CAL_C, CAL_T0 = load_cal(R_CAL_PATH)


def _stop_on_sigterm(signum, frame):
    """The webui stops a capture with SIGTERM: turn it into the same clean
    shutdown as Ctrl-C, so run_meta.json gets its end time and totals."""
    raise KeyboardInterrupt


# --- Socket helpers (length-prefixed pickle, shared with solve_worker.py) ---

def send_msg(conn, obj):
    data = pickle.dumps(obj)
    conn.sendall(len(data).to_bytes(4, "big") + data)


def recv_msg(conn):
    header = _recv_exact(conn, 4)
    return pickle.loads(_recv_exact(conn, int.from_bytes(header, "big")))


def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("solver connection closed")
        buf += chunk
    return buf


def sanitize(obj):
    """Make numpy types and odd metadata values JSON-serializable."""
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


def quat_rotate(q, v, inverse=False):
    """Rotate vector v by quaternion q = (i, j, k, w) -- body-to-world for the
    game rotation vector, matching scipy's Rotation.from_quat(...).apply().
    inverse=True applies the world-to-body rotation instead."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    if inverse:
        x, y, z = -x, -y, -z
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_mul(q1, q2):
    """Hamilton product, (i, j, k, w) layout: the rotation that applies q2
    first, then q1 (matches scipy's Rotation composition R1 * R2)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def rotvec_quat(rv):
    """Quaternion (i, j, k, w) for a rotation-vector (axis * angle, rad)."""
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return np.array([0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2], 1.0])
    s = np.sin(0.5 * angle) / angle
    return np.array([rv[0] * s, rv[1] * s, rv[2] * s, np.cos(0.5 * angle)])


# --- Fix computation (same math as the original camera_client) --------------

_C2I_CACHE = {}


def icrs_to_cirs_matrix(obs_time):
    """ICRS -> celestial intermediate frame (CIRS) at obs_time: frame bias,
    precession and nutation (IAU 2006/2000A, erfa.c2i06a). The star solve is
    in ICRS (the catalog frame) while the Earth rotation angle is measured in
    CIRS; treating the two as equal (before 2026-10-11) put the zenith ~0.15
    deg (~9 nmi) off, varying slowly with date and place. The matrix moves by
    < 0.01 arcsec per minute, so it is computed once per minute."""
    key = int(obs_time.unix // 60)
    m = _C2I_CACHE.get(key)
    if m is None:
        tt = obs_time.tt
        m = erfa.c2i06a(tt.jd1, tt.jd2)
        _C2I_CACHE.clear()
        _C2I_CACHE[key] = m
    return m


def calculate_fix(ra_deg, dec_deg, roll_deg, gravity_xyz, obs_time, m_case,
                  cal_a, cal_b, cal_c=None, temp_c=None, temp_ref_c=25.0):
    """Single-image celestial fix: camera attitude (star solve) plus the
    gravity vector (IMU, rotated into the camera frame) give the zenith in
    ICRS; rotated to CIRS (precession-nutation) it gives the latitude, and via
    the Earth rotation angle, the longitude."""
    era_deg = obs_time.earth_rotation_angle("tio").deg

    v = cal_a @ (m_case @ np.asarray(gravity_xyz, dtype=float)) + cal_b
    if cal_c is not None and temp_c is not None:     # sensor offset vs temperature
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
    zenith = icrs_to_cirs_matrix(obs_time) @ zenith_icrs     # equator of date

    latitude = np.degrees(np.arcsin(np.clip(zenith[2], -1.0, 1.0)))
    lst_deg = np.degrees(np.arctan2(zenith[1], zenith[0])) % 360
    longitude = (lst_deg - era_deg + 180) % 360 - 180

    return {"latitude": float(latitude), "longitude": float(longitude),
            "lst_deg": float(lst_deg), "era_deg": float(era_deg)}


def smoothed_fix(fix_history, window_s, gate_deg=SMOOTH_GATE_DEG):
    """Trailing mean of the instant fix positions. The fused IMU tilt
    wobbles at the wave period (decorrelated by ~40 s) and the solver adds
    per-frame white noise; both average out here, leaving the calibration
    floor. Longitude is averaged wrap-safely around the first sample.

    Fixes more than gate_deg from the window median are left out of the
    mean: a false star solve lands thousands of nmi away and would spoil
    the mean for a full window, while wave noise stays below ~2 deg."""
    if not fix_history:
        return None
    lats = np.array([la for (t, la, lo) in fix_history])
    ref = fix_history[0][2]
    lons = np.array([((lo - ref + 180.0) % 360.0 - 180.0) + ref
                     for (t, la, lo) in fix_history])
    mla, mlo = np.median(lats), np.median(lons)
    keep = np.hypot(lats - mla, (lons - mlo) * np.cos(np.radians(mla))) <= gate_deg
    return {"latitude": float(np.mean(lats[keep])),
            "longitude": float((np.mean(lons[keep]) + 180.0) % 360.0 - 180.0),
            "n_fixes": int(keep.sum()),
            "n_rejected": int((~keep).sum()),
            "window_s": window_s}


# --- IMU --------------------------------------------------------------------

class ImuLogger:
    """Polls the IMU in a background thread. Streams every sample to
    imu.jsonl (t, gravity = raw accel in m/s^2, gyro in rad/s) and keeps a
    history of (t, gravity, gyro-integrated attitude, gravity-in-gyro-frame)
    long enough for the trailing wave window. The gyro-frame gravity is
    rotated once per sample, so the per-frame wave average stays cheap.

    The gyro-integrated attitude is our own dead-reckoned orientation
    (cumulative quaternion from the gyro alone, arbitrary reference frame).
    Only relative rotations across the wave window are used; its weakness is
    the gyro's zero-rate offset, uncorrected for now (see WAVE_DEFAULT_S)."""

    def __init__(self, jsonl_path, temp_jsonl_path, wave_s, imu_cfg):
        self.imu = imu_backends.make_imu(imu_cfg)
        self.temp_file = open(temp_jsonl_path, "a")
        self.temp_samples = 0
        self._t_last_temp = 0.0
        self.last_temp = None      # deg C, refreshed every TEMP_LOG_EVERY_S
        self._gyro_q = np.array([0.0, 0.0, 0.0, 1.0])   # body -> gyro ref
        self._gyro_prev = None                          # (t, omega) last sample
        self.gyro_samples = 0
        self.file = open(jsonl_path, "a")
        self.history = deque(maxlen=max(1000, int(wave_s * 120) + 200))
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.samples = 0
        self.errors = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _log_temperature(self, t):
        if t - self._t_last_temp < TEMP_LOG_EVERY_S:
            return
        try:
            temp = self.imu.temperature
        except Exception:
            return
        if temp is None:          # sensor without a thermometer
            self._t_last_temp = t
            return
        self.temp_file.write(json.dumps({
            "t": round(t, 2),
            "temp_raw": round(temp, 2),   # degrees C (key name kept for tools)
        }) + "\n")
        self.temp_file.flush()
        self.temp_samples += 1
        self._t_last_temp = t
        self.last_temp = temp

    def start(self):
        self.thread.start()

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                gravity = self.imu.acceleration   # raw accel = gravity at rest
                gyro = self.imu.gyro              # rad/s
            except Exception:
                self.errors += 1
                time.sleep(0.05)
                continue
            t = time.time()
            w = np.asarray(gyro, dtype=float)
            if self._gyro_prev is not None:
                t_prev, w_prev = self._gyro_prev
                dt = t - t_prev
                # attitude kinematics C(t) = C_prev . exp(omega dt); skip
                # across gaps where the integral would be meaningless
                if 0.0 < dt < 0.2:
                    self._gyro_q = quat_mul(
                        self._gyro_q, rotvec_quat(0.5 * (w_prev + w) * dt))
            self._gyro_prev = (t, w)
            self.gyro_samples += 1
            g = np.asarray(gravity, dtype=float)
            # gravity in the gyro reference frame, rotated once here: the wave
            # average then needs only a mean per frame (re-rotating the whole
            # 20 s window every frame took ~0.7 s on the Zero 2W and starved
            # this thread of IMU samples during the exposure)
            gg = quat_rotate(self._gyro_q, g)
            with self.lock:
                self.history.append((t, g, self._gyro_q, gg))
            self.file.write(json.dumps({
                "t": round(t, 4),
                "gravity": [round(v, 5) for v in gravity],
                "gyro": [round(v, 5) for v in gyro],
            }) + "\n")
            self.samples += 1
            if self.samples % 200 == 0:
                self.file.flush()
            self._log_temperature(t)
            time.sleep(0.01)

    def gravity_window(self, t_start, t_end):
        with self.lock:
            return [g for (t, g, gq, gg) in self.history
                    if t_start <= t <= t_end]

    def wave_gravity(self, t_mid, window_s, min_samples=20):
        """Trailing de-rotated gravity average: every sample is carried to the
        exposure's body frame with the gyro-integrated attitude, so wave
        acceleration is near zero-mean over the window. Returns None until a
        full window of history exists."""
        with self.lock:
            window = [(t, gg, gq) for (t, g, gq, gg) in self.history
                      if t_mid - window_s <= t <= t_mid]
        if len(window) < min_samples or window[0][0] > t_mid - 0.9 * window_s:
            return None
        times = np.array([t for t, gg, gq in window])
        mean_ref = np.mean([gg for t, gg, gq in window], axis=0)
        gq_nearest = window[int(np.argmin(np.abs(times - t_mid)))][2]
        return quat_rotate(gq_nearest, mean_ref, inverse=True)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self.file.close()
        self.temp_file.close()


# --- PNG writer (PNG encoding is slow on the Zero 2W, so it runs off the
#     capture loop; a bounded queue applies backpressure instead of OOM) -----

def png_writer(q):
    while True:
        item = q.get()
        if item is None:
            return
        path, gray = item
        Image.fromarray(gray).save(path, compress_level=1)   # lossless, fast


def clock_sync_info():
    """Snapshot of the Pi's clock sync state for run_meta (longitude depends
    directly on absolute time: ~0.25 nmi per second of clock error)."""
    info = {"checked_at": Time.now().iso}
    try:
        out = subprocess.run(["timedatectl", "show"], capture_output=True,
                             text=True, timeout=5).stdout
        info["timedatectl"] = dict(line.split("=", 1)
                                   for line in out.strip().splitlines() if "=" in line)
    except Exception as e:
        info["error"] = str(e)
    return info


# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration-min", type=float, default=15.0)
    parser.add_argument("--interval-s", type=float, default=0.0,
                        help="minimum seconds per cycle (0 = as fast as possible)")
    parser.add_argument("--wave-s", type=float, default=WAVE_DEFAULT_S,
                        help="trailing gravity-averaging window in seconds "
                             "(default 0 = off, see WAVE_DEFAULT_S; 20 is "
                             "the window validated at sea)")
    parser.add_argument("--smooth-s", type=float, default=SMOOTH_WINDOW_S,
                        help="trailing fix-position averaging window in "
                             "seconds (0 disables the smoothed fix)")
    parser.add_argument("--png-every", type=int, default=10,
                        help="save 1 frame image in N (default 10; 1 = all). "
                             "All images cost ~125 MB per minute; they serve "
                             "offline re-solving only -- the fix and the "
                             "calibration use the logs")
    parser.add_argument("--runs-dir", default=os.path.join(REPO_DIR, "runs"))
    parser.add_argument("--socket", default=SOCKET_PATH)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _stop_on_sigterm)
    instrument = load_instrument(INSTRUMENT_PATH)
    cam_cfg, imu_cfg = instrument["camera"], instrument["imu"]
    cam_controls = camera_controls(cam_cfg)
    sensor_mode = int(cam_cfg.get("sensor_mode", 0))

    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.runs_dir, run_id)
    images_dir = os.path.join(run_dir, "images")
    solves_dir = os.path.join(run_dir, "solves")
    os.makedirs(images_dir)
    os.makedirs(solves_dir)
    print(f"Run folder: {run_dir}")

    # Solver connection (worker must already be running in the tetra3 venv)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(args.socket)
    except OSError as e:
        sys.exit(f"Cannot reach solve worker at {args.socket} ({e}). "
                 f"Start solve_worker.py first.")
    hello = recv_msg(sock)
    solver_config = hello["config"]
    config_name = solver_config.get("name", "unnamed")
    print(f"Solver connected: config '{config_name}', database {hello.get('database')}")
    print(f"M_CASE: {M_CASE_SOURCE}")
    if CAL_KIND != "identity":
        print(f"Cal: {CAL_KIND} loaded from {R_CAL_PATH}"
              + (f", with temperature term (T0 {CAL_T0:g} C)" if CAL_C is not None else ""))
    else:
        print(f"WARNING: {R_CAL_PATH} not found -- using M_CASE alone, fixes "
              f"will be UNCALIBRATED (see docs/calibration.md).")
    print(f"Wave compensation: " + (f"trailing {args.wave_s:g} s window"
                                    if args.wave_s > 0 else "OFF"))
    print(f"Smoothed fix: " + (f"trailing {args.smooth_s:g} s mean"
                               if args.smooth_s > 0 else "OFF"))

    # Camera
    picam2 = Picamera2()
    mode = picam2.sensor_modes[sensor_mode]
    width, height = mode["size"]
    config = picam2.create_still_configuration(
        # YUV420: the Y plane IS the grayscale image (BT.601 luma), so no
        # RGB-to-gray conversion is needed on the Zero 2W.
        main={"format": "YUV420", "size": mode["size"]},
        sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]},
    )
    picam2.configure(config)
    picam2.set_controls(cam_controls)
    picam2.start()
    for _ in range(3):   # let exposure/gain settle before the run starts
        picam2.capture_request().release()

    # IMU
    imu = ImuLogger(os.path.join(run_dir, "imu.jsonl"),
                    os.path.join(run_dir, "temp.jsonl"), wave_s=args.wave_s,
                    imu_cfg=imu_cfg)
    imu.start()
    print("Camera and IMU ready.")
    if args.wave_s > 0:
        print("NOTE: wave compensation uses the raw gyro, whose offset is not "
              "corrected yet -- expect large errors (see WAVE_DEFAULT_S).")

    # PNG writer thread
    # 4 frames (~12 MB): PNG saving is only slightly slower than capture,
    # so a deep queue just fills up and parks ~90 MB in RAM (30 frames
    # pushed the Zero 2W into an OOM kill on 2026-10-09)
    png_queue = queue.Queue(maxsize=4)
    png_thread = threading.Thread(target=png_writer, args=(png_queue,), daemon=True)
    png_thread.start()

    # Run metadata (written now, finalized with stats at the end)
    meta = {
        "run_id": run_id,
        "script": "capture_run_wave.py",
        "version": VERSION,
        "protocol_step": "wave",
        "created": Time.now().iso,
        "planned_duration_min": args.duration_min,
        "png_every": args.png_every,
        "camera": {
            "name": cam_cfg.get("name"),
            "sensor_mode": sensor_mode,
            "size": [width, height],
            "bit_depth": mode["bit_depth"],
            "controls": {k: (v if isinstance(v, (int, float)) else str(v))
                         for k, v in cam_controls.items()},
            "image_format": "grayscale PNG from YUV420 Y plane (BT.601 luma)",
        },
        "imu": {
            "sensor": imu_cfg.get("driver"),
            "config": imu_cfg,
            "streams": ["gravity (raw accel, m/s^2)", "gyro (rad/s)"],
            "temperature_log": {
                "file": "temp.jsonl",
                "every_s": TEMP_LOG_EVERY_S,
                "units": "degrees C",
            },
        },
        "wave_compensation": {
            "window_s": args.wave_s,
            "method": "trailing de-rotated gravity average (gyro-integrated "
                      "attitude, gyro offset uncorrected), exposure-median "
                      "fallback; 0 = off, exposure median only",
        },
        "smoothed_fix": {
            "window_s": args.smooth_s,
            "gate_deg": SMOOTH_GATE_DEG,
            "method": "trailing mean of instant fix positions, without "
                      "those > gate_deg from the window median (false solves) "
                      "(fix_smoothed per frame; the navigation output)",
        },
        "M_case": M_CASE.tolist(),
        "M_case_source": M_CASE_SOURCE,
        "R_cal": CAL_A.tolist(),
        "cal_b": CAL_B.tolist(),
        "cal_kind": CAL_KIND,
        "cal_c": CAL_C.tolist() if CAL_C is not None else None,
        "cal_temp_ref_c": CAL_T0,
        "fix_model": "CIRS (precession-nutation IAU 2006/2000A, erfa.c2i06a)",
        "solver": {"config": solver_config, "database": hello.get("database")},
        "clock_sync": clock_sync_info(),
        "hostname": socket.gethostname(),
    }
    meta_path = os.path.join(run_dir, "run_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    capture_log = open(os.path.join(run_dir, "capture.jsonl"), "a")
    solve_log = open(os.path.join(solves_dir, f"live-{config_name}.jsonl"), "a")
    solve_log.write(json.dumps(sanitize({
        "type": "meta", "mode": "live", "config": solver_config,
        "M_case": M_CASE.tolist(), "R_cal": CAL_A.tolist(),
        "cal_b": CAL_B.tolist(), "cal_kind": CAL_KIND,
        "cal_c": CAL_C.tolist() if CAL_C is not None else None,
        "cal_temp_ref_c": CAL_T0,
        "wave_s": args.wave_s,
        "created": Time.now().iso,
    })) + "\n")

    frame_id = 0
    frames_solved = 0
    fixes_computed = 0
    wave_fallbacks = 0
    cycle_times = []
    fix_history = deque()   # (t_exp_mid, lat, lon) for the smoothed fix
    t_run_end = time.time() + args.duration_min * 60

    # Pipeline: frame N+1 exposes while frame N is saved/solved/logged.
    job = picam2.capture_request(wait=False)
    try:
        while time.time() < t_run_end:
            t_cycle0 = time.perf_counter()

            request = picam2.wait(job)
            t_wait_ms = (time.perf_counter() - t_cycle0) * 1000
            metadata = request.get_metadata()
            yuv = request.make_array("main")
            request.release()
            job = picam2.capture_request(wait=False)   # next exposure starts now

            gray = np.ascontiguousarray(yuv[:height, :width])

            # Exposure window in wall time. SensorTimestamp is CLOCK_BOOTTIME
            # (ns) at start of frame readout, i.e. ~end of exposure.
            boot_to_wall = time.time() - time.clock_gettime(time.CLOCK_BOOTTIME)
            exposure_s = metadata["ExposureTime"] / 1e6
            t_exp_end = metadata["SensorTimestamp"] / 1e9 + boot_to_wall
            t_exp_start = t_exp_end - exposure_s
            t_exp_mid = t_exp_end - exposure_s / 2
            obs_time = Time(t_exp_mid, format="unix")

            image_name = f"frame_{frame_id:05d}.png"
            image_saved = frame_id % max(1, args.png_every) == 0
            if image_saved:
                png_queue.put((os.path.join(images_dir, image_name), gray))

            # Median gravity over the exposure window (baseline, and the
            # fallback while the wave window is still filling)
            window = imu.gravity_window(t_exp_start, t_exp_end)
            gravity_median = (np.median(np.asarray(window), axis=0).tolist()
                              if window else None)

            # Wave compensation (off by default): trailing de-rotated average
            gravity_wave = (imu.wave_gravity(t_exp_mid, args.wave_s)
                            if args.wave_s > 0 else None)
            gravity = (gravity_wave.tolist() if gravity_wave is not None
                       else gravity_median)
            if args.wave_s > 0 and gravity_wave is None and gravity_median:
                wave_fallbacks += 1

            # Solve (frame handed to the worker via /dev/shm)
            shm_path = os.path.join(SHM_DIR, f"ns_frame_{frame_id}.npy")
            np.save(shm_path, gray)
            t0 = time.perf_counter()
            send_msg(sock, {"frame_id": frame_id, "path": shm_path})
            response = recv_msg(sock)
            t_solve_ms = (time.perf_counter() - t0) * 1000
            try:
                os.remove(shm_path)   # worker normally removes it; fallback
            except FileNotFoundError:
                pass

            solution = response.get("solution")
            solved = solution is not None and solution.get("RA") is not None

            # Fix
            fix, fix_error = None, None
            t_fix_ms = 0.0
            if solved and gravity is not None:
                t0 = time.perf_counter()
                try:
                    fix = calculate_fix(solution["RA"], solution["Dec"],
                                        solution["Roll"], gravity, obs_time,
                                        M_CASE, CAL_A, CAL_B, CAL_C,
                                        imu.last_temp, CAL_T0)
                except Exception as e:
                    fix_error = str(e)
                t_fix_ms = (time.perf_counter() - t0) * 1000

            # Smoothed fix: trailing mean of the instant fix positions
            fix_smooth = None
            if fix is not None and args.smooth_s > 0:
                fix_history.append((t_exp_mid, fix["latitude"],
                                    fix["longitude"]))
                while fix_history[0][0] < t_exp_mid - args.smooth_s:
                    fix_history.popleft()
                fix_smooth = smoothed_fix(fix_history, args.smooth_s)

            t_cycle_ms = (time.perf_counter() - t_cycle0) * 1000
            timings = {"capture_wait": round(t_wait_ms, 1),
                       "solve_roundtrip": round(t_solve_ms, 1),
                       "fix": round(t_fix_ms, 1),
                       "cycle": round(t_cycle_ms, 1)}

            capture_log.write(json.dumps(sanitize({
                "frame_id": frame_id,
                "image": f"images/{image_name}",
                "image_saved": image_saved,
                "t_exposure_start": t_exp_start,
                "t_exposure_mid": t_exp_mid,
                "t_exposure_end": t_exp_end,
                "iso_mid": obs_time.iso,
                "camera_metadata": metadata,
                "imu_window": {"n_samples": len(window),
                               "gravity_median": gravity_median},
                "timings_ms": timings,
            }), default=str) + "\n")
            capture_log.flush()

            solve_log.write(json.dumps(sanitize({
                "type": "frame",
                "frame_id": frame_id,
                "image": f"images/{image_name}",
                "time_iso_mid": obs_time.iso,
                "status": response.get("status"),
                "n_centroids": response.get("n_centroids"),
                "centroids": response.get("centroids"),
                "solution": solution,
                "gravity_used": gravity,
                "imu_temp_c": imu.last_temp,
                "gravity_median": gravity_median,
                "gravity_wave_gyro": (gravity_wave.tolist()
                                      if gravity_wave is not None else None),
                "wave": {"applied": gravity_wave is not None,
                         "window_s": args.wave_s},
                "n_imu_samples": len(window),
                "fix": fix,
                "fix_smoothed": fix_smooth,
                "fix_error": fix_error,
                "timings_ms": {**response.get("timings", {}), **timings},
            }), default=str) + "\n")
            solve_log.flush()

            cycle_times.append(t_cycle_ms)
            if solved:
                frames_solved += 1
            if fix is not None:
                fixes_computed += 1
                tag = "wave" if gravity_wave is not None else "med "
                smooth_str = (f"  avg ({fix_smooth['latitude']:+.3f}, "
                              f"{fix_smooth['longitude']:+.3f})"
                              f"[{fix_smooth['n_fixes']}]"
                              if fix_smooth else "")
                print(f"frame {frame_id:05d}  {response.get('n_centroids', 0):3d} cent  "
                      f"fix ({fix['latitude']:+.3f}, {fix['longitude']:+.3f}) {tag}"
                      f"{smooth_str}  "
                      f"solve {t_solve_ms:4.0f} ms  cycle {t_cycle_ms:4.0f} ms")
            else:
                reason = response.get("status") if not solved else "no_imu"
                print(f"frame {frame_id:05d}  {response.get('n_centroids', 0):3d} cent  "
                      f"{reason}  solve {t_solve_ms:4.0f} ms  cycle {t_cycle_ms:4.0f} ms")

            frame_id += 1
            if args.interval_s:
                leftover = args.interval_s - (time.perf_counter() - t_cycle0)
                if leftover > 0:
                    time.sleep(leftover)

    except KeyboardInterrupt:
        print("\nInterrupted -- finalizing run.")
    except (ConnectionError, BrokenPipeError) as e:
        print(f"\nLost solver connection ({e}) -- ending run early.")
    finally:
        try:
            picam2.wait(job).release()   # drain the in-flight capture
        except Exception:
            pass
        picam2.stop()
        imu.close()
        png_queue.put(None)
        png_thread.join(timeout=60)
        capture_log.close()
        solve_log.close()
        sock.close()

        meta.update({
            "ended": Time.now().iso,
            "frames_total": frame_id,
            "frames_solved": frames_solved,
            "solve_rate": round(frames_solved / frame_id, 3) if frame_id else None,
            "fixes_computed": fixes_computed,
            "wave_fallback_frames": wave_fallbacks,
            "imu_samples_logged": imu.samples,
            "imu_read_errors": imu.errors,
            "gyro_samples_integrated": imu.gyro_samples,
            "temp_samples_logged": imu.temp_samples,
            "cycle_ms_mean": round(float(np.mean(cycle_times)), 1) if cycle_times else None,
            "cycle_ms_median": round(float(np.median(cycle_times)), 1) if cycle_times else None,
        })
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Run finished: {frame_id} frames, {frames_solved} solved, "
              f"{fixes_computed} fixes ({wave_fallbacks} wave fallbacks) -> {run_dir}")


if __name__ == "__main__":
    main()
