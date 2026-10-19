#!/usr/bin/env python3
"""PiSextant setup assistant -- runs on the Pi, in the capture venv.

Guides you from a fresh install to a calibrated instrument, step by step.
Each step can be run again alone; the progress is kept in
configs/setup_state.json.

    ~/pisextant-hw/bin/python setup_pisextant.py            # menu
    ~/pisextant-hw/bin/python setup_pisextant.py check      # one step

Steps:
    check     pre-checks: venvs, I2C, camera, clock, disk, Earth rotation table
    i2c       scan the I2C bus and name the chips found
    imu       choose or create the IMU driver, then test it live
    camera    choose the sensor mode, compute the field of view, exposure
    database  write configs/solver.json and explain how to get the star database
    solve     solve one live frame (checks the database and the field of view)
    mcase     two runs at a known position -> mounting rotation (M_CASE)
    accel     accelerometer calibration (several orientations) or re-zero plan
    finish    summary and the next steps

WHERE THE FITTING RUNS: the calibration fits (fit_mcase.py,
calibrate_affine.py, rezero.py) load whole runs into memory and need scipy.
Run them on a LAPTOP or desktop: this script records the runs on the Pi and
prints the exact commands to run there. Only on a stronger board (Pi 4/5 with
2 GB of RAM or more) you can fit on the board itself with --fit-here.
"""

import argparse
import glob
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(REPO, "configs")
STATE_PATH = os.path.join(CONFIGS, "setup_state.json")
INSTRUMENT_PATH = os.path.join(CONFIGS, "instrument.json")
SOLVER_PATH = os.path.join(CONFIGS, "solver.json")
SOCKET_PATH = "/tmp/pisextant_solver.sock"
SOLVER_PYTHON = os.path.expanduser("~/pisextant-solver/bin/python")
CATALOG_URL = "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat"

# Chips often found on the I2C bus (several chips share some addresses).
KNOWN_I2C = {
    0x68: "DS3231 / DS1307 RTC, or MPU-6050/6500/9250 IMU (they clash)",
    0x69: "MPU-6050/9250 IMU (AD0 high), or ICM-20948",
    0x6a: "LSM6DS family IMU (ISM330DHCX, LSM6DSOX, LSM6DS3...)",
    0x6b: "LSM6DS family IMU (ADR high)",
    0x28: "BNO055 IMU", 0x29: "BNO055 IMU (alt) / VL53L0X",
    0x4a: "BNO08x IMU", 0x4b: "BNO08x IMU (alt)",
    0x1c: "LIS3MDL / MMA845x", 0x1e: "LIS3MDL / HMC5883L magnetometer",
    0x57: "AT24C32 EEPROM (on DS3231 boards)",
    0x76: "BME280/BMP280 pressure sensor", 0x77: "BME280/BMP280 / BMP388",
}
KNOWN_DRIVERS = {0x6a: "ism330dhcx", 0x6b: "ism330dhcx"}

# Common Raspberry Pi cameras: focal length (mm), pixel size (um), full width (px).
CAMERAS = {
    "1": ("Camera Module 3 (IMX708), standard lens", 4.74, 1.4, 4608),
    "2": ("Camera Module 3 Wide (IMX708)", 2.75, 1.4, 4608),
    "3": ("Camera Module 2 (IMX219)", 3.04, 1.12, 3280),
    "4": ("HQ Camera (IMX477) -- you give the lens focal length", None, 1.55, 4056),
    "5": ("Global Shutter Camera (IMX296) -- you give the lens focal length", None, 3.45, 1456),
}


# --- small helpers ----------------------------------------------------------------

def say(text=""):
    print(text)


def title(text):
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)


def ok(text):
    print(f"  [ ok ] {text}")


def warn(text):
    print(f"  [warn] {text}")


def bad(text):
    print(f"  [FAIL] {text}")


def ask(question, default=None):
    hint = f" [{default}]" if default not in (None, "") else ""
    try:
        answer = input(f"{question}{hint}: ").strip()
    except EOFError:
        answer = ""
    return answer or ("" if default is None else str(default))


def ask_float(question, default=None):
    while True:
        a = ask(question, default)
        try:
            return float(a)
        except ValueError:
            print("  please type a number")


def yes(question, default="y"):
    return ask(question + " (y/n)", default).lower().startswith("y")


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def state():
    return load_json(STATE_PATH, {}) or {}


def set_state(**kw):
    s = state()
    s.update(kw)
    os.makedirs(CONFIGS, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(s, f, indent=2)


def instrument():
    return load_json(INSTRUMENT_PATH, {}) or {}


def run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout + p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def known_position():
    s = state()
    lat = ask_float("Known latitude in decimal degrees (north +)", s.get("lat"))
    lon = ask_float("Known longitude in decimal degrees (east +)", s.get("lon"))
    set_state(lat=lat, lon=lon)
    return lat, lon


def laptop_or_here(cmd, fit_here):
    """Run a fitting command here (strong board) or print it for the laptop."""
    if fit_here:
        say(f"\nRunning here: {cmd}")
        subprocess.run(cmd, shell=True, cwd=REPO)
        return
    say("\nRun this ON YOUR LAPTOP (the fit needs more memory than a Pi Zero has):")
    say("  1. copy the run folders from the Pi, for example:")
    say(f"       rsync -a --exclude images/ <user>@<pi>:{REPO}/runs/ runs/")
    say(f"  2. in your PiSextant folder on the laptop:\n       {cmd}")
    say("  3. copy the file it writes into configs/ on the Pi (scp).")


# --- solver process + one-frame solve ---------------------------------------------

def solver_running():
    if not os.path.exists(SOCKET_PATH):
        return False
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(SOCKET_PATH)
        s.close()
        return True
    except OSError:
        return False


def start_solver():
    """Start solve_worker.py in the solver venv; returns the process (or None
    if a worker was already running)."""
    if solver_running():
        say("Solver already running.")
        return None
    if not os.path.exists(SOLVER_PATH):
        bad("configs/solver.json missing -- run the 'database' step first")
        return None
    py = SOLVER_PYTHON if os.path.exists(SOLVER_PYTHON) else sys.executable
    say(f"Starting the solver ({py}) ... (loading the database takes ~30 s)")
    proc = subprocess.Popen([py, "solve_worker.py", SOLVER_PATH], cwd=REPO,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    for _ in range(120):
        if solver_running():
            ok("solver ready")
            return proc
        if proc.poll() is not None:
            bad(f"solver stopped (exit {proc.returncode}); run it by hand to see the error:\n"
                f"       {py} solve_worker.py configs/solver.json")
            return None
        time.sleep(1)
    bad("solver did not start in 2 minutes")
    return proc


def stop_solver(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def capture_runs(minutes, png_every=30):
    """Record one run with the normal capture script; returns the run id."""
    before = set(os.listdir(os.path.join(REPO, "runs"))) if os.path.isdir(os.path.join(REPO, "runs")) else set()
    say(f"Recording {minutes:g} min ... (Ctrl-C stops early and keeps the run)")
    try:
        subprocess.run([sys.executable, "capture_run_wave.py", "--duration-min", str(minutes),
                        "--png-every", str(png_every)], cwd=REPO)
    except KeyboardInterrupt:
        pass
    after = set(os.listdir(os.path.join(REPO, "runs")))
    new = sorted(after - before)
    if not new:
        bad("no run folder was written")
        return None
    rid = new[-1]
    meta = load_json(os.path.join(REPO, "runs", rid, "run_meta.json"), {})
    solved, total = meta.get("frames_solved"), meta.get("frames_total")
    if total:
        (ok if solved and solved / total > 0.5 else warn)(f"run {rid}: {solved}/{total} frames solved")
    return rid


# --- steps ----------------------------------------------------------------------------

def step_check(args):
    title("Pre-checks")
    say(f"Python {sys.version.split()[0]} at {sys.executable}")
    for name, path in (("capture venv", "~/pisextant-hw/bin/python"),
                       ("solver venv", "~/pisextant-solver/bin/python")):
        (ok if os.path.exists(os.path.expanduser(path)) else bad)(f"{name}: {path}")
    (ok if os.path.exists("/dev/i2c-1") else bad)("I2C bus /dev/i2c-1 "
        + ("present" if os.path.exists("/dev/i2c-1") else "missing: sudo raspi-config nonint do_i2c 0, then reboot"))
    try:
        from picamera2 import Picamera2
        cams = Picamera2.global_camera_info()
        (ok if cams else bad)(f"camera: {cams[0].get('Model') if cams else 'none detected (cable? camera_auto_detect=1?)'}")
    except Exception as e:
        bad(f"picamera2 not usable here ({e}); use the capture venv (--system-site-packages)")
    code, out = run(["timedatectl", "show", "--property", "NTPSynchronized", "--value"])
    ntp = out.strip() == "yes"
    rtc = os.path.exists("/dev/rtc0")
    if ntp:
        ok("clock synchronized by NTP" + (" -- now save it in the RTC: sudo hwclock -w" if rtc else ""))
    elif rtc:
        warn("no NTP; the clock comes from the RTC (fine if the RTC was set once)")
    else:
        bad("no NTP and no RTC: the clock can be wrong (1 s = 0.25 nmi of longitude)")
    free = shutil.disk_usage(REPO).free / 1e9
    (ok if free > 4 else warn)(f"disk: {free:.1f} GB free")
    try:
        from astropy.time import Time
        from astropy.utils import iers
        iers.conf.auto_download = False
        tab = iers.IERS_Auto.open()
        end = Time(tab["MJD"][-1], format="mjd")
        days = end.mjd - Time.now().mjd
        (ok if days > 30 else warn)(f"Earth rotation table: predictions until {end.iso[:10]} "
            + ("" if days > 30 else "-- refresh it (README step 4)"))
    except Exception as e:
        warn(f"Earth rotation table not checked ({str(e)[:80]})")
    cfg_files = ["instrument.json", "solver.json", "m_case.json", "accel_cal.json"]
    for f in cfg_files:
        p = os.path.join(CONFIGS, f)
        (ok if os.path.exists(p) else warn)(f"configs/{f} " + ("present" if os.path.exists(p) else "not written yet"))


def parse_i2cdetect(out):
    """Addresses in an 'i2cdetect -y 1' table: hex = a device, UU = a device
    held by a kernel driver (e.g. the RTC). Columns are 3 characters wide
    after the 'NN:' row label; empty cells are blank, '--' or 'UU'."""
    found = []
    for line in out.splitlines()[1:]:
        if ":" not in line:
            continue
        row = int(line.split(":")[0], 16)
        body = line.split(":", 1)[1]
        for col in range(16):
            cell = body[col * 3:col * 3 + 3].strip()
            if cell and cell != "--":
                found.append(row + col)
    return found


def step_i2c(args):
    title("I2C scan (bus 1)")
    code, out = run(["i2cdetect", "-y", "1"])
    if code != 0:
        bad(f"i2cdetect failed: {out.strip()[:200]}\n         install it: sudo apt install i2c-tools")
        return
    found = parse_i2cdetect(out)
    if not found:
        bad("no device answers on bus 1: check 3.3 V, GND, SDA (pin 3), SCL (pin 5)")
        return
    for a in sorted(set(found)):
        say(f"  0x{a:02x}  {KNOWN_I2C.get(a, 'unknown chip')}" + ("   (in use by a kernel driver, e.g. the RTC)" if a == 0x68 else ""))
    imus = [a for a in found if a in KNOWN_DRIVERS]
    if imus:
        ok(f"IMU with a ready driver: {KNOWN_DRIVERS[imus[0]]} at 0x{imus[0]:02x}")
        set_state(imu_guess={"driver": KNOWN_DRIVERS[imus[0]], "address": f"0x{imus[0]:02x}"})
    else:
        warn("no IMU with a ready driver: the 'imu' step helps you write one")


def step_imu(args):
    title("IMU driver")
    drivers = sorted(os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join(REPO, "imu_drivers", "*.py"))
                     if not os.path.basename(p).startswith(("_", "template")))
    guess = state().get("imu_guess", {})
    say("Drivers available: " + (", ".join(drivers) or "none"))
    name = ask("Driver to use (or a NEW name to create one from the template)",
               guess.get("driver", drivers[0] if drivers else ""))
    path = os.path.join(REPO, "imu_drivers", name + ".py")
    if not os.path.exists(path):
        shutil.copy(os.path.join(REPO, "imu_drivers", "template.py"), path)
        say(f"\nCreated {path} from the template. Write the driver for your sensor:")
        say("  open_imu(cfg) must return an object with")
        say("    acceleration  (ax, ay, az) in m/s^2, RAW (|a| ~ 9.8 at rest)")
        say("    gyro          (gx, gy, gz) in rad/s")
        say("    temperature   degrees C, or None")
        say("  The capture logs each sample in imu.jsonl as:")
        say('    {"t": 1792152031.2345, "gravity": [0.41, -9.66, 0.53], "gyro": [-0.0037, 0.0063, -0.0008]}')
        say("Then run this step again to test it.")
        return
    imu_cfg = {"driver": name}
    address = ask("I2C address (leave empty if the driver needs none)", guess.get("address", ""))
    if address:
        imu_cfg["address"] = address
    inst = instrument()
    inst["imu"] = imu_cfg
    save_json(INSTRUMENT_PATH, inst)
    ok(f"configs/instrument.json: imu = {imu_cfg}")

    say("\nLive test: keep the instrument STILL for 3 s ...")
    sys.path.insert(0, REPO)
    import numpy as np
    import imu_backends
    try:
        imu = imu_backends.make_imu(imu_cfg)
    except Exception as e:
        bad(f"the driver did not start: {e}")
        return
    acc, gyr, t0 = [], [], time.time()
    while time.time() - t0 < 3:
        try:
            acc.append(imu.acceleration)
            gyr.append(imu.gyro)
        except Exception as e:
            bad(f"read error: {e}")
            return
        time.sleep(0.005)
    acc, gyr = np.array(acc, float), np.array(gyr, float)
    rate = len(acc) / 3
    g = np.linalg.norm(acc.mean(0))
    say(f"  sample: {json.dumps({'gravity': acc[-1].round(4).tolist(), 'gyro': gyr[-1].round(5).tolist()})}")
    (ok if rate >= 50 else warn)(f"rate {rate:.0f} samples/s (50 or more is good)")
    (ok if 9.3 < g < 10.3 else bad)(f"|gravity| = {g:.3f} m/s^2 (expected ~9.8; else wrong units or scale)")
    (ok if acc.std(0).max() < 0.2 else warn)(f"accel noise {acc.std(0).max():.3f} m/s^2 (still?)")
    gw = np.degrees(np.abs(gyr.mean(0))).max()
    (ok if gw < 3 else warn)(f"gyro at rest {gw:.2f} deg/s (expected < ~3; check rad/s units)")
    try:
        temp = imu.temperature
        (ok if temp is None or -20 < temp < 80 else warn)(f"temperature {temp}")
    except Exception as e:
        warn(f"temperature not readable ({e})")
    u0 = acc.mean(0) / g
    if yes("\nAxis test: tilt the instrument by ~30 degrees now, then press Enter. Ready?"):
        time.sleep(0.5)
        u1 = np.mean([imu.acceleration for _ in range(50)], axis=0)
        ang = math.degrees(math.acos(max(-1, min(1, float(u0 @ (u1 / np.linalg.norm(u1)))))))
        (ok if ang > 10 else bad)(f"gravity direction moved by {ang:.0f} deg")
    set_state(imu_ok=True)


def fov_deg(width_mm, focal_mm):
    return math.degrees(2 * math.atan(width_mm / (2 * focal_mm)))


def step_camera(args):
    title("Camera")
    from picamera2 import Picamera2
    cam = Picamera2()
    modes = cam.sensor_modes
    for i, m in enumerate(modes):
        say(f"  mode {i}: {m.get('size')} {m.get('bit_depth')} bit, crop {m.get('crop_limits')}")
    props = cam.camera_properties
    has_af = "LensPosition" in cam.camera_controls
    cam.close()
    say("A binned mode (half resolution over the full sensor) gives brighter stars and faster solves.")
    mode = int(ask_float("Sensor mode", 1 if len(modes) > 1 else 0))
    say("\nCameras:")
    for k, (n, f, p, w) in CAMERAS.items():
        say(f"  {k}. {n}")
    say("  6. other")
    choice = ask("Your camera", "1")
    name, focal, pix, full_w = CAMERAS.get(choice, ("other", None, None, None))
    if focal is None:
        focal = ask_float("Lens focal length (mm)")
    if pix is None:
        pix = ask_float("Pixel size (micrometres)")
    crop = modes[mode].get("crop_limits")
    width_px = crop[2] if crop else (full_w or props.get("PixelArraySize", (0, 0))[0])
    fov = fov_deg(width_px * pix / 1000, focal)
    say(f"\nHorizontal field of view: {fov:.2f} deg ({width_px} px x {pix} um, focal {focal} mm)")
    exposure = int(ask_float("Exposure in microseconds (stars at sea: ~200000-300000)", 300000))
    gain = ask_float("Analogue gain", 12.0)
    inst = instrument()
    inst["camera"] = {"name": name, "sensor_mode": mode, "exposure_us": exposure,
                      "analogue_gain": gain, "focal_length_mm": focal, "pixel_size_um": pix,
                      "fov_deg": round(fov, 3)}
    if has_af:
        inst["camera"]["lens_position"] = 0.0
        say("Autofocus camera: focus fixed at infinity (lens_position 0.0).")
    save_json(INSTRUMENT_PATH, inst)
    ok("configs/instrument.json: camera written")
    set_state(fov_deg=round(fov, 3))


def step_database(args):
    title("Star database and solver config")
    fov = state().get("fov_deg") or instrument().get("camera", {}).get("fov_deg")
    if not fov:
        bad("run the 'camera' step first (it computes the field of view)")
        return
    ref = load_json(os.path.join(REPO, "examples", "reference-build", "solver.json"), {})
    cfg = json.loads(json.dumps(ref)) if ref else {"detect": {}, "solve": {}, "database": {}}
    cfg["name"] = "my-instrument"
    cfg["database"]["path"] = "data/database.npz"
    cfg["database"].setdefault("build_params", {})["max_fov"] = round(fov * 1.02, 3)
    cfg["solve"]["fov_estimate"] = round(fov, 3)
    save_json(SOLVER_PATH, cfg)
    ok(f"configs/solver.json written (fov_estimate {fov:.2f} deg, database data/database.npz)")
    db = os.path.join(REPO, "data", "database.npz")
    if os.path.exists(db):
        ok(f"{db} present")
        return
    max_fov = round(fov * 1.02, 2)
    say(f"""
The star database is not there yet. Build it once, on a LAPTOP (a few minutes):

  1. Get the Hipparcos main catalog file hip_main.dat (~51 MB):
       {CATALOG_URL}
  2. Install cedar-solve on the laptop (README, step 3; no Rust needed for this).
  3. In your PiSextant folder on the laptop:
       python build_database.py --catalog hip_main.dat --max-fov {max_fov}
  4. Copy data/database.npz to the Pi, into {os.path.join(REPO, 'data')}/

Same camera and lens as the reference build? Use its database instead:
  set "path": "examples/reference-build/database.npz" in configs/solver.json.""")


def step_solve(args):
    title("Solve one live frame")
    proc = start_solver()
    if not solver_running():
        return
    try:
        sys.path.insert(0, REPO)
        import numpy as np
        from picamera2 import Picamera2
        import pickle
        cam_cfg = instrument().get("camera", {})
        cam = Picamera2()
        mode = cam.sensor_modes[int(cam_cfg.get("sensor_mode", 0))]
        cam.configure(cam.create_still_configuration(
            main={"format": "YUV420", "size": mode["size"]},
            sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]}))
        from capture_run_wave import camera_controls
        cam.set_controls(camera_controls(cam_cfg))
        cam.start()
        for _ in range(3):
            cam.capture_request().release()
        yuv = cam.capture_array("main")
        cam.stop()
        cam.close()
        w, h = mode["size"]
        gray = np.ascontiguousarray(yuv[:h, :w])
        path = "/dev/shm/pisextant_setup_frame.npy"
        np.save(path, gray)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(SOCKET_PATH)

        def recv():
            hdr = s.recv(4)
            n = int.from_bytes(hdr, "big")
            buf = b""
            while len(buf) < n:
                buf += s.recv(n - len(buf))
            return pickle.loads(buf)
        recv()                                   # hello
        data = pickle.dumps({"frame_id": 0, "path": path})
        s.sendall(len(data).to_bytes(4, "big") + data)
        r = recv()
        s.close()
        say(f"  stars detected: {r.get('n_centroids')}, status: {r.get('status')}")
        sol = r.get("solution") or {}
        if sol.get("RA") is None:
            warn("no solution: clear sky? stars sharp? (fewer than ~15 stars: longer exposure or more gain)")
            return
        ok(f"solved: RA {sol['RA']:.2f}, Dec {sol['Dec']:.2f}, Roll {sol['Roll']:.1f}, "
           f"{sol.get('Matches')} stars matched")
        conf = load_json(SOLVER_PATH, {}).get("solve", {}).get("fov_estimate")
        if conf and sol.get("FOV"):
            diff = abs(sol["FOV"] - conf) / conf
            (ok if diff < 0.02 else warn)(f"measured field of view {sol['FOV']:.2f} deg "
                f"(configured {conf:.2f}, {100 * diff:.1f} % off)")
            if diff >= 0.02:
                say("  Put the measured value in configs/solver.json (solve.fov_estimate) and,"
                    " if it is larger than the database max_fov, rebuild the database.")
        set_state(solve_ok=True)
    finally:
        stop_solver(proc)


def step_mcase(args):
    title("Mounting rotation (M_CASE)")
    say("""Needs: a clear sky, the instrument STATIC at a KNOWN position (about
200 m or better), and the solver database. Two runs of ~10 minutes:
  run 1: enclosure flat (or in its normal position)
  run 2: one side lifted by 20-30 degrees""")
    lat, lon = known_position()
    minutes = ask_float("Minutes per run", 10)
    proc = start_solver()
    if not solver_running():
        return
    try:
        ids = []
        for label in ("FLAT (normal position)", "TILTED 20-30 degrees"):
            if not yes(f"\nPut the instrument {label}. Start the run?"):
                return
            rid = capture_runs(minutes)
            if rid:
                ids.append(rid)
        set_state(mcase_runs=ids)
    finally:
        stop_solver(proc)
    if len(ids) == 2:
        cmd = (f"python fit_mcase.py runs/{ids[0]} runs/{ids[1]} --lat {lat} --lon {lon}")
        laptop_or_here(cmd, args.fit_here)
        say("It writes configs/m_case.json. The fixes become usable after the next step.")


def step_accel(args):
    title("Accelerometer calibration")
    say("""How will the instrument be used?
  1. fixed on a ship (near level, one orientation)
  2. at several orientations (moved, handheld, different mounts)""")
    use = ask("Choice", "1")
    lat, lon = known_position()
    if use == "1":
        say("""
For a fixed mount, the minimum is M_CASE (previous step) + a RE-ZERO once the
instrument is installed. The full calibration below is still RECOMMENDED: a
ship's static heel and trim change by a few degrees between days, and without
it this costs a few nautical miles.""")
        if not yes("Do the full calibration now (6-9 orientations, ~1 h)?", "y"):
            say(f"""
Re-zero after installation, with the ship's GPS track in each run's gps.jsonl:
  python rezero.py runs/<night1> runs/<night2> --cal configs/accel_cal.json --out configs/accel_cal.json.new
(no accel_cal.json yet? write {{"A": identity, "b": [0, 0, 0]}} first, see docs/calibration.md)""")
            return
    n = int(ask_float("Number of orientations (6-9)", 8))
    minutes = ask_float("Minutes per orientation", 6)
    poses = ["flat", "tilted ~30 deg toward the front", "tilted ~30 deg toward the back",
             "tilted ~30 deg to the left", "tilted ~30 deg to the right",
             "tilted ~50 deg toward the front", "rolled ~45 deg about the camera axis",
             "tilted ~20 deg front-left", "tilted ~20 deg back-right"]
    say("Start COLD if you can (the warm-up gives the temperature range). Move the whole")
    say("enclosure between runs; never press on the camera or the IMU.")
    proc = start_solver()
    if not solver_running():
        return
    ids = []
    try:
        for i in range(n):
            if not yes(f"\nOrientation {i + 1}/{n}: {poses[i % len(poses)]}. Start?"):
                break
            rid = capture_runs(minutes)
            if rid:
                ids.append(rid)
        set_state(accel_runs=ids)
    finally:
        stop_solver(proc)
    if len(ids) >= 4:
        runs = " ".join(f"runs/{r}" for r in ids)
        cmd = (f"python calibrate_affine.py {runs} --position={lat},{lon} --temp "
               f"--out configs/accel_cal.json")
        laptop_or_here(cmd, args.fit_here)
        say("Check the 'leave one orientation out' numbers it prints: that is the error to expect.")
    else:
        warn("fewer than 4 runs: not enough for the calibration")


def step_finish(args):
    title("Summary")
    for f in ("instrument.json", "solver.json", "m_case.json", "accel_cal.json"):
        p = os.path.join(CONFIGS, f)
        (ok if os.path.exists(p) else warn)(f"configs/{f}")
    db = load_json(SOLVER_PATH, {}).get("database", {}).get("path")
    if db:
        (ok if os.path.exists(os.path.join(REPO, db)) else warn)(f"star database {db}")
    say("""
Next steps:
  - web interface at boot: README, install step 6
  - install the instrument on the ship, then RE-ZERO with one or two nights
    of the ship's GPS track (docs/calibration.md, step 3)
  - before each trip: refresh the Earth rotation table and the map tiles
  - check a run against GPS: python sea_eval.py runs/<run> (on the laptop)""")


STEPS = [("check", step_check), ("i2c", step_i2c), ("imu", step_imu), ("camera", step_camera),
         ("database", step_database), ("solve", step_solve), ("mcase", step_mcase),
         ("accel", step_accel), ("finish", step_finish)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", nargs="?", choices=[n for n, _ in STEPS])
    ap.add_argument("--fit-here", action="store_true",
                    help="run the calibration fits on this board (Pi 4/5 with >= 2 GB only)")
    args = ap.parse_args()
    os.chdir(REPO)
    if args.step:
        dict(STEPS)[args.step](args)
        return
    while True:
        title("PiSextant setup")
        for i, (n, _) in enumerate(STEPS, 1):
            say(f"  {i}. {n}")
        say("  q. quit")
        a = ask("Step", "")
        if a.lower() in ("q", "quit", ""):
            break
        try:
            idx = int(a) - 1
            STEPS[idx][1](args)
        except (ValueError, IndexError):
            say("type a step number")
        except KeyboardInterrupt:
            say("\n(step interrupted)")


if __name__ == "__main__":
    main()
