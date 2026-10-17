# PiSextant

**A standalone celestial position system on a Raspberry Pi.** A camera looks at
the night sky, an accelerometer measures the local vertical, and the Pi computes
its latitude and longitude continuously, with no GPS. It is made to be fixed on
a ship: you calibrate it once, and then it runs alone and shows its position
and track in a web page.

<!-- PHOTO: the assembled instrument -->

> Status: working prototype. Tested at sea in October 2026 (see
> [Accuracy](#accuracy)). The software is stable; the documentation and the
> support for other sensors are still in progress.

## Contents

- [How it works](#how-it-works)
- [Accuracy](#accuracy)
- [Hardware](#hardware)
- [Software installation](#software-installation)
- [Calibration](#calibration)
- [Running](#running)
- [Checking a run against a known track](#checking-a-run-against-a-known-track)
- [Files and folders](#files-and-folders)
- [Other cameras and IMUs](#other-cameras-and-imus)
- [Known limits](#known-limits)
- [Credits and licenses](#credits-and-licenses)

## How it works

A position on the Earth is the point where the local vertical (the zenith)
meets the sky. PiSextant measures this in two parts:

1. **The camera** takes an image of the stars (300 ms exposure, about 2 frames
   per second). The solver finds the star spots, matches their pattern with a
   star catalog and gives the **attitude** of the camera: where it points in
   the sky and how it is turned.
2. **The IMU accelerometer** gives the direction of gravity in the instrument.
   With the calibration, this becomes the direction of the **zenith in the
   camera frame**.

The attitude converts the zenith from the camera frame into the sky. With the
exact time (Earth rotation, precession and nutation), the zenith in the sky
gives the latitude and the longitude.

Each frame gives one fix. On a moving ship, one fix has an error of 10-100
nautical miles, mostly because the ship accelerates in the waves. These errors
are random, so the **60-second average** of the fixes removes most of them.
This average is the navigation output.

Two rules of thumb:

- an error of 1 arc minute in the vertical gives **1 nautical mile** (nmi) of
  position error;
- an error of 1 second in the clock gives **0.25 nmi** of longitude error.

## Accuracy

Results at sea, instrument fixed on the deck of a ship, compared with the
ship's GPS. "Offset" is the median error over the full run; "60 s fix" is the
error of the 60-second average (the navigation output).

| Night (2026) | Duration | Calibration | Offset | 60 s fix, median / 95 % | Solve rate |
|---|---|---|---|---|---|
| Oct 13 | 38 min | at the berth, then carried to the deck | 1.2 nmi | 2.4 / 4.3 nmi | 92 % |
| Oct 15 | 3 h | same, after a knock during a move | 7 nmi | 7.1 / 10.8 nmi | 92 % |
| Oct 16 | 3 h 20 | re-zeroed at sea on Oct 15 | 2.8 nmi | 3.6 / 7.8 nmi | 85 % |
| Oct 17 | 4 h | re-zeroed on Oct 15 + 16, clock from RTC only | **1.35 nmi** | **1.6 / 4.2 nmi** | 96 % |

What these runs show:

- **The fix does not drift with time** (less than 0.5 nmi per hour over 3-4 h).
- **A knock can turn the camera relative to the IMU** by 0.1°, which is about
  6 nmi. A re-zero (one or two nights with a GPS track) removes it.
- **The DS3231 real-time clock is good enough** to run without internet.

## Hardware

<!-- PHOTO: the parts, the wiring, the inside of the enclosure -->

| Part | Reference used | Note |
|---|---|---|
| Computer | Raspberry Pi Zero 2 W | 512 MB RAM; any Pi with a CSI camera port should work |
| Camera | Raspberry Pi Camera Module 3, standard lens (Sony IMX708), not the Wide version | focus at infinity; ~68° horizontal field of view in the 2304×1296 mode |
| IMU | Adafruit ISM330DHCX (STEMMA QT) | accelerometer + gyroscope, I²C |
| Clock | DS3231 RTC module with a coin cell | keeps the time without network |
| Enclosure | 3D printed (files in `hardware/`, coming) | see the mounting rules below |
| Power | 5 V USB supply | the Pi warms the enclosure: see [Known limits](#known-limits) |

<!-- TODO: 3D printable files in hardware/ -->

### Wiring

The IMU and the RTC share the I²C bus 1 of the Pi:

| Signal | Pi header pin | ISM330DHCX | DS3231 |
|---|---|---|---|
| 3.3 V | 1 | VIN | VCC |
| SDA (GPIO 2) | 3 | SDA | SDA |
| SCL (GPIO 3) | 5 | SCL | SCL |
| GND | 6 | GND | GND |

Expected I²C addresses: ISM330DHCX at `0x6a` (`0x6b` if its ADR jumper is
bridged), DS3231 at `0x68` (and its EEPROM at `0x57`). Check with
`i2cdetect -y 1` (package `i2c-tools`).

The camera connects to the CSI port (the Pi Zero needs the narrow 22-pin
cable).

### Mounting rules

These rules come from real failures. They matter more than any software
setting.

1. **The camera and the IMU must not move relative to each other.** A rotation
   of 0.1° between them gives about 6 nmi of error. Screw both boards to the
   same rigid part. A knock or a drop can shift them; re-zero after any shock.
2. **Fix the enclosure rigidly to the ship**, as low and as close to the
   centre of the ship as the sky view permits. High positions (a mast) add
   more wave acceleration.
3. **Give the camera a clear sky.** The camera must not see the ship's lights,
   masts or funnel smoke. The reference build tilts the camera about 23° from
   the vertical.
4. **Keep the IMU away from heat sources** if you can. Its offset changes with
   temperature (about 1 nmi per °C after the temperature calibration).

## Software installation

Tested on Raspberry Pi OS (Debian 13 "Trixie", 64-bit) on a Pi Zero 2 W. The
Pi runs three programs:

| Program | Python | Role |
|---|---|---|
| `webui/server.py` | system `python3`, standard library only | web page, starts and stops the two others |
| `solve_worker.py` | venv `~/pisextant-solver` (Python 3.11) | star detection and plate solving |
| `capture_run_wave.py` | venv `~/pisextant-hw` (system Python 3.13) | camera, IMU, fix computation, all logs |

The two venvs are separate because the camera stack (picamera2, from apt) needs
the system Python 3.13, while the solver needs numpy < 2, which exists only
for Python 3.12 and older. The versions in `requirements/` are the ones of the
reference instrument.

### 1. System

```bash
sudo raspi-config nonint do_i2c 0                  # enable I2C
sudo apt install git python3-venv python3-picamera2 i2c-tools
git clone <this repository> ~/PiSextant
```

Real-time clock: add this line to `/boot/firmware/config.txt`, then reboot.

```
dtoverlay=i2c-rtc,ds3231
```

At boot, the kernel then sets the system time from the DS3231. When the Pi has
network once (NTP synchronized), write the good time into the RTC:

```bash
timedatectl                 # check "System clock synchronized: yes"
sudo hwclock -w
```

If the kernel log shows `rtc-ds1307 ... SET TIME!` at each boot, the RTC still
has its "oscillator stopped" flag; `sudo hwclock -w` clears it.

### 2. Capture venv (camera + IMU)

```bash
python3 -m venv --system-site-packages ~/pisextant-hw
~/pisextant-hw/bin/pip install -r ~/PiSextant/requirements/pi-hw.txt
```

`--system-site-packages` makes picamera2 (from apt) visible in the venv.

### 3. Solver venv (cedar-solve and cedar-detect)

The solver is [cedar-solve](https://github.com/smroid/cedar-solve) (a fork of
ESA's tetra3) with the [cedar-detect](https://github.com/smroid/cedar-detect)
star detector. cedar-detect is a Rust program; its executable
`cedar-detect-server` must be built for the Pi (aarch64) and placed in
cedar-solve's `tetra3/bin/` folder.

The solver venv uses Python 3.11, installed with
[uv](https://docs.astral.sh/uv/) (it downloads its own Python):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # installs uv
uv venv --python 3.11 ~/pisextant-solver
uv pip install --python ~/pisextant-solver -r ~/PiSextant/requirements/pi-solver.txt

git clone https://github.com/smroid/cedar-solve ~/cedar-solve
cd ~/cedar-solve
git checkout 191b8f6                                       # the version used here
git apply ~/PiSextant/patches/cedar-solve-pillow9.patch
git apply ~/PiSextant/patches/tetra3-trimmed-centroids.patch
uv pip install --python ~/pisextant-solver .

# cedar-detect-server, built with Rust (https://rustup.rs) on the Pi or
# cross-compiled for aarch64 on another computer:
git clone https://github.com/smroid/cedar-detect ~/cedar-detect
cd ~/cedar-detect
cargo build --release --bin cedar-detect-server
cp target/release/cedar-detect-server ~/cedar-solve/tetra3/bin/
```

Building Rust on a Pi Zero 2 W is slow and needs swap; cross-compiling on a
laptop is faster. `solve_worker.py` looks for the executable in tetra3's
`bin/` folder; set the environment variable `CEDAR_DETECT_BINARY` to use
another path.

**Why the patches:**

- `cedar-solve-pillow9.patch` allows Pillow 9 (cedar-solve asks for Pillow < 9,
  which is not available for this setup).
- `tetra3-trimmed-centroids.patch` (4 lines) protects against an `IndexError`:
  the solver configuration uses up to 95 star spots, more than the 75
  verification stars of the database, and tetra3 can then use an index past
  the end of the trimmed list. The reference instrument ran without it with
  no such crash seen in 46,000 frames, so it is a protection only.

The star pattern database for the reference camera is included:
`data/database_1.npz` (Hipparcos catalog, 68.3° field of view, magnitude 6.5).
For another lens, see [Other cameras and IMUs](#other-cameras-and-imus).

The versions used in October 2026: cedar-solve 0.5.1 (commit `191b8f6`);
cedar-detect built from its repository in May 2026 (exact commit not
recorded; any recent commit should work).

### 4. Earth rotation data (astropy)

The longitude needs UT1, the real Earth rotation time. astropy takes the
difference UT1−UTC from IERS tables that it downloads. **A table that is too
old gives a longitude error of up to 0.25 nmi**, and at sea the Pi often has
no internet. Before you leave port, refresh the table, and allow astropy to
use predictions for up to one year:

```bash
mkdir -p ~/.astropy/config
printf '[utils.iers.iers]\nauto_max_age = 365\n' >> ~/.astropy/config/astropy.cfg
~/pisextant-hw/bin/python -c "from astropy.utils import iers; iers.IERS_Auto.open()"
```

Do the same on the laptop that you use for calibration, so that both compute
the same Earth rotation.

### 5. Web interface at boot

```bash
cd ~/PiSextant
cp webui/webui_config.example.json webui/webui_config.json
# edit webui_config.json: put your contact in tile_user_agent (map tile servers ask for it)
sed -i "s/YOUR_USER/$USER/g" systemd/pisextant-webui.service
sudo cp systemd/pisextant-webui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pisextant-webui
```

Open `http://<pi-address>:8080` in a browser on the same network (for example
a phone hotspot). The map works offline from tiles that you download in
advance with the "Tiles" panel.

## Calibration

The calibration lives in two files in `configs/`. They belong to your
instrument, so git ignores them.

| File | Content | Made by |
|---|---|---|
| `configs/m_case.json` | rotation from the IMU frame to the camera frame | `fit_mcase.py` |
| `configs/r_cal_ism330dhcx.json` | accelerometer model: matrix `A`, offset `b`, temperature term `c` | `calibrate_affine.py`, then `rezero.py` |

The model is

```
zenith_in_camera = normalize( A @ (M_case @ g) + b + c * (T - T0) )
```

where `g` is the measured acceleration and `T` the IMU temperature.

All calibration steps need a **clear sky** and a **known position**: either a
fixed place known to ~200 m (pass `--position=LAT,LON`), or a GPS track in
each run folder as `gps.jsonl` (see
[Checking a run](#checking-a-run-against-a-known-track)). The fitting scripts
run on a laptop: copy the run folders from the Pi (`runs/<run>/`) first.

### Step 1: mounting rotation (M_CASE)

Without `configs/m_case.json`, the capture uses the value of the reference
enclosure. If your boards are mounted differently, measure it:

1. Record two runs of about 10 minutes at the known position: one with the
   enclosure flat, one with one side lifted by 20-30°.
2. On the laptop:

   ```bash
   python fit_mcase.py runs/<flat> runs/<tilted> --lat 48.8566 --lon 2.3522
   ```

   It writes `configs/m_case.json`. Copy it to the Pi.

### Step 2: accelerometer calibration

The accelerometer has gains, cross-axis errors and an offset. They change the
fix by tens of nmi and depend on the orientation, so this step needs several
orientations.

1. In one session, at the known position, record **6 to 9 runs of 5-10
   minutes** with the enclosure in clearly different orientations: flat,
   tilted 20-50° toward different sides, and rolled about the camera axis.
   Change the orientation by moving the whole enclosure, never by pressing
   the camera or the IMU.
2. For the temperature term, start with a cold instrument: the capture load
   warms the Pi by a few °C in the first 20 minutes.
3. On the laptop:

   ```bash
   python calibrate_affine.py runs/<r1> runs/<r2> ... --position=48.8566,2.3522 \
       --temp --out configs/r_cal_ism330dhcx.json
   ```

   It prints the error per run and a "leave one orientation out" test, which
   is the honest number for a new orientation. Copy the file to the Pi.

The capture reads both files at its start. A run without
`r_cal_ism330dhcx.json` is uncalibrated, and the webui checklist shows it.

### Step 3: re-zero (after a knock, or once installed on the ship)

A knock or a new mount can turn the camera relative to the IMU by a small
angle. The gains and the offset stay the same, so you only need to measure this
rotation:

- **At a known fixed position:** one flat run and one tilted run (~30°):

  ```bash
  python rezero.py runs/<flat> runs/<tilted> --cal configs/r_cal_ism330dhcx.json \
      --position=48.8566,2.3522 --out configs/r_cal_ism330dhcx.json.new
  ```

- **At sea, with the ship's GPS track** in each run's `gps.jsonl`: one or more
  runs of 1-4 hours, without `--position`. Runs on two nights, with the ship on
  different headings, give a better result than one run.

  ```bash
  python rezero.py runs/<night1> runs/<night2> --cal configs/r_cal_ism330dhcx.json \
      --out configs/r_cal_ism330dhcx.json.new
  ```

Then replace the file on the Pi (keep a copy of the old one).

### Optional: multi-heading correction

An error that is fixed in the instrument (a calibration residual, a small
shift) moves the fix in a direction that turns with the instrument. With runs
at several headings (at least 60° apart, on both sides),
`multiheading_fix.py` separates this error from the position. At a berth, with
5 headings, it gave 0.42 nmi where single runs gave 3.5-4.9 nmi.

```bash
python multiheading_fix.py runs/<r1> runs/<r2> runs/<r3> [--dr-from-gps]
```

## Running

### From the web page

1. Wait about 10 minutes after power-on: the checklist shows when the IMU
   temperature is stable.
2. Check the chips at the top: the clock shows `NTP` (best) or `RTC` (yellow,
   good to about 1 s per week).
3. Start the **solver**, wait about 30 s, then start the **capture** with a
   duration and an image rate ("save 1 image in N"; images are only for
   offline work).
4. The map shows each fix (dots) and the 60-second average (line). The 60 s
   average is the position to use.

### From the command line (for example over ssh)

```bash
cd ~/PiSextant
~/pisextant-solver/bin/python solve_worker.py configs/solver.json &
sleep 30
~/pisextant-hw/bin/python capture_run_wave.py --duration-min 240 --png-every 30
```

Or through the web server, which keeps the map live:

```bash
curl -s -X POST localhost:8080/api/proc/start -H 'Content-Type: application/json' -d '{"name":"solver"}'
curl -s -X POST localhost:8080/api/proc/start -H 'Content-Type: application/json' \
     -d '{"name":"capture","duration_min":240,"png_every":30}'
curl -s localhost:8080/api/status
curl -s -X POST localhost:8080/api/proc/stop -H 'Content-Type: application/json' -d '{"name":"capture"}'
```

Ctrl-C or the stop button finishes the run cleanly.

Main capture options: `--duration-min`, `--png-every N` (default 10; all
images cost about 125 MB per minute), `--smooth-s` (averaging window, default
60 s). `--wave-s` (gyro-based wave compensation) is off by default: the gyro
offset of the ISM330DHCX is not corrected yet and makes it worse.

## Checking a run against a known track

Put the truth in the run folder as `gps.jsonl`, one JSON object per line:

```json
{"type": "fix", "t_gps": 1792152000, "lat": 22.40240, "lon": 125.27118}
```

`t_gps` is the UTC time in Unix seconds. The tools interpolate linearly between
the points, so a few points on a straight course are enough; add points at the
start and the end to cover the whole run. `nmea_logger.py` writes this file
from a GPS on a serial port.

```bash
python sea_eval.py runs/<run>                  # offset, 60 s fix error, table per 10 min
python sea_eval.py runs/<run> --bin-min 30
python drift_check.py runs/<run> --position=48.8566,2.3522   # drift with temperature / time
```

## Files and folders

### Scripts

| Script | Runs on | Role |
|---|---|---|
| `capture_run_wave.py` | Pi | camera + IMU + fixes, writes the run folder |
| `solve_worker.py` | Pi | star detection and solving, serves the capture over a Unix socket |
| `imu_backends.py` | Pi | ISM330DHCX driver |
| `webui/server.py` | Pi | web interface and process control |
| `set_clock_from_gps.py` | Pi | sets the system clock and the RTC from a serial GPS |
| `fit_mcase.py` | laptop | calibration step 1 (mounting rotation) |
| `calibrate_affine.py` | laptop | calibration step 2 (accelerometer model) |
| `rezero.py` | laptop | calibration step 3 (re-zero) |
| `calibrate_rcal.py` | laptop | shared calibration helpers (zenith model, rotation fit) |
| `multiheading_fix.py` | laptop | multi-heading correction |
| `sea_eval.py` | laptop | fix error of a run against `gps.jsonl` |
| `drift_check.py` | laptop | drift with temperature and time |
| `replay_solve.py` | laptop | re-solves the saved images of a run with another solver config |
| `build_database.py` | laptop | builds a star pattern database for another field of view |
| `nmea_logger.py` | laptop | logs a serial GPS as `gps.jsonl` |

### A run folder

```
runs/<YYYY-MM-DD_HH-MM-SS>/        (Pi local time)
  run_meta.json          camera, IMU, calibration, solver config, clock state, end-of-run totals
  capture.jsonl          per frame: exposure times, camera metadata, IMU window
  imu.jsonl              every IMU sample (~80 Hz): t, gravity (m/s^2), gyro (rad/s)
  temp.jsonl             IMU temperature every 5 s
  solves/live-<config>.jsonl   per frame: star spots, solution, gravity used, fix, 60 s fix
  images/frame_NNNNN.png       saved frames (1 in N)
```

Everything needed to recompute or recalibrate is in this folder. A 3-hour run
writes about 0.5 GB of logs (the solve log keeps every detected star spot).

## Other cameras and IMUs

Current state:

- **Calibration is generic.** `fit_mcase.py`, `calibrate_affine.py` and
  `rezero.py` work for any camera and any accelerometer: they only use the
  logged gravity vector and the star solutions.
- **The IMU driver is one small file**, `imu_backends.py`: `make_imu()` must
  return an object with `.acceleration` (m/s², 3 values), `.gyro` (rad/s) and
  `.temperature` (°C). Another I²C IMU needs a new version of this function and
  a new calibration. The calibration file name `r_cal_ism330dhcx.json` is
  still fixed in the capture script.
- **The camera settings are fixed** in `capture_run_wave.py` (sensor mode,
  300 ms exposure, gain 12, focus at infinity) for the IMX708. Another camera
  needs these settings, a star database for its field of view
  (`build_database.py`, needs the Hipparcos catalog file `hip_main.dat`), and
  the field of view in `configs/solver.json` (`fov_estimate`).

A cleaner way to select the sensors is planned.

## Known limits

- **Shocks** shift the camera-IMU geometry (0.1° = 6 nmi). Re-zero after any
  knock.
- **Temperature**: the IMU offset changes with temperature. The temperature
  term reduces this to about 1 nmi per °C; a bench calibration over a wider
  temperature range is planned. The Pi's own heat warms the IMU.
- **Waves**: one frame can be 10-100 nmi off in a rough sea. The 60 s average
  handles normal seas; in rough seas use a longer average. Fewer frames solve
  when the stars blur.
- **False star solves**: rarely, a frame matches a wrong star pattern and is
  thousands of nmi off. The 60 s average ignores fixes more than 5° from the
  window median.
- **Clock**: 1 s of clock error gives 0.25 nmi of longitude. Keep the RTC set
  and the IERS table fresh.
- **Clouds, moon and lights** reduce or stop the star solves.

## Credits and licenses

- Star solving: [cedar-solve](https://github.com/smroid/cedar-solve) (Apache
  2.0), Steven Rosenthal's fork of [tetra3](https://github.com/esa/tetra3)
  (ESA, Apache 2.0); star detection:
  [cedar-detect](https://github.com/smroid/cedar-detect) (Steven Rosenthal).
- Star catalog: Hipparcos (ESA), in `data/database_1.npz`.
- Time and Earth orientation: [astropy](https://www.astropy.org) and
  [ERFA](https://github.com/liberfa/erfa).
- Map: [Leaflet](https://leafletjs.com) 1.9.4 (BSD 2-Clause, see
  `webui/static/vendor/LEAFLET-LICENSE`); map data © OpenStreetMap
  contributors.
- IMU driver: Adafruit CircuitPython LSM6DS.

PiSextant is released under the MIT License, see [LICENSE](LICENSE).
Copyright (c) 2026 Gaspard de Francony. The third-party parts keep their own
licenses: Leaflet (BSD 2-Clause, `webui/static/vendor/LEAFLET-LICENSE`) and
the two files in `patches/`, which modify cedar-solve and stay under its
Apache 2.0 license.
