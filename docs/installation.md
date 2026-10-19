# Installation details

The [README](../README.md#install) gives the commands. This page explains them.

Tested on Raspberry Pi OS (Debian 13 "Trixie", 64-bit) on a Pi Zero 2 W.

## The three programs

| Program | Python | Role |
|---|---|---|
| `webui/server.py` | system `python3`, standard library only | web page, starts and stops the two others |
| `solve_worker.py` | venv `~/pisextant-solver` (Python 3.11) | star detection and plate solving |
| `capture_run_wave.py` | venv `~/pisextant-hw` (system Python 3.13) | camera, IMU, fix computation, all logs |

The two venvs are separate because the camera stack (picamera2, from apt) needs
the system Python 3.13, while the solver needs numpy < 2, which exists only
for Python 3.12 and older. The versions in `requirements/` are the ones of the
reference instrument.

## Capture venv

`python3 -m venv --system-site-packages` makes picamera2 and Pillow (from apt)
visible in the venv. Without this option, the capture cannot open the camera.

## Real-time clock (DS3231)

With `dtoverlay=i2c-rtc,ds3231` in `/boot/firmware/config.txt`, the kernel sets
the system time from the DS3231 at boot, before any network. When the Pi has
network (NTP synchronized), write the good time into the RTC with
`sudo hwclock -w`. The DS3231 drifts by about 1 s per week; 1 s of clock error
gives 0.25 nmi of longitude error.

If the kernel log shows `rtc-ds1307 ... SET TIME!` at each boot, the RTC still
has its "oscillator stopped" flag. The time can still be correct;
`sudo hwclock -w` clears the flag.

## Solver: cedar-solve and cedar-detect

The solver is [cedar-solve](https://github.com/smroid/cedar-solve) (a fork of
ESA's tetra3) with the [cedar-detect](https://github.com/smroid/cedar-detect)
star detector. cedar-detect is a Rust program; its executable
`cedar-detect-server` must be built for the Pi (aarch64) and placed in
cedar-solve's `tetra3/bin/` folder.

The solver venv uses Python 3.11, installed with
[uv](https://docs.astral.sh/uv/), which downloads its own Python.

Building Rust on a Pi Zero 2 W is slow and needs swap; cross-compiling for
aarch64 on a laptop is faster. `solve_worker.py` looks for the executable in
tetra3's `bin/` folder; set the environment variable `CEDAR_DETECT_BINARY` to
use another path.

**Why the patches:**

- `cedar-solve-pillow9.patch` allows Pillow 9 (cedar-solve asks for Pillow < 9,
  which is not available for this setup).
- `tetra3-trimmed-centroids.patch` (4 lines) protects against an `IndexError`:
  the solver configuration uses up to 95 star spots, more than the 75
  verification stars of the database, and tetra3 can then use an index past
  the end of the trimmed list. The reference instrument ran without it with
  no such crash seen in 46,000 frames, so it is a protection only.

The star pattern database must match your camera's field of view. The
reference build's database is included: `examples/reference-build/database.npz`
(Hipparcos catalog, 68.3° field of view, magnitude 6.5). For another camera or
lens, build one with `build_database.py` (see [Other cameras and IMUs](other-sensors.md)).

The versions used in October 2026: cedar-solve 0.5.1 (commit `191b8f6`);
cedar-detect built from its repository in May 2026 (exact commit not
recorded; any recent commit should work).

## Earth rotation data (astropy)

The longitude needs UT1, the real Earth rotation time. astropy takes the
difference UT1−UTC from IERS tables that it downloads. **A table that is too
old gives a longitude error of up to 0.25 nmi**, and at sea the Pi often has
no internet. Before you leave port, refresh the table. The setting
`auto_max_age = 365` lets astropy use the table's predictions for up to one
year without a new download.

Do the same on the laptop that you use for calibration, so that both compute
the same Earth rotation.

## Web interface

`webui/webui_config.json` sets the Python of each venv, the solver config and
the map tile server. Map tile servers ask for a contact in `tile_user_agent`.
The map works offline from tiles that you download in advance with the
"Tiles" panel. The systemd unit starts the web interface at boot; it needs
only the system `python3`.
