# PiSextant

**A standalone celestial position system on a Raspberry Pi.** A camera looks at
the night sky, an accelerometer measures the local vertical, and the Pi computes
its latitude and longitude continuously, with no GPS. It is made to be fixed on
a ship: you calibrate it once, and then it runs alone and shows its position
and track in a web page.

<!-- PHOTO: the assembled instrument -->

> Status: working prototype. At sea in October 2026, the 60-second fix had a
> median error of 1.6 nmi over a 4-hour night ([details](docs/how-it-works.md#accuracy)).

## What you need

- Raspberry Pi Zero 2 W, Raspberry Pi Camera Module 3 (standard lens),
  Adafruit ISM330DHCX IMU, DS3231 RTC module, an enclosure.
- Parts list, wiring and mounting rules: [docs/hardware.md](docs/hardware.md).

## Install

On the Pi (Raspberry Pi OS, Debian 13 "Trixie", 64-bit). The reasons for each
step are in [docs/installation.md](docs/installation.md).

**1. System, clock and repository**

```bash
sudo raspi-config nonint do_i2c 0
sudo apt install git python3-venv python3-picamera2 i2c-tools
echo "dtoverlay=i2c-rtc,ds3231" | sudo tee -a /boot/firmware/config.txt
git clone <this repository> ~/PiSextant
sudo reboot
```

After the reboot, check the sensors with `i2cdetect -y 1` (`6a` = IMU,
`68` = RTC). When the Pi has network, save the time in the RTC:
`sudo hwclock -w`.

**2. Capture venv (camera + IMU)**

```bash
python3 -m venv --system-site-packages ~/pisextant-hw
~/pisextant-hw/bin/pip install -r ~/PiSextant/requirements/pi-hw.txt
```

**3. Solver venv (cedar-solve + cedar-detect)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 ~/pisextant-solver
uv pip install --python ~/pisextant-solver -r ~/PiSextant/requirements/pi-solver.txt

git clone https://github.com/smroid/cedar-solve ~/cedar-solve
cd ~/cedar-solve && git checkout 191b8f6
git apply ~/PiSextant/patches/cedar-solve-pillow9.patch
git apply ~/PiSextant/patches/tetra3-trimmed-centroids.patch
uv pip install --python ~/pisextant-solver .

# star detector (Rust, https://rustup.rs); cross-compiling on a laptop is faster
git clone https://github.com/smroid/cedar-detect ~/cedar-detect
cd ~/cedar-detect && cargo build --release --bin cedar-detect-server
cp target/release/cedar-detect-server ~/cedar-solve/tetra3/bin/
```

**4. Earth rotation data** (needs internet; repeat before each long trip)

```bash
mkdir -p ~/.astropy/config
printf '[utils.iers.iers]\nauto_max_age = 365\n' >> ~/.astropy/config/astropy.cfg
~/pisextant-hw/bin/python -c "from astropy.utils import iers; iers.IERS_Auto.open()"
```

**5. Web interface at boot**

```bash
cd ~/PiSextant
cp webui/webui_config.example.json webui/webui_config.json   # put your contact in tile_user_agent
sed -i "s/YOUR_USER/$USER/g" systemd/pisextant-webui.service
sudo cp systemd/pisextant-webui.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pisextant-webui
```

## Get started

1. Open `http://<pi-address>:8080` in a browser on the same network (for
   example a phone hotspot).
2. Put the instrument under a clear sky and wait about 10 minutes: the
   checklist shows when the IMU temperature is stable.
3. Start the **solver**, wait about 30 s, then start the **capture**.
4. The map shows each fix (dots) and the 60-second average (line). The 60 s
   average is the position to use.

The first fixes are **not calibrated** and can be tens or hundreds of nautical miles off.
To get the accuracy above, calibrate the instrument once at a known position:
[docs/calibration.md](docs/calibration.md). Then fix it on the ship and
re-zero it.

More: [running from the command line](docs/running.md),
[checking a run against GPS](docs/checking-a-run.md), and the
[full documentation](docs/README.md).

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 Gaspard de Francony.
Third-party parts keep their own licenses: see [docs/credits.md](docs/credits.md).
