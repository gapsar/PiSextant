# Known limits

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
