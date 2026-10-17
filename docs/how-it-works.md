# How it works

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
  6 nmi. A [re-zero](calibration.md#step-3-re-zero-after-a-knock-or-once-installed-on-the-ship)
  (one or two nights with a GPS track) removes it.
- **The DS3231 real-time clock is good enough** to run without internet.

See also [Known limits](limits.md).
