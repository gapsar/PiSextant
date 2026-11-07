# Wave work: what limits the fix at sea, and what changed

Work done in October-November 2026 on the sea runs of the reference build
(instrument on the deck of a ship, 10 nights, ship GPS as truth). Result:
capture **v2.5**, whose navigation output `fix_nav` is the **star-anchored
mean of all IMU samples**. This page gives the tests, the numbers and the
decisions.

## 1. The problem

One frame gives one fix with an error of 15-100 nmi, because the ship
accelerates in the waves. Up to v2.4 the navigation output was the 60 s mean
of these fixes (`fix_smoothed`). Around the offset of each night, that mean
still had **1.3-2.4 nmi rms (95 %: 3.3-7.4 nmi)** on normal nights.

The question: what survives the 60 s mean, and how can we remove it?

## 2. What was tested

Each test ran on all sea nights, with the same frames, against the ship GPS.

| Idea | Result | Decision |
|---|---|---|
| Split the error into period bands | Per frame, the < 10 s band (waves) is the largest, but it averages out well. After 60 s, the error with periods of 30 s to 20 min is as large. A real ship acceleration cannot make 1-2 nmi at these periods. | Look for an instrument or sampling cause |
| Smooth window shapes (Hann, triangle) on the fixes | Not better than the plain 60 s mean. A longer window helps (120 s: -15..-38 %). | Window shape: no change |
| **Mean of ALL IMU samples, each turned to the Earth frame with the star attitude and the gyro** | **rms -17..-55 % on every normal night, never worse on any night** | **Adopted (v2.4 logged, v2.5 output)** |
| Wave model (double integration of the acceleration) | A little better on calm nights; fails at 11.7 kn and on short segments | Rejected |
| Attitude from the stars only (no gyro between frames) | Still better than before, but +4..+44 % worse than with the gyro | Gyro kept between frames |
| Why the exposure-only sampling is worse | The accelerometer sees 0.02-0.03 m/s² of vibration at 5-15 Hz; the frames come at 1.66 Hz. Sampling only during the exposures folds this vibration into slow errors (likely, not proven). | The remedy does not depend on the cause |
| Kalman filter / ship motion model | At most the fast third of what is left (~20-30 %); a longer window gives the same | Not done |
| Coriolis / Eötvös | 0.006 nmi on station, 0.12 nmi at 12 kn (22° N) | Ignored |

## 3. Results

60 s error around the night offset, rms / 95 % (nmi). "Before" = the 60 s
mean of the fixes; "after" = the star-anchored mean of all IMU samples
(offline, same frames).

| Night | Before | After |
|---|---|---|
| Oct 13 | 2.25 / 4.12 | **1.20 / 2.12** |
| Oct 15 | 2.97 / 5.84 | **1.59 / 2.71** |
| Oct 16 | 3.09 / 5.67 | **1.42 / 2.55** |
| Oct 17 | 1.75 / 3.32 ¹ | **1.42 / 2.54** |
| Oct 18 | 1.98 / 3.71 | **1.47 / 2.90** |
| Oct 20 (11.7 kn, rough) | 3.89 / 7.18 | **2.45 / 4.48** |
| Oct 21 | 1.87 / 3.12 | **1.54 / 2.75** |
| Oct 24 (one fix + dead reckoning) | 8.75 / 15.9 | **7.30 / 12.2** |
| Oct 25 | 6.32 / 11.7 | **4.67 / 8.47** |
| Oct 26 (clouds) | 4.96 / 8.78 | **3.75 / 6.35** |

¹ Same sampling recomputed; the logged value (9.76 rms) contains one false
solve, from before the false-solve gate existed.

With a **120 s** window: 0.92-1.42 nmi rms and 95 % below 2.5 nmi on the
normal nights (below 2 nmi on four of them).

**Live on the Pi Zero 2 W** (v2.4, logged only, 2 nights): 0 errors,
26 ms per frame, the live value equals the offline replay within 0.12 nmi.
On the second night: 2.56 → 1.53 nmi rms around the offset.

## 4. How the new output works (`star_imu_live.py`)

1. Between two solved frames less than 5 s apart, the attitude of each IMU
   sample (~79 Hz) comes from the gyro, started at the star attitude of the
   first frame. The mismatch with the star attitude of the second frame is
   spread over the gap.
2. The gyro offset (0.15-0.6 °/s on this IMU) comes from the stars: the
   rotation between two close solved frames minus the gyro rotation. After
   this correction, what is left (0.05-0.08 °/s) is the noise of the star
   attitude.
3. Each calibrated accelerometer sample is turned to the Earth-fixed frame
   and averaged in 0.1 s bins. The mean up vector of the last 60 s gives the
   latitude and the longitude.
4. A fix more than 2° from the 60 s median (a false star solve) is not used
   as an anchor.

`fix_nav` uses this mean when at least 30 % of its window has samples;
otherwise (start of a run, clouds) it uses `fix_smoothed`. Its `source`
field tells which. The window is `NAV_WINDOW_S` in `capture_run_wave.py`
(or `--nav-s`): 60 s by default; 120 s is smoother and lags 60 s more
(nothing on station, ~0.2 nmi at 12 kn).

## 5. What is left

After this change, on a normal night (60 s output):

- **Alignment between the camera and the IMU.** The largest error: each
  handling of the box can change it (a knock: ~7 nmi; a fall: ~110 nmi;
  carried to another place: about ±2 nmi from night to night). A stiffer
  enclosure is planned. See [Calibration](calibration.md) for the re-zero.
- **Fast wave error** that leaks through the window edges: ~0.5-1 nmi.
- **Slow drift inside a night: ~0.7 nmi.** The 20 min mean of the wave
  noise alone gives 0.5-1.2 nmi, so the real drift is small. Sea state,
  star solve quality, static tilt, camera elevation and time do not
  predict it. The temperature term of the calibration helps
  (2.47 → 2.21 nmi), but a better term was not consistent from night to
  night. A temperature test on land is planned.

## 6. Lessons for other instruments (hand-held, phone)

- Do not use the gravity of the exposures only. Average all IMU samples,
  each turned to the Earth frame with the star attitude and the gyro.
- The gyro offset comes free from consecutive star attitudes.
- The window length matters; its shape does not.
- Vibration or hand tremor folded by exposure-gated sampling can make the
  same kind of slow error: test it first.
