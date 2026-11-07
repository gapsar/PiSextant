# CONVENTIONS.md — fixed conventions for this folder

Written in step 0 (21 Oct 2026). Every later file in this folder follows
this file. Changes here need a note in WORKLOG.md.

## Axes and indexing

- Arrays are numpy arrays indexed (row, col). row is y (down), col is x (right).
- Logged centroids from the pipeline are (row, col). Verified on 4 frames
  (first coordinate < 1296, second up to 2304; bright pixel at [c0, c1]).
- tetra3's get_centroids_from_image returns (N, 2) arrays ordered (y, x) =
  (row, col). Checked in the source, and the injection test below pairs 20/20
  without swaps.

## Pixel-centre convention (MEASURED, was open before step 0)

- This folder uses the corner convention everywhere: pixel (i, j) covers the
  square [i, i+1) x [j, j+1); its centre is at (i + 0.5, j + 0.5).
- Injection test (see WORKLOG 2026-10-21 and reports/step0/): 20 noise-free
  injected Gaussian stars, random sub-pixel positions, frame_06000.png of the
  15 Oct run, tetra3 with sigma=3 and max_area raised to 1000.
- Real-frame result: mean bias row +0.003 px, mean bias col +0.000 px. Per-trial rms
  0.034 px per axis (noise in the real background under the stars, threshold
  truncation, quantization: zero-mean effects). 5 of 20 stars are more
  than 0.05 px off (worst 0.074 px).
- Flat-background result (--flat, constant 2 DN, no noise): mean bias
  +0.001 / +0.000 px, rms 0.009 px, worst 0.021 px. All 20 within 0.05 px.
  So the real-frame scatter comes from the background noise. Conclusion: tetra3 reports in
  the corner convention. This matches the 16 Oct analysis and the source.
- NOT measured: cedar-detect (the production detector, gRPC server not
  available on this laptop). Open question for Claude. The 16 Oct analysis
  says it agrees; the guide required this test on tetra3 only.
- MEASURED 30 Oct (step 9, src/real_eval.py, 311 131 real stars of 17 + 18
  Oct): CNN centroid - cedar-detect centroid, mean +0.04 row / +0.02 col px
  (median +0.03 / +0.01). A centre-convention detector would give about
  -0.5 px. So cedar-detect ALSO uses the corner convention. Closed.

## Patch centre

- Centre pixel = (floor(row), floor(col)) of the coarse centroid, in corner
  coords: the pixel that contains the coarse centroid. Not round(): with
  round() the star lands up to 1 px from the centre pixel, always on the
  same side. (Changed 21 Oct after the step 0 review, see WORKLOG.)
- Patch top-left = centre pixel - 16 on both axes, so the centre pixel is
  at patch-local index 16. Its centre (the point the model's outputs
  refer to) is at patch-local corner coords (16.5, 16.5).
- The model output (d_row, d_col) = the star's mid-time position minus
  (top-left + 16.5), per axis, in px, in corner coords.
- The generator and the production code use this same rule.

## PSF shape parameters (added 23 Oct after the step 1 (v2) review, item 5; see WORKLOG)

- angle_deg = 0 puts the major axis along +col. A positive angle turns it
  toward +row, which is clockwise on the screen, because row goes down.
  The rule is the same for both shapes.
- major_px is sigma for a Gaussian and alpha for a Moffat. axis_ratio is
  major over minor, at least 1.

## SNR (added 23 Oct, step 2, review note a — written before any plot)

- The SNR of a patch in this folder is: SNR = F / (sigma_pix * sqrt(2 * pi * s * s)).
  F = the star total light, DN (the generator definition, infinite plane).
  sigma_pix = the per-pixel noise std at the patch background, DN.
  s = the PSF sigma, px; for a Moffat, s = the sigma of the Gaussian with
  the same FWHM.
- sigma_pix is the MODEL value from the noise model (a placeholder until
  step 3 fits it), not a measured value.
- Faint-star bound: the formula above gives a bound of about 2 s / SNR per
  axis at the photon-noise limit, against the guide's rough s / SNR
  (section 5). The factor comes from the definition (review 24 Oct, item
  5, working from the earlier derivation).
- Every curve and table in this folder uses this one definition.

## Gyro sign and projection (measured 21 Oct by imu_time_offset.py; confirm
on real trails in step 4 per the guide)

- The star rotation between two frames equals MINUS the gyro integral
  (camera frame). For small rotation theta: delta_s = - theta x s.
- w_cam = M_case · w_imu (first line of the solve log). Camera frame axes:
  (boresight, left, up).
- Pinhole mapping before distortion: col = 1152 - f·s_l/s_b,
  row = 648 - f·s_u/s_b, f ≈ 1700 px. Left = decreasing col, up =
  decreasing row.
- CONFIRMED on real trails (step 4, 28 Oct, src/check_trails.py, 15 and 16
  Oct, 304 stars): among the stars where the test can tell the hypotheses
  apart, the nominal mapping beats mirror_col (99-100%), mirror_row
  (92-98%), swap (97%), roll sign negated (96-100%), and on curved trails
  the opposite rotation sign (91%, 11 stars). Distortion: tetra3
  r_u = r_d (1 - k (2/W)^2 r_d^2) / (1 - k), k ≈ 0.018, about (648, 1152).

## Time reference

- SensorTimestamp = end of the exposure of the first row (measured
  convention, confirmed 21 Oct).
- Label time = MIDDLE of the exposure of the row where the star is
  (rolling shutter). t_line ≈ 18 ms / 1296 rows (estimate, to measure in
  step 3). Global: t_mid = timestamp - 150 ms + row·t_line.
- IMU-to-image offset tau: nominal +10 ms, measured spread -9 to +22 ms
  (six nights), +64 ms outlier on 20 Oct. Generator treats it as random.

## Units

- px, DN, rad/s, s. Channel 1 in units of whole-frame background noise
  (4.1.13). Channels 2 and 3 in pixels. Seeds fixed.

## Channels 2 and 3 (added 29 Oct, step 4, src/motion.py)

- Channel 2 = dwell-time map of the MEASURED path over the star's own 300 ms
  (row-dependent window, rolling shutter), 32 x 32, sum 1. The path's
  MID-TIME point is at the patch centre (16.5, 16.5) corner coords, path
  points at 1 ms steps, bilinear weights (the first moment is exact; the
  splat adds about 1/6 px^2 of variance per axis).
- Channel 3 = per pixel, the dwell-weighted mean time fraction (0 = exposure
  start, 1 = end); 0 where channel 2 is 0.
- Measured rate between gyro samples: LINEAR interpolation (the live code
  must do the same). True motion in the generator: PCHIP through the real
  samples (no overshoot).
- A sample whose measured path goes > 14 px from its mid-time point is
  rejected (0.4% of the real-gyro samples incl. the synthetic share).
