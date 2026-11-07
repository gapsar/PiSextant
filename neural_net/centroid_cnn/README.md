# Star-centroid CNN (research, NOT finished)

Status 2026-11-08: steps 0-9 done, step 10 (running on the Pi) started.
The model is **too slow for the Pi Zero 2 W so far** and it does not change
the position fix yet. Nothing here is used by the instrument on `main`.

## Goal

The star solver needs the centre of each star spot (the centroid). On a
moving ship the spots are short trails, and the classical centroid has a
bias along the trail. A small CNN gets the image patch of one star plus a
"gyro channel" (the motion of the camera during the exposure) and gives the
centroid and its uncertainty. A requirement: no over-fitting to this lens,
camera or IMU (inputs in noise units, wide randomized ranges).

## Steps and results

| Step | What | Result |
|---|---|---|
| 0-1 | Conventions and real-image facts | pixel-centre convention measured; background 2.0 DN, noise 1.48 DN, PSF sigma ~3 px (`CONVENTIONS.md`) |
| 2 | Classical centroid error vs SNR | bias source found (missing background term in the weighted fit) |
| 3 | Measure the camera for the generator | background noise, hot pixels, PSF shape (narrower at the edges), tone curve, flux distribution (`configs/step3.json`) |
| 4 | Motion during the exposure | trail signs and axes checked on real trails; gyro channel defined (`configs/step4.json`) |
| 5 | Synthetic patch generator | 40 % instrument / 60 % wide ranges; checks pass; in an eye test the user could not tell synthetic stars from real ones |
| 6 | Classical baselines | best = `gyro_fit` (median 0.43 px, 95 % 2.95 px, 10.9 % failures) |
| 7 | Train the CNN | 40k parameters, 4.6 M multiply-adds; 1 M training patches; hybrid with `gyro_fit` for the brightest 6 % of stars |
| 8 | Synthetic evaluation | better than `gyro_fit` in total rms and failures in every bin; core precision equal except very bright stars, long trails and close neighbours; weak on PSF < 0.8 px |
| 9 | Real sea images (4 nights) | star-solve RMSE -14..-23 %; solve rate and false solves unchanged; **fix scatter unchanged** (the wave error dominates) |
| 10 | Export for the Pi | ONNX and numpy (`src/pi_cnn.py`, the Pi has no onnxruntime) equal to torch within 5e-6 px; laptop numpy 0.41 ms per patch -> Pi estimate **0.4-0.8 s per frame, budget ~0.1 s**; the input preparation (~0.9 s per frame on the laptop) costs more |

## Next

1. Measure the real time on the Pi: `python src/pi_cnn.py --bench export/cnn_weights.npz`.
2. Make the input preparation faster, and run the CNN only on the stars
   that the solver uses.
3. Decide if the gain is worth the time per frame.

## Folder

- `src/`: one script per step (generator, measurements, training,
  evaluation, export, numpy CNN for the Pi).
- `configs/`: the measured values and the settings of each step.
- `run_step*.sh`: the long offline runs (log to files).
- `reports/stepN/`: summaries and plots of each step.
- `CONVENTIONS.md`: axes, pixel centre, units, channels.

Not in git (local only): `datasets/` (regenerate with `run_step5.sh`;
~7 GB), `checkpoints/`, `export/` weights, `scratch/`, and the internal
work notes (work log, guide and lessons for the working model, handoff).
Some code comments name those notes ("WORKLOG", "LESSONS", "HANDOFF",
"review item"); they are not published.
