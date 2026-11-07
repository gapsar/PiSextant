#!/usr/bin/env python3
"""Star-anchored IMU mean -- the navigation output fix_nav of
capture_run_wave.py since v2.5 (logged only in v2.4).

The live fix uses the gravity of the exposures of the solved frames only.
At sea, that gated sampling leaves a slow error that the 60 s mean of the
fixes does not remove (docs/wave-work.md). This class
uses ALL IMU samples instead:

1. Between two solved frames less than MAX_GAP_S apart, the attitude of
   every IMU sample comes from the gyro (offset-corrected), started at the
   first frame's star attitude; the mismatch with the second frame's star
   attitude is spread linearly over the gap. Gaps are short, so the gyro
   offset does not matter.
2. The gyro offset itself comes from the stars: rotation between two solved
   frames (< PAIR_GAP_S apart) minus the gyro rotation, divided by the time;
   median of the last BIAS_WINDOW_S.
3. Each sample's calibrated specific force (A M g + b + c dT, as the live
   fix) is carried to the Earth-fixed frame (ICRS -> CIRS -> ERA), averaged
   in 0.1 s bins; the up vector is the mean of the bins of the last
   window_s seconds, and gives a latitude and longitude.

Numpy only (the Pi capture venv has no scipy) and causal (past data only).
"""

import math
from collections import deque

import numpy as np

ERA_RATE_DEG_S = 360.98564736629 / 86400.0


def rotvec_matrix(rv):
    """Rotation matrix of a rotation vector (Rodrigues); rv shape (..., 3)."""
    rv = np.asarray(rv, dtype=float)
    th = np.linalg.norm(rv, axis=-1)[..., None, None]
    k = rv / np.maximum(np.linalg.norm(rv, axis=-1, keepdims=True), 1e-15)
    z = np.zeros(rv.shape[:-1])
    K = np.stack([np.stack([z, -k[..., 2], k[..., 1]], -1),
                  np.stack([k[..., 2], z, -k[..., 0]], -1),
                  np.stack([-k[..., 1], k[..., 0], z], -1)], -2)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def matrix_rotvec(R):
    """Rotation vector of a rotation matrix (inverse of rotvec_matrix)."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * v
    return v * th / (2 * math.sin(th))


class StarImuMean:
    MAX_GAP_S = 5.0          # solved frames further apart: samples not used
    PAIR_GAP_S = 2.0         # pairs used for the gyro offset
    BIAS_WINDOW_S = 300.0    # gyro offset = median of the pairs of the last 5 min
    MAX_MISMATCH_DEG = 0.5   # gyro vs stars at the next frame: else not used
    GATE_DEG = 2.0           # false solve: fix this far from the 60 s median
    BIN_S = 0.1
    MIN_COVER = 0.3          # fraction of the window that must hold samples

    def __init__(self, m_case, cal_a, cal_b, cal_c=None, temp_ref_c=25.0,
                 window_s=60.0):
        self.M = np.asarray(m_case, float)
        self.A = np.asarray(cal_a, float)
        self.b = np.asarray(cal_b, float)
        self.c = None if cal_c is None else np.asarray(cal_c, float)
        self.temp_ref = temp_ref_c
        self.window_s = window_s
        self.samples = deque()        # (t, raw accel, raw gyro), IMU frame
        self.t_last_sample = -math.inf
        self.anchor = None            # (t, ICRS->camera matrix)
        self.pairs = deque()          # (t, raw gyro offset, camera frame)
        self.offset = np.zeros(3)     # rad/s, camera frame
        self.fixes = deque()          # (t, lat, lon) for the false-solve gate
        self.bins = {}                # bin index -> [sum (3), n]
        self.t0 = None
        self.n_intervals = 0
        self.n_rejected = 0

    # ---------------------------------------------------------------- input

    def add_imu(self, rows):
        """rows: (t, accel, gyro) of the new IMU samples, time order."""
        for t, g, w in rows:
            if t > self.t_last_sample:
                self.samples.append((t, g, w))
                self.t_last_sample = t
        if self.t0 is None and self.samples:
            self.t0 = self.samples[0][0]
        # no solved frame for a while (clouds): older samples can never be
        # used (gap > MAX_GAP_S), so the buffer stays small on the Pi
        while self.samples and self.samples[0][0] < self.t_last_sample - self.MAX_GAP_S - 2.0:
            self.samples.popleft()

    def add_solved_frame(self, t, r_att, lat, lon, era_deg, c2i, temp_c=None):
        """A solved frame: exposure-mid time, star attitude (ICRS -> camera),
        its instant fix (for the false-solve gate), ERA and ICRS->CIRS matrix
        at t, IMU temperature. Processes the samples since the last solved
        frame."""
        if not self._gate(t, lat, lon):
            return
        prev, self.anchor = self.anchor, (t, np.asarray(r_att, float))
        if prev is None or t - prev[0] >= self.MAX_GAP_S:
            self._drop_samples(t)
            return
        t_i, C_i = prev
        rows = [s for s in self.samples if t_i <= s[0] < t]
        self._drop_samples(t)
        if len(rows) < 3:
            return
        ts = np.array([r[0] for r in rows])
        g = np.array([r[1] for r in rows], float)
        w = np.array([r[2] for r in rows], float) @ self.M.T - self.offset
        # G[k]: camera frame at the first sample -> camera frame at sample k
        dt = np.diff(ts)
        steps = rotvec_matrix(-w[:-1] * dt[:, None])
        G = np.empty((len(ts), 3, 3))
        G[0] = np.eye(3)
        for k in range(len(dt)):
            G[k + 1] = steps[k] @ G[k]
        C_j = self.anchor[1]
        S = C_j @ C_i.T                          # stars: cam_i -> cam_j
        dt_ij = t - t_i
        rv = matrix_rotvec(S @ G[-1].T)          # what the gyro missed
        if dt_ij < self.PAIR_GAP_S and np.linalg.norm(rv) / dt_ij < 0.05:
            self.pairs.append((t, rv / dt_ij + self.offset))
            while self.pairs[0][0] < t - self.BIAS_WINDOW_S:
                self.pairs.popleft()
            self.offset = np.median(np.array([p[1] for p in self.pairs]), axis=0)
        E = matrix_rotvec(C_j @ (G[-1] @ C_i).T)
        if np.degrees(np.linalg.norm(E)) > self.MAX_MISMATCH_DEG:
            self.n_rejected += 1
            return
        s = ((ts - t_i) / dt_ij)[:, None]
        Ck = rotvec_matrix(E * s) @ G @ C_i      # ICRS -> camera, per sample
        f = g @ self.M.T @ self.A.T + self.b
        if self.c is not None and temp_c is not None:
            f = f + self.c * (temp_c - self.temp_ref)
        f_icrs = np.einsum("kji,kj->ki", Ck, f)
        f_cirs = f_icrs @ np.asarray(c2i, float).T
        era = np.radians(era_deg + ERA_RATE_DEG_S * (ts - t))
        ce, se = np.cos(era), np.sin(era)
        f_e = np.c_[f_cirs[:, 0] * ce + f_cirs[:, 1] * se,
                    -f_cirs[:, 0] * se + f_cirs[:, 1] * ce, f_cirs[:, 2]]
        idx = np.floor((ts - self.t0) / self.BIN_S).astype(int)
        for i, v in zip(idx.tolist(), f_e):
            b = self.bins.get(i)
            if b is None:
                self.bins[i] = [v.copy(), 1]
            else:
                b[0] += v
                b[1] += 1
        self.n_intervals += 1
        oldest = int((t - self.window_s - 10 - self.t0) / self.BIN_S)
        for i in [i for i in self.bins if i < oldest]:
            del self.bins[i]

    # --------------------------------------------------------------- output

    def fix(self, t):
        """Mean up vector of the last window_s -> dict, or None."""
        if self.t0 is None:
            return None
        lo = (t - self.window_s - self.t0) / self.BIN_S - 0.5
        hi = (t - self.t0) / self.BIN_S - 0.5
        means = [s / n for i, (s, n) in self.bins.items() if lo <= i <= hi and n >= 3]
        if len(means) < self.MIN_COVER * self.window_s / self.BIN_S:
            return None
        u = np.mean(means, axis=0)
        u = u / np.linalg.norm(u)
        return {"latitude": float(math.degrees(math.asin(u[2]))),
                "longitude": float(math.degrees(math.atan2(u[1], u[0]))),
                "n_bins": len(means),
                "coverage": round(len(means) * self.BIN_S / self.window_s, 3),
                "gyro_offset_dps": [round(float(x), 4) for x in np.degrees(self.offset)],
                "window_s": self.window_s}

    # -------------------------------------------------------------- helpers

    def _gate(self, t, lat, lon):
        self.fixes.append((t, lat, lon))
        while self.fixes[0][0] < t - 60.0:
            self.fixes.popleft()
        if len(self.fixes) < 10:
            return True
        la = np.array([f[1] for f in self.fixes])
        lo = np.array([f[2] for f in self.fixes])
        lo = (lo - lo[0] + 180.0) % 360.0 - 180.0 + lo[0]
        d = math.hypot(lat - np.median(la),
                       ((lon - np.median(lo) + 180.0) % 360.0 - 180.0)
                       * math.cos(math.radians(lat)))
        return d <= self.GATE_DEG

    def _drop_samples(self, t):
        while self.samples and self.samples[0][0] < t:
            self.samples.popleft()
