"""Online wave filter: learns, during the run, how the ship's rotation
predicts the false tilt that wave acceleration puts into the gravity vector,
and removes the predicted part from each fix. Pure numpy; used live by
capture_run_wave.py and offline by wave_learner_replay.py (same code, same
result).

Why it works: the false tilt (wave acceleration seen as a tilt) is at the
wave period, and the ship's sway follows its roll with a lag that a linear
filter of the past gyro rates captures. The relation changes from night to
night (swell period and direction), so nothing is stored: the filter is
retrained every few minutes on the last minutes of the run.

No truth is needed. The training target of a frame is its tilt relative to
the mean zenith of the frames within +-HP_S/2 (Earth-fixed frame): the ship
moves too little in two minutes to matter, while the wave part is removed
by that mean. A frame is used for training only once its window is
complete, i.e. HP_S/2 later.

    feats = GyroFeatures()
    filt = OnlineTiltFilter()
    feats.add(t, w_cam)                       # every IMU sample (camera frame, rad/s)
    up_corr = filt.add_frame(t, feats.features(t), up_cam, cam_from_earth)
"""

from collections import deque

import numpy as np

BIN_S = 0.25            # gyro averaged per bin
N_BINS = 40             # 10 s of past gyro
BIAS_S = 300.0          # trailing mean = gyro zero-rate offset (ship rotation ~zero-mean)
GUARD_RAD_S = np.radians(10.0)   # instrument handled by hand: no features
TRAIN_S = 1200.0        # train on the last 20 min
RETRAIN_S = 120.0       # retrain every 2 min
HP_S = 120.0            # target = tilt relative to the centered 2 min mean zenith
MIN_TRAIN = 300         # frames needed before the first model
RIDGE = 100.0           # on standardized features
GATE_RAD = np.radians(2.0)       # false solves left out of training and the mean


class GyroFeatures:
    """Causal features from the gyro: mean rate per BIN_S bin over the last
    N_BINS bins, minus the trailing BIAS_S mean (the zero-rate offset)."""

    def __init__(self):
        self.samples = deque()           # (t, w) of the last N_BINS * BIN_S + 1 s
        self.blocks = deque()            # (t_block, sum_w, n) per 1 s block, BIAS_S long
        self._block = None

    def add(self, t, w):
        w = np.asarray(w, dtype=float)
        self.samples.append((t, w))
        while self.samples[0][0] < t - N_BINS * BIN_S - 1.0:
            self.samples.popleft()
        tb = np.floor(t)
        if self._block is None or self._block[0] != tb:
            if self._block is not None:
                self.blocks.append(tuple(self._block))
            self._block = [tb, np.zeros(3), 0]
            while self.blocks and self.blocks[0][0] < t - BIAS_S:
                self.blocks.popleft()
        self._block[1] = self._block[1] + w
        self._block[2] += 1

    def bias(self):
        n = sum(b[2] for b in self.blocks)
        return sum(b[1] for b in self.blocks) / n if n else None

    def features(self, t):
        """Feature vector at time t (rad/s, 3 * N_BINS), or None (gap,
        gyro offset not known yet, or instrument handled)."""
        if len(self.blocks) < 0.5 * BIAS_S:
            return None
        ts = np.array([s[0] for s in self.samples])
        ws = np.array([s[1] for s in self.samples])
        k = (ts > t - N_BINS * BIN_S) & (ts <= t)
        if not k.any() or np.abs(ws[k]).max() > GUARD_RAD_S:
            return None
        idx = np.floor((t - ts[k]) / BIN_S).astype(int)
        counts = np.bincount(idx, minlength=N_BINS)[:N_BINS]
        if (counts == 0).any():
            return None
        sums = np.stack([np.bincount(idx, ws[k][:, j], minlength=N_BINS)[:N_BINS]
                         for j in range(3)], -1)
        return (sums / counts[:, None] - self.bias()).ravel()


class OnlineTiltFilter:
    """Stores the frames of the last TRAIN_S, retrains a ridge regression
    every RETRAIN_S and returns the corrected up vector of each new frame."""

    def __init__(self):
        self.frames = deque()      # (t, feat, up_cam, cam_from_earth)
        self.model = None          # (mu, sd, W)
        self.t_trained = None
        self.n_train = 0
        self.n_models = 0

    def add_frame(self, t, feat, up_cam, cam_from_earth):
        """up_cam: measured up (camera frame, any length); cam_from_earth:
        3x3 rotation Earth-fixed -> camera for this frame. Returns the
        corrected unit up vector, or None while there is no model or no
        features."""
        up = np.asarray(up_cam, dtype=float)
        up = up / np.linalg.norm(up)
        if feat is not None:
            self.frames.append((t, np.asarray(feat), up, np.asarray(cam_from_earth)))
        while self.frames and self.frames[0][0] < t - TRAIN_S - HP_S:
            self.frames.popleft()
        if self.t_trained is None or t - self.t_trained >= RETRAIN_S:
            self._train(t)
        if self.model is None or feat is None:
            return None
        mu, sd, W = self.model
        p = ((np.asarray(feat) - mu) / sd) @ W
        p = p - (p @ up) * up                      # tilt only, no length change
        v = up - p
        return v / np.linalg.norm(v)

    def _targets(self, t_now):
        """Training targets: tilt of each frame relative to the centered
        HP_S mean zenith (Earth-fixed), for the frames whose window is
        complete."""
        t = np.array([f[0] for f in self.frames])
        z = np.array([f[3].T @ f[2] for f in self.frames])       # Earth-fixed zenith
        c = np.vstack([np.zeros((1, 3)), np.cumsum(z, 0)])
        i0 = np.searchsorted(t, t - HP_S / 2)
        i1 = np.searchsorted(t, t + HP_S / 2, side="right")
        zm = (c[i1] - c[i0]) / (i1 - i0)[:, None]
        zm /= np.linalg.norm(zm, axis=1, keepdims=True)
        X, Y = [], []
        for i, (ti, feat, up, M) in enumerate(self.frames):
            if ti > t_now - HP_S / 2 or ti < t_now - TRAIN_S - HP_S / 2:
                continue
            us = M @ zm[i]                                         # smoothed up, camera frame
            if np.arccos(np.clip(up @ us, -1, 1)) > GATE_RAD:
                continue
            X.append(feat)
            Y.append(up - (up @ us) * us)                          # tilt (rad), camera frame
        return np.array(X), np.array(Y)

    def _train(self, t_now):
        self.t_trained = t_now
        if len(self.frames) < MIN_TRAIN:
            return
        X, Y = self._targets(t_now)
        if len(X) < MIN_TRAIN:
            return
        mu, sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - mu) / sd
        W = np.linalg.solve(Z.T @ Z + RIDGE * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
        self.model = (mu, sd, W)
        self.n_train = len(X)
        self.n_models += 1
