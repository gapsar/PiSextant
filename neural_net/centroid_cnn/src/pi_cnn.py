"""Step 10: the star-centroid CNN in plain numpy, for the Pi (no torch, no
onnxruntime). Written by Claude, 6 Nov 2026.

Weights: export/cnn_weights.npz (written by src/export_model.py).
Input X (n, 3, 32, 32) float32, as real_eval.build_inputs:
  ch1 = (Y - median of the outer 3-px frame) / sigma_frame
  ch2 = dwell map of the measured gyro path (sum 1), ch3 = time ramp (0..1)
  patch top-left = floor(coarse centroid) - 16.
Output: mu (n, 2) = (d_row, d_col) px of the mid-time position from the
patch point 16.5 (centroid = r0 + 16.5 + d_row, c0 + 16.5 + d_col);
log_sigma (n, 2), clamped to -4..4.

Timing on the Pi (soft-env has numpy):
    ~/soft-env/bin/python pi_cnn.py --bench cnn_weights.npz
"""
import sys
import time

import numpy as np


class CentroidCnn:
    def __init__(self, path):
        w = np.load(path)
        self.k1, self.s2 = float(w["k1"]), float(w["s2"])
        self.in_ch = int(w["in_ch"])
        self.convs = []
        i = 1
        while f"c{i}_w" in w.files:
            # weight (out, in, k, k) -> (k * k * in, out) for one matmul on
            # channels-last windows (fewer array copies than channels-first)
            wt = w[f"c{i}_w"].astype(np.float32)
            self.convs.append((wt.transpose(2, 3, 1, 0).reshape(-1, wt.shape[0]).copy(),
                               w[f"c{i}_b"].astype(np.float32),
                               wt.shape[2], int(w[f"c{i}_stride"])))
            i += 1
        self.f1 = (w["f1_w"].T.astype(np.float32).copy(), w["f1_b"].astype(np.float32))
        self.f2 = (w["f2_w"].T.astype(np.float32).copy(), w["f2_b"].astype(np.float32))

    @staticmethod
    def conv(x, wmat, b, k, stride):
        """x (n, h, w, c) channels-last -> relu(conv), padding k // 2."""
        p = k // 2
        if p:
            x = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)))
        n, h, w, c = x.shape
        ho, wo = (h - k) // stride + 1, (w - k) // stride + 1
        cols = np.empty((n, ho, wo, k, k, c), np.float32)
        for dr in range(k):
            for dc in range(k):
                cols[:, :, :, dr, dc] = x[:, dr:dr + stride * ho:stride, dc:dc + stride * wo:stride]
        y = cols.reshape(n * ho * wo, k * k * c) @ wmat + b
        return np.maximum(y, 0).reshape(n, ho, wo, -1)

    def __call__(self, X):
        X = np.asarray(X, np.float32)[:, :self.in_ch]
        ch = [np.arcsinh(X[:, 0:1] / self.k1)]
        if self.in_ch > 1:
            ch += [X[:, 1:2] * self.s2, X[:, 2:3]]
        x = np.concatenate(ch, 1).astype(np.float32).transpose(0, 2, 3, 1)
        for wmat, b, k, st in self.convs:
            x = self.conv(x, wmat, b, k, st)
        x = x.transpose(0, 3, 1, 2)               # torch flattens (c, h, w)
        h = np.maximum(x.reshape(len(x), -1) @ self.f1[0] + self.f1[1], 0)
        out = h @ self.f2[0] + self.f2[1]
        return out[:, :2], np.clip(out[:, 2:], -4, 4)


def bench(path, n=95, repeat=20):
    """CPU time for one frame of n patches (random input, same shapes)."""
    net = CentroidCnn(path)
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n, 3, 32, 32)).astype(np.float32)
    net(X)                                    # warm-up
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        net(X)
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1000
    print(f"{n} patches per frame: median {np.median(t):.1f} ms, min {t.min():.1f} ms, "
          f"max {t.max():.1f} ms -> {np.median(t) / n:.3f} ms per patch (numpy {np.__version__})")
    return float(np.median(t))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--bench":
        bench(sys.argv[2])
    else:
        print(__doc__)
