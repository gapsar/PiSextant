"""Step 7: train the star-centroid CNN on datasets/v2 (guide step 7).

Written by Claude, 29 Oct 2026. Usage (from the centroid_cnn folder):
    python src/train_cnn.py --selftest            # augmentation unit test (trap 1)
    python src/train_cnn.py --overfit             # one batch of 64 -> near-zero loss
    python src/train_cnn.py --name full           # channels 1-3
    python src/train_cnn.py --name nogyro --channels 1   # ablation for step 8
Outputs: checkpoints/step7_<name>.pt, reports/step7/<name>.txt/.json/.png.
Choices use the val split only; the test split is not opened here.

Decisions (HANDOFF_STEP7 section 5):
1. Input scaling is fixed and inside the model: ch1 -> asinh(ch1 / 4) (linear
   to a few sigma, log above; ch1 p99 of the patch max = 144 sigma), ch2 x 4
   (patch max p50 0.38), ch3 as is. No per-patch normalization (trap 5).
2. Augmentation: the 8 rot/flip symmetries. A flip about the centre pixel
   (16.5) is flip + roll by 1 px (the wrapped row is the old edge row), so the
   label maps exactly d -> -d and the ch2 mid-time point stays at 16.5.
   Tested by --selftest.
3. Output: (d_row, d_col) + log-sigma per axis. Loss = Huber (beta 1 px) on
   the position + 0.2 x Gaussian NLL with the position DETACHED: the sigma
   head learns from the errors but cannot pull the position estimate.
4. Failures: Huber limits their weight; the reports give fail and rms<2 apart.
5. Ablation "nogyro" (channel 1 only), same recipe.
6. Best checkpoint = lowest val Huber loss (position only, no augmentation).
Network: conv3x3 s1 (16) - conv3x3 s2 (32) - conv3x3 s1 (32) - conv3x3 s2 (32)
- 1x1 (8) - flatten 8x8x8 - dense 32 - 4 outputs, ReLU, no BatchNorm, no
pooling (strides only after the first full-resolution layer). About 40 k
parameters and 4.6 M multiply-adds per patch (counted below). The flatten
keeps the position information that a global pool would lose.

CHANGED after the first overfit run (29 Oct): log-sigma floor -6 -> -4 (sigma
>= 0.018 px). With -6 the NLL term made the overfit oscillate (Huber 2e-3..3e-2);
without NLL it reached 1e-9 (cause proven: scratch/overfit_nll_test_claude*).
The floor -4 did NOT fix it (3.6e-3): the NLL competes for the shared layers.
So: --overfit now tests the position path only (nll weight 0; it checks the
model + data pipeline), and a control run "full_nonll" (--nll-weight 0)
measures on val what the sigma head costs the position.

Round 2 (29 Oct): --arch b keeps full resolution for two layers:
conv s1 8 - conv s1 16 - conv s2 32 - conv s1 32 - conv s2 32 - same head;
about 41 k parameters, 5.5 M multiply-adds. Arch a = round 1 (unchanged).
Prediction round 2: b lowers the val median at SNR 50-300 by 0.01-0.02 px
(0.257 -> about 0.24, = gyro_fit) and changes little at SNR < 20; SNR 300+
stays worse than gyro_fit (0.126).

Round 2 result: b = a (val Huber 0.4032 vs 0.4034); SNR 300+ median 0.21 vs
gyro_fit 0.126. Diagnostic (--min-snr 50, train AND val restricted): if a
specialist reaches gyro_fit at SNR >= 50, the limit is the loss weighting
(faint stars dominate the loss); if not, the input transform / network.
Prediction: the specialist reaches 0.24 at SNR 50-300 and about 0.15 at 300+.

Specialist result: 0.262 / 0.197 (= general model): loss weighting is NOT
the cause. Next diagnostics, one change each on the specialist: --fp32 (no
bf16 autocast) and --ch1-k 50 (almost linear ch1). Prediction: one of them
reaches 0.24 / 0.15.

fp32 / k50 result: 0.257 / 0.256 at SNR 50-300, about 0.19 at 300+: neither
is the cause. Diagnostic --arch big (about 0.3 M parameters, about 58 M MAC,
outside the Pi budget) on the specialist: does the Pi budget limit precision?
Prediction: big reaches 0.24 / 0.15.

Round 3 (29 Oct): --train-dataset v3 (1M train; v3 train = v2 train + 900 k
new samples, same seed/config); val stays v2 val (paired with rounds 1-2).
A train set > 2 GB stays in CPU memory; each batch goes to the GPU.
Prediction (arch a, 40 epochs = 2x the sample-steps of 200 epochs on v2):
val Huber 0.40 -> 0.398, SNR 50-300 median 0.257 -> 0.25, 300+ 0.22 -> 0.19;
still behind gyro_fit (0.244 / 0.126) -> the big net is then needed.

Round 4 (29 Oct): arch a on v3, 120 epochs (run_step7d.sh). Prediction: val
Huber 0.387 -> about 0.380 and levels off; SNR 300+ median 0.183 -> about
0.16 (still above gyro_fit 0.126); other bins tie or beat gyro_fit.

Predictions BEFORE the first run:
- selftest: COM of a transformed test star = transformed label within 0.01 px.
- overfit: Huber loss of 64 samples < 1e-3 after 1500 steps.
- full, val: median about 0.30 px (gyro_fit on test 0.43), rms<2 about 0.55,
  fail about 8 % (gyro_fit 10.9 %). Val loss levels off; small train/val gap
  (100k x 8 symmetries for a 40 k-parameter net).
- nogyro: about the same on still stars, clearly worse on trails > 3 px
  (median +30 % or more).
- sigma: |e| / sigma < 1 per axis for 60-75 % of val samples (68 % = calibrated).
- pixel locking (trap 14): mean error vs fractional true position flat within
  0.02 px.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
import check_step5  # noqa: E402
import metrics  # noqa: E402

OUT = HERE / "reports" / "step7"
CKPT = HERE / "checkpoints"
CFG = {"lr": 2e-3, "weight_decay": 1e-4, "batch": 512, "epochs": 60, "huber_beta": 1.0,
       "nll_weight": 0.2, "seed": 7, "ch1_k": 4.0, "ch2_scale": 4.0, "dataset": "v2"}


ARCHS = {"a": [(16, 1), (32, 2), (32, 1), (32, 2)],               # (channels, stride) of the 3x3 convs
         "b": [(8, 1), (16, 1), (32, 2), (32, 1), (32, 2)],
         "big": [(32, 1), (64, 1), (64, 2), (128, 1), (128, 2)]}   # diagnostic only, NOT Pi budget


class Net(nn.Module):
    def __init__(self, in_ch=3, k1=4.0, s2=4.0, arch="a"):
        super().__init__()
        self.in_ch, self.k1, self.s2 = in_ch, k1, s2
        self.convs, c = [], in_ch
        for i, (w, st) in enumerate(ARCHS[arch]):
            self.convs.append(nn.Conv2d(c, w, 3, st, 1))
            c = w
        self.convs.append(nn.Conv2d(c, 8, 1))
        for i, m in enumerate(self.convs):                # names c1, c2, ... (round-1 checkpoints load)
            setattr(self, f"c{i + 1}", m)
        self.f1 = nn.Linear(8 * 8 * 8, 32)
        self.f2 = nn.Linear(32, 4)

    def forward(self, x):
        x = x[:, :self.in_ch].float()
        ch = [torch.asinh(x[:, 0:1] / self.k1)]
        if self.in_ch > 1:
            ch += [x[:, 1:2] * self.s2, x[:, 2:3]]
        x = torch.cat(ch, 1)
        for c in self.convs:
            x = F.relu(c(x))
        out = self.f2(F.relu(self.f1(x.flatten(1))))
        return out[:, :2], out[:, 2:].clamp(-4, 4)     # (d_row, d_col), log-sigma (floor 0.018 px)


def count_macs(model):
    macs, h = 0, 32
    for c in model.convs:
        h = (h + 2 * c.padding[0] - c.kernel_size[0]) // c.stride[0] + 1
        macs += h * h * c.out_channels * c.in_channels * c.kernel_size[0] * c.kernel_size[1]
    macs += model.f1.in_features * model.f1.out_features + model.f2.in_features * model.f2.out_features
    return macs, sum(p.numel() for p in model.parameters())


def dihedral(X, y, k):
    """Symmetry k (0-7) about the centre pixel (16.5, 16.5). bit0: flip rows,
    bit1: flip cols, bit2: transpose. Flip = flip + roll 1 px (exact d -> -d)."""
    y = y.clone()
    if k & 1:
        X, y[:, 0] = X.flip(-2).roll(1, -2), -y[:, 0]
    if k & 2:
        X, y[:, 1] = X.flip(-1).roll(1, -1), -y[:, 1]
    if k & 4:
        X, y = X.transpose(-1, -2), y[:, [1, 0]]
    return X, y


def augment(X, y, gen):
    k = torch.randint(0, 8, (len(X),), device=X.device, generator=gen)
    Xo, yo = torch.empty_like(X), torch.empty_like(y)
    for j in range(8):
        m = k == j
        if m.any():
            Xo[m], yo[m] = dihedral(X[m], y[m], j)
    return Xo, yo


def selftest():
    rng = np.random.default_rng(0)
    y = torch.tensor(rng.uniform(-3, 3, (200, 2)), dtype=torch.float32)
    c = torch.arange(32) + 0.5
    rr, cc = torch.meshgrid(c, c, indexing="ij")
    pr, pc = 16.5 + y[:, 0, None, None], 16.5 + y[:, 1, None, None]
    g = torch.exp(-((rr - pr) ** 2 + (cc - pc) ** 2) / (2 * 1.5 ** 2))
    X = torch.stack([g * 50, g / g.sum((1, 2), keepdim=True), g * cc / 32], 1)
    worst = 0.0
    for k in range(8):
        Xk, yk = dihedral(X, y, k)
        for ch in (0, 1):
            w = Xk[:, ch]
            com = torch.stack([(w * rr).sum((1, 2)), (w * cc).sum((1, 2))], 1) / w.sum((1, 2))[:, None]
            worst = max(worst, float((com - 16.5 - yk).abs().max()))
    ok = worst < 0.01
    print(f"selftest: worst |COM - transformed label| over 8 symmetries = {worst:.5f} px -> {'PASS' if ok else 'FAIL'}")
    return ok


def load(split, dev, ds=None):
    X, y, P = check_step5.load_split(HERE / "datasets" / (ds or CFG["dataset"]), split)
    if CFG.get("min_snr"):
        m = P["snr"] >= CFG["min_snr"]
        X, y, P = X[m], y[m], {k: v[m] for k, v in P.items()}
    assert np.isfinite(X.astype(np.float32)).all() and np.isfinite(y).all(), "non-finite data (trap 10)"
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    if X.nbytes > 2e9:                                    # too big for the GPU: pinned CPU memory
        return Xt.pin_memory(), yt.pin_memory(), P, X
    return Xt.to(dev), yt.to(dev), P, X


def loss_fn(mu, ls, y):
    hub = F.smooth_l1_loss(mu, y, beta=CFG["huber_beta"], reduction="none").sum(1)
    nll = (ls + 0.5 * (mu.detach() - y) ** 2 * torch.exp(-2 * ls)).sum(1)
    return hub.mean(), nll.mean()


@torch.no_grad()
def predict(model, X):
    model.eval()
    mu, ls = [], []
    for i in range(0, len(X), 4096):
        m, s = model(X[i:i + 4096])
        mu.append(m.float()), ls.append(s.float())
    return torch.cat(mu), torch.cat(ls)


def overfit(dev):
    torch.manual_seed(CFG["seed"])
    X, y, _, _ = load("val", dev)
    X, y = X[:64], y[:64]
    model = Net(3, CFG["ch1_k"], CFG["ch2_scale"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    for step in range(1, 1501):
        model.train()
        mu, ls = model(X)
        hub, nll = loss_fn(mu, ls, y)
        opt.zero_grad()
        hub.backward()                    # position path only (see header)
        opt.step()
        if step in (1, 100, 500, 1000, 1500):
            print(f"overfit step {step:5d}  huber {hub.item():.5f}")
    ok = hub.item() < 1e-3
    print(f"overfit: final huber {hub.item():.2e} (prediction < 1e-3) -> {'PASS' if ok else 'FAIL'}")
    return ok


def train(name, in_ch, dev):
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    torch.manual_seed(CFG["seed"])
    gen = torch.Generator(device=dev).manual_seed(CFG["seed"])
    Xt, yt, _, _ = load("train", dev, CFG.get("train_dataset"))
    cpu = Xt.device.type == "cpu"
    Xv, yv, Pv, Xv_np = load("val", dev)
    model = Net(in_ch, CFG["ch1_k"], CFG["ch2_scale"], CFG["arch"]).to(dev)
    macs, npar = count_macs(model)
    print(f"{name}: {npar} parameters, {macs / 1e6:.2f} M multiply-adds per patch, train {len(Xt)}, val {len(Xv)}")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    steps = CFG["epochs"] * (len(Xt) // CFG["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"], total_steps=steps, pct_start=0.05,
                                                anneal_strategy="cos")
    u = metrics.trail_dirs(Xv_np)
    hist, best, t0 = [], (math.inf, -1), time.time()
    for ep in range(1, CFG["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(Xt), device=dev, generator=gen)
        if cpu:
            perm = perm.cpu()
        tl, n = 0.0, 0
        for i in range(0, len(Xt) - CFG["batch"] + 1, CFG["batch"]):
            idx = perm[i:i + CFG["batch"]]
            xb, yb = Xt[idx], yt[idx]
            if cpu:
                xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            xb, yb = augment(xb, yb, gen)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=CFG.get("amp", True)):
                mu, ls = model(xb)
            hub, nll = loss_fn(mu.float(), ls.float(), yb)
            opt.zero_grad(set_to_none=True)
            (hub + CFG["nll_weight"] * nll).backward()
            opt.step()
            sched.step()
            tl, n = tl + hub.item(), n + 1
        mu, ls = predict(model, Xv)
        vl = loss_fn(mu, ls, yv)[0].item()
        s = metrics.stats((mu - yv).cpu().numpy(), u)
        hist.append({"epoch": ep, "train_huber": tl / n, "val_huber": vl, "val_median": s["median"],
                     "val_rms_no_fail": s["rms_no_fail"], "val_fail": s["fail"]})
        if vl < best[0]:
            best = (vl, ep)
            torch.save({"state": model.state_dict(), "in_ch": in_ch, "arch": CFG["arch"], "cfg": CFG, "epoch": ep}, CKPT / f"step7_{name}.pt")
        print(f"ep {ep:3d}  train {tl / n:.4f}  val {vl:.4f}  median {s['median']:.3f}  rms<2 {s['rms_no_fail']:.3f}"
              f"  fail {s['fail']:.4f}  {time.time() - t0:5.0f} s", flush=True)
    report(name, in_ch, dev, Xv, yv, Pv, Xv_np, hist, best, macs, npar)


def report(name, in_ch, dev, Xv, yv, Pv, Xv_np, hist, best, macs, npar):
    ck = torch.load(CKPT / f"step7_{name}.pt", map_location=dev)
    model = Net(in_ch, CFG["ch1_k"], CFG["ch2_scale"], CFG["arch"]).to(dev)
    model.load_state_dict(ck["state"])
    mu, ls = predict(model, Xv)
    e = (mu - yv).cpu().numpy()
    sig = torch.exp(ls).cpu().numpy()
    yn = yv.cpu().numpy()
    L = [f"step 7 -- {name} (channels {in_ch}), val split of {CFG['dataset']}, best epoch {best[1]} "
         f"(val huber {best[0]:.4f})", f"config {json.dumps(CFG)}",
         f"{npar} parameters, {macs / 1e6:.2f} M multiply-adds per patch", ""]
    _, lines = metrics.table({name: e}, Pv, Xv_np)
    L += lines + [""]
    z = np.abs(e) / sig
    L.append("sigma calibration (fraction |e|/sigma < 1, < 2 per axis; Gaussian 0.683, 0.954):")
    L.append(f"  all: {np.mean(z < 1):.3f} {np.mean(z < 2):.3f}")
    for lo, hi in ((2, 5), (5, 20), (20, 1e9)):
        m = (Pv["snr"] >= lo) & (Pv["snr"] < hi)
        L.append(f"  snr {lo:g}-{hi:g}: {np.mean(z[m] < 1):.3f} {np.mean(z[m] < 2):.3f}  "
                 f"median sigma {np.median(sig[m]):.3f} px")
    L += ["", "pixel locking (trap 14): mean error per axis vs fractional true position (snr >= 20):"]
    frac = (16.5 + yn) % 1.0
    hi_snr = Pv["snr"] >= 20
    for ax, an in ((0, "row"), (1, "col")):
        b = [e[hi_snr & (frac[:, ax] >= f) & (frac[:, ax] < f + 0.2), ax].mean() for f in np.arange(0, 1, 0.2)]
        L.append(f"  {an}: " + " ".join(f"{v:+.3f}" for v in b) + f"   (span {max(b) - min(b):.3f} px)")
    last = [h["val_huber"] for h in hist[-10:]]
    L += ["", f"val huber last 10 epochs: first {last[0]:.4f} last {last[-1]:.4f} "
          f"(still falling = more data/epochs may help)"]
    txt = "\n".join(L)
    print(txt)
    (OUT / f"{name}.txt").write_text(txt + "\n")
    (OUT / f"{name}.json").write_text(json.dumps({"cfg": CFG, "best_epoch": best[1], "params": npar, "macs": macs,
                                                  "history": hist}, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, [h["train_huber"] for h in hist], label="train (augmented)")
    ax[0].plot(ep, [h["val_huber"] for h in hist], label="val")
    ax[0].set_xlabel("epoch"), ax[0].set_ylabel("Huber loss"), ax[0].legend(), ax[0].set_yscale("log")
    ax[1].plot(ep, [h["val_median"] for h in hist], label="val median |e|")
    ax[1].plot(ep, [h["val_rms_no_fail"] for h in hist], label="val rms |e|<=2")
    ax[1].plot(ep, [h["val_fail"] for h in hist], label="val fail rate")
    ax[1].set_xlabel("epoch"), ax[1].legend()
    fig.suptitle(f"step 7 {name}")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--name", default="full")
    ap.add_argument("--channels", type=int, default=3, choices=(1, 3))
    ap.add_argument("--nll-weight", type=float, default=CFG["nll_weight"])
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--arch", default="a", choices=sorted(ARCHS))
    ap.add_argument("--min-snr", type=float, default=0, help="diagnostic: train and val on snr >= this only")
    ap.add_argument("--fp32", action="store_true", help="no bf16 autocast")
    ap.add_argument("--ch1-k", type=float, default=CFG["ch1_k"])
    ap.add_argument("--train-dataset", default=None, help="train split from this dataset (val stays v2)")
    a = ap.parse_args()
    CFG["min_snr"], CFG["amp"], CFG["ch1_k"] = a.min_snr, not a.fp32, a.ch1_k
    CFG["train_dataset"] = a.train_dataset or CFG["dataset"]
    CFG["nll_weight"], CFG["epochs"], CFG["arch"] = a.nll_weight, a.epochs, a.arch
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    assert torch.cuda.is_available()
    dev = "cuda"
    if a.overfit:
        sys.exit(0 if overfit(dev) else 1)
    train(a.name, a.channels, dev)


if __name__ == "__main__":
    main()
