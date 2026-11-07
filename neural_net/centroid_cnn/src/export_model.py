"""Step 10: export the step 9 CNN (configs/hybrid.json "model") to ONNX and to
a numpy weight file, check both against PyTorch, and time them on the laptop
CPU with one thread. Written by Claude, 6 Nov 2026. Run from the centroid_cnn
folder:
    python src/export_model.py
Output: export/model.onnx (opset 17), export/cnn_weights.npz (for
src/pi_cnn.py), reports/step10/summary.txt.
Predictions: ONNX and numpy agree with PyTorch to < 1e-5 px (float32 round
off); the model is ~0.1 ms per patch on one laptop core, so 1-2 ms per patch
on the Pi (10-20x slower): above the 1 ms budget at the upper end.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
import train_cnn as T  # noqa: E402
import pi_cnn  # noqa: E402

OUT = HERE / "export"
REP = HERE / "reports" / "step10"
RULE = json.load(open(HERE / "configs" / "hybrid.json"))
N_CHECK, N_FRAME = 1000, 95


class Wrapped(torch.nn.Module):
    """Same computation as T.Net.forward, but asinh written as
    sign(x) log(|x| + sqrt(x^2 + 1)): the TorchScript ONNX exporter has no
    rule for aten::asinh (the check below compares with the real net)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        x = x[:, :n.in_ch].float()
        a = x[:, 0:1] / n.k1
        ch = [torch.sign(a) * torch.log(a.abs() + torch.sqrt(a * a + 1))]
        if n.in_ch > 1:
            ch += [x[:, 1:2] * n.s2, x[:, 2:3]]
        x = torch.cat(ch, 1)
        for c in n.convs:
            x = torch.relu(c(x))
        out = n.f2(torch.relu(n.f1(x.flatten(1))))
        return out[:, :2], out[:, 2:].clamp(-4, 4)


def time_it(fn, X, repeat=30):
    fn(X)
    t = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(X)
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)


def main():
    OUT.mkdir(exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    ck = torch.load(HERE / "checkpoints" / f"{RULE['model']}.pt", map_location="cpu")
    arch = ck.get("arch", "a")
    net = T.Net(ck["in_ch"], ck["cfg"]["ch1_k"], ck["cfg"]["ch2_scale"], arch)
    net.load_state_dict(ck["state"])
    net.eval()
    n_par = sum(p.numel() for p in net.parameters())

    # numpy weights
    w = {"k1": net.k1, "s2": net.s2, "in_ch": net.in_ch}
    for i, c in enumerate(net.convs):
        w[f"c{i + 1}_w"] = c.weight.detach().numpy()
        w[f"c{i + 1}_b"] = c.bias.detach().numpy()
        w[f"c{i + 1}_stride"] = c.stride[0]
    for name in ("f1", "f2"):
        lin = getattr(net, name)
        w[f"{name}_w"], w[f"{name}_b"] = lin.weight.detach().numpy(), lin.bias.detach().numpy()
    np.savez(OUT / "cnn_weights.npz", **w)

    # ONNX
    x0 = torch.zeros(1, 3, 32, 32)
    torch.onnx.export(Wrapped(net), x0, OUT / "model.onnx", opset_version=17, dynamo=False,
                      input_names=["x"], output_names=["mu", "log_sigma"],
                      dynamic_axes={"x": {0: "n"}, "mu": {0: "n"}, "log_sigma": {0: "n"}})
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(OUT / "model.onnx"), so, providers=["CPUExecutionProvider"])

    # 1000 test patches (v3 test split, never used for training)
    d = np.load(HERE / "datasets" / "v3" / "test" / "shard_000.npz")
    X = d["X"][:N_CHECK].astype(np.float32)
    y = d["y"][:N_CHECK]
    with torch.no_grad():
        mu_t, ls_t = [a.numpy() for a in net(torch.from_numpy(X))]
    mu_o, ls_o = sess.run(None, {"x": X})
    npnet = pi_cnn.CentroidCnn(OUT / "cnn_weights.npz")
    mu_n, ls_n = npnet(X)
    d_onnx = float(np.abs(mu_o - mu_t).max())
    d_np = float(np.abs(mu_n - mu_t).max())
    d_ls = float(max(np.abs(ls_o - ls_t).max(), np.abs(ls_n - ls_t).max()))
    rmse = float(np.sqrt(((mu_t - y) ** 2).sum(1).mean()))

    # timing: one frame = 95 patches, one thread
    Xf = X[:N_FRAME]
    with torch.no_grad():
        t_torch = time_it(lambda a: net(torch.from_numpy(a)), Xf)
    t_onnx = time_it(lambda a: sess.run(None, {"x": a}), Xf)
    t_np = time_it(npnet, Xf)
    t_one = time_it(lambda a: sess.run(None, {"x": a}), X[:1], repeat=200)

    lines = [
        f"Step 10 export -- {time.strftime('%Y-%m-%d %H:%M')}",
        f"model {RULE['model']} arch {arch}, {n_par} parameters, in_ch {net.in_ch}, "
        f"k1 {net.k1}, s2 {net.s2}",
        f"files: export/model.onnx ({(OUT / 'model.onnx').stat().st_size} B, opset 17, "
        f"input x (n,3,32,32) float32, outputs mu, log_sigma), "
        f"export/cnn_weights.npz ({(OUT / 'cnn_weights.npz').stat().st_size} B)",
        f"check on {N_CHECK} v3 test patches: max |mu onnx - torch| {d_onnx:.2e} px, "
        f"max |mu numpy - torch| {d_np:.2e} px, max log_sigma diff {d_ls:.2e} "
        f"-> {'PASS' if max(d_onnx, d_np) < 1e-4 else 'FAIL'} (limit 1e-4 px)",
        f"torch RMSE vs label on these patches {rmse:.4f} px (sanity)",
        f"laptop CPU one thread, {N_FRAME} patches: torch {t_torch:.2f} ms, onnxruntime {t_onnx:.2f} ms, "
        f"numpy {t_np:.2f} ms; onnxruntime 1 patch {t_one:.3f} ms",
        f"per patch numpy {t_np / N_FRAME:.3f} ms -> Pi estimate x10-20: "
        f"{10 * t_np / N_FRAME:.2f}-{20 * t_np / N_FRAME:.2f} ms per patch, "
        f"{10 * t_np:.0f}-{20 * t_np:.0f} ms per frame (budget ~1 ms / ~100 ms)",
        "Pi measurement: ~/soft-env/bin/python pi_cnn.py --bench cnn_weights.npz",
    ]
    txt = "\n".join(lines)
    print(txt)
    (REP / "summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()
