"""Step 7: paired comparison on the VAL split of the step-7 CNNs with the
step-6 baselines (centre, gauss, gyro_fit). Val only: the test split stays for
step 8. Written by Claude, 29 Oct 2026.
    source ../../dev_env.sh; python src/compare_val.py [model names, default full nogyro]
Output: reports/step7/compare_val.txt, scratch/val_baselines.npz (cache).
Prediction: the CNN beats gyro_fit in failures in every bin and in median at
SNR < 20 and on trails; at SNR 50-300 the median is about equal.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))
import baselines as B  # noqa: E402
import check_step5  # noqa: E402
import metrics  # noqa: E402
import train_cnn as T  # noqa: E402


def main():
    X, y, P = check_step5.load_split(HERE / "datasets" / "v2", "val")
    cache = HERE / "scratch" / "val_baselines.npz"
    if cache.exists():
        est = dict(np.load(cache))
    else:
        with Pool(16) as pool:
            res = pool.map(B.one, list(X), chunksize=50)
        est = {m: np.array([r[0][m] for r in res]) for m in ("centre", "gauss", "gyro_fit")}
        np.savez(cache, **est)
    errors = {m: est[m] - y for m in ("centre", "gauss", "gyro_fit")}
    Xg = torch.from_numpy(X).cuda()
    names = sys.argv[1:] or ["full", "nogyro"]
    for name in names:
        ck = torch.load(HERE / "checkpoints" / f"step7_{name}.pt", map_location="cuda")
        model = T.Net(ck["in_ch"], arch=ck.get("arch", "a")).cuda()
        model.load_state_dict(ck["state"])
        errors["cnn_" + name] = T.predict(model, Xg)[0].cpu().numpy() - y
    subsets = {"still, snr>=20": (P["centre_len_px"] < 1) & (P["snr"] >= 20),
               "trail>3, snr>=20": (P["centre_len_px"] > 3) & (P["snr"] >= 20)}
    _, lines = metrics.table(errors, P, X, subsets)
    r = {m: np.linalg.norm(e, axis=1) for m, e in errors.items()}
    lines.append("")
    for name in names:
        d = r["cnn_" + name] - r["gyro_fit"]
        lines.append(f"paired, all val: cnn_{name} better than gyro_fit on {np.mean(d < 0):.3f} of samples; "
                     f"median |e| difference {np.median(d):+.3f} px")
    txt = f"step 7 val comparison {' '.join(names)} (v2 val, n {len(y)})\n" + "\n".join(lines)
    (HERE / "reports" / "step7" / "compare_val.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
