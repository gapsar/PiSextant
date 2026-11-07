#!/bin/bash
# Step 7 round 4 offline: arch a on datasets/v3 for 120 epochs, then the paired
# val comparison (v2 val) with the 100k and the 1M/40-epoch models. From this folder:
#     setsid nohup ./run_step7d.sh > scratch/run_step7d.log 2>&1 < /dev/null &
# Results: reports/step7/full_v3_e120.txt/.json/.png, reports/step7/compare_val.txt
# (copy: reports/step7/compare_val_round4.txt), checkpoints/step7_full_v3_e120.pt.
cd "$(dirname "$0")"
source ../../dev_env.sh
echo "start $(date)"
# Wait (up to 12 h) until the GPU has >= 2 GB free: another program (strata) can hold it.
for i in $(seq 720); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge 2000 ] && break
    [ $((i % 30)) -eq 1 ] && echo "GPU busy ($free MiB free), waiting $(date)"
    sleep 60
done
[ "$free" -ge 2000 ] || { echo "GPU still busy after 12 h, stop $(date)"; exit 1; }
echo "GPU free ($free MiB), training $(date)"
python src/train_cnn.py --name full_v3_e120 --train-dataset v3 --epochs 120 > scratch/train_full_v3_e120_stdout.txt 2>&1 \
    && echo "full_v3_e120 done $(date)" || { echo "full_v3_e120 FAILED, see scratch/train_full_v3_e120_stdout.txt"; exit 1; }
python src/compare_val.py full_e200 full_v3 full_v3_e120 > scratch/compare_val_round4_stdout.txt 2>&1 \
    && cp reports/step7/compare_val.txt reports/step7/compare_val_round4.txt \
    && echo "compare done $(date)" || echo "compare FAILED, see scratch/compare_val_round4_stdout.txt"
echo "end $(date)"
