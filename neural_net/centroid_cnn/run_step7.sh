#!/bin/bash
# Step 7 offline: augmentation self-test, one-batch overfit, then the full CNN
# (+ a control without the NLL term) and the no-gyro ablation. From this folder:
#     nohup ./run_step7.sh > scratch/run_step7.log 2>&1 &
# Results: reports/step7/{full,nogyro}.txt/.json/.png, checkpoints/step7_*.pt.
cd "$(dirname "$0")"
mkdir -p reports/step7
echo "start $(date)"
python src/train_cnn.py --selftest > reports/step7/selftest.txt 2>&1 \
    || { echo "selftest FAILED, see reports/step7/selftest.txt"; exit 1; }
python src/train_cnn.py --overfit > reports/step7/overfit.txt 2>&1 \
    || { echo "overfit FAILED, see reports/step7/overfit.txt"; exit 1; }
echo "checks done $(date)"
python src/train_cnn.py --name full > scratch/train_full_stdout.txt 2>&1 \
    && echo "full done $(date)" || echo "full FAILED, see scratch/train_full_stdout.txt"
python src/train_cnn.py --name full_nonll --nll-weight 0 > scratch/train_full_nonll_stdout.txt 2>&1 \
    && echo "full_nonll done $(date)" || echo "full_nonll FAILED, see scratch/train_full_nonll_stdout.txt"
python src/train_cnn.py --name nogyro --channels 1 > scratch/train_nogyro_stdout.txt 2>&1 \
    && echo "nogyro done $(date)" || echo "nogyro FAILED, see scratch/train_nogyro_stdout.txt"
echo "end $(date)"
