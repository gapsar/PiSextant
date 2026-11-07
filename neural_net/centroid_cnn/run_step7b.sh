#!/bin/bash
# Step 7 round 2 offline: arch b (full resolution longer) and the nogyro
# ablation at 200 epochs, then the paired val comparison. From this folder:
#     nohup ./run_step7b.sh > scratch/run_step7b.log 2>&1 &
cd "$(dirname "$0")"
source ../../dev_env.sh
echo "start $(date)"
python src/train_cnn.py --name full_b_e200 --arch b --epochs 200 > scratch/train_full_b_e200_stdout.txt 2>&1 \
    && echo "full_b_e200 done $(date)" || echo "full_b_e200 FAILED"
python src/train_cnn.py --name nogyro_e200 --channels 1 --epochs 200 > scratch/train_nogyro_e200_stdout.txt 2>&1 \
    && echo "nogyro_e200 done $(date)" || echo "nogyro_e200 FAILED"
python src/compare_val.py full_e200 full_b_e200 nogyro_e200 > scratch/compare_val_round2_stdout.txt 2>&1 \
    && echo "compare done $(date)" || echo "compare FAILED"
echo "end $(date)"
