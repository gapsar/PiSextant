#!/bin/bash
# Step 7 round 3 offline: dataset v3 (1M train, same config and seed as v2),
# arch a trained on it, paired val comparison (v2 val). From this folder:
#     nohup ./run_step7c.sh > scratch/run_step7c.log 2>&1 &
cd "$(dirname "$0")"
source ../../dev_env.sh
echo "start $(date)"
if [ ! -f datasets/v3/meta.json ]; then
    python src/generator.py --name v3 --train 1000000 --val 1000 --test 1000 --workers 20 \
        > scratch/generator_v3_stdout.txt 2>&1 \
        && echo "generator v3 done $(date)" || { echo "generator FAILED, see scratch/generator_v3_stdout.txt"; exit 1; }
fi
python src/train_cnn.py --name full_v3 --train-dataset v3 --epochs 40 > scratch/train_full_v3_stdout.txt 2>&1 \
    && echo "full_v3 done $(date)" || { echo "full_v3 FAILED, see scratch/train_full_v3_stdout.txt"; exit 1; }
python src/compare_val.py full_e200 full_v3 > scratch/compare_val_round3_stdout.txt 2>&1 \
    && echo "compare done $(date)" || echo "compare FAILED"
echo "end $(date)"
