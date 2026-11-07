#!/bin/bash
# Step 5, run offline (no Claude session needed). From this folder:
#     nohup ./run_step5.sh > scratch/run_step5.log 2>&1 &
# Generates datasets/v1 (100k / 10k / 10k) and runs the checks.
# Results: reports/step5/checks.txt (+ png). Read checks.txt when done.
cd "$(dirname "$0")"
echo "start $(date)"
python src/generator.py --name v1 --train 100000 --val 10000 --test 10000 \
    > scratch/generator_v1_stdout.txt 2>&1 \
    && echo "generator done $(date)" || { echo "generator FAILED, see scratch/generator_v1_stdout.txt"; exit 1; }
python src/check_step5.py --name v1 > scratch/check_step5_stdout.txt 2>&1 \
    && echo "checks done $(date)" || echo "checks FAILED, see scratch/check_step5_stdout.txt"
echo "end $(date)"
