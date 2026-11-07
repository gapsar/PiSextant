#!/bin/bash
# Generator v2 + step 5 checks + step 6 baselines, offline. From this folder:
#     nohup ./run_step6.sh > scratch/run_step6.log 2>&1 &
# Results: reports/step5/checks.txt, reports/step6/baselines.txt.
cd "$(dirname "$0")"
source ../../dev_env.sh          # tetra3 for the baselines
echo "start $(date)"
python src/generator.py --name v2 --train 100000 --val 10000 --test 10000 \
    > scratch/generator_v2_stdout.txt 2>&1 \
    && echo "generator v2 done $(date)" || { echo "generator FAILED, see scratch/generator_v2_stdout.txt"; exit 1; }
python src/check_step5.py --name v2 > scratch/check_step5_v2_stdout.txt 2>&1 \
    && echo "step 5 checks done $(date)" || echo "step 5 checks FAILED, see scratch/check_step5_v2_stdout.txt"
python src/baselines.py --name v2 > scratch/baselines_v2_stdout.txt 2>&1 \
    && echo "baselines done $(date)" || echo "baselines FAILED, see scratch/baselines_v2_stdout.txt"
echo "end $(date)"
