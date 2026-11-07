#!/bin/bash
# Step 9 offline: real-data replay with the CNN on the held-out runs (17, 18
# Oct) and the test runs (14 Oct, 20 Oct first 75 min, tau 10 and 64 ms). From this folder:
#     setsid nohup ./run_step9.sh > scratch/run_step9.log 2>&1 < /dev/null &
# Result: reports/step9/summary.txt (+ <run>_frames.jsonl).
cd "$(dirname "$0")"
source ../../dev_env.sh
echo "start $(date)"
python src/real_eval.py 2026-10-17_16-00-56 2026-10-18_16-19-09 2026-10-14_15-01-37 2026-10-14_15-05-44 --workers 10 \
    > scratch/real_eval_stdout.txt 2>&1 && echo "17/18/14 done $(date)" || echo "FAILED, see scratch/real_eval_stdout.txt"
python src/real_eval.py 2026-10-20_15-36-33 --max-min 75 --workers 10 > scratch/real_eval_20_stdout.txt 2>&1 \
    && echo "20 Oct tau 10 done $(date)" || echo "20 Oct FAILED"
python src/real_eval.py 2026-10-20_15-36-33 --max-min 75 --tau 0.064 --tag _tau64 --workers 10 > scratch/real_eval_20b_stdout.txt 2>&1 \
    && echo "20 Oct tau 64 done $(date)" || echo "20 Oct tau64 FAILED"
python src/real_eval.py --summary 2026-10-17_16-00-56 2026-10-18_16-19-09 2026-10-14_15-01-37 2026-10-14_15-05-44 \
    2026-10-20_15-36-33 2026-10-20_15-36-33_tau64 > scratch/real_summary_stdout.txt 2>&1 \
    && echo "summary done $(date)" || echo "summary FAILED, see scratch/real_summary_stdout.txt"
echo "end $(date)"
