#!/bin/bash
# Step 4 checks, run offline (no Claude session needed). From this folder:
#     nohup ./run_step4.sh > scratch/run_step4.log 2>&1 &
# Each check writes its results to reports/step4/ (.json + .txt summary)
# and its stdout to scratch/. Read reports/step4/*.txt when it is done.
cd "$(dirname "$0")"
set -u
echo "start $(date)"
python src/measure_gyro.py > scratch/measure_gyro_stdout.txt 2>&1 \
    && echo "gyro done $(date)" || echo "gyro FAILED, see scratch/measure_gyro_stdout.txt"
for args in "--run 2026-10-15_16-00-55 --frames 40" \
            "--run 2026-10-16_16-05-50 --rate-max 8 --frames 60"; do
    tag=$(echo $args | awk '{print substr($2, 1, 10)}')
    python src/check_trails.py $args 2>&1 | grep -v 'Warning\|results =' \
        > scratch/check_trails_${tag}_stdout.txt
    if [ ${PIPESTATUS[0]} -eq 0 ]; then echo "trails $tag done $(date)"
    else echo "trails $tag FAILED, see scratch/check_trails_${tag}_stdout.txt"; fi
done
python src/motion.py > scratch/motion_stdout.txt 2>&1 \
    && echo "motion done $(date)" || echo "motion FAILED, see scratch/motion_stdout.txt"
echo "end $(date)"
