#!/bin/bash
# Step 8 offline: test sets (configs/step8/*.json), the no-gyro ablation on v3
# (same recipe as step7_full_v3_e120), then src/evaluate.py. From this folder:
#     setsid nohup ./run_step8.sh > scratch/run_step8.log 2>&1 < /dev/null &
# Result: reports/step8/eval.txt (+ eval.json).
cd "$(dirname "$0")"
source ../../dev_env.sh
mkdir -p reports/step8
echo "start $(date)"
for f in configs/step8/s8_*.json; do
    s=$(basename "$f" .json)
    [ -f "datasets/$s/meta.json" ] && continue
    python src/generator.py --name "$s" --train 0 --val 0 --test 5000 --workers 20 --override "$f" \
        > "scratch/generator_${s}_stdout.txt" 2>&1 && echo "$s done $(date)" || echo "$s FAILED, see scratch/generator_${s}_stdout.txt"
done
if [ ! -f checkpoints/step7_nogyro_v3_e120.pt ]; then
    for i in $(seq 720); do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [ "$free" -ge 2000 ] && break
        [ $((i % 30)) -eq 1 ] && echo "GPU busy ($free MiB free), waiting $(date)"
        sleep 60
    done
    echo "ablation training $(date)"
    python src/train_cnn.py --name nogyro_v3_e120 --channels 1 --train-dataset v3 --epochs 120 \
        > scratch/train_nogyro_v3_e120_stdout.txt 2>&1 \
        && echo "nogyro_v3_e120 done $(date)" || echo "nogyro_v3_e120 FAILED, see scratch/train_nogyro_v3_e120_stdout.txt"
fi
python src/evaluate.py > scratch/evaluate_stdout.txt 2>&1 \
    && echo "evaluate done $(date)" || echo "evaluate FAILED, see scratch/evaluate_stdout.txt"
echo "end $(date)"
