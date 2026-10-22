#!/usr/bin/env bash
# Step B of the run_2 calibration: (re)run M3/M4/M5 so that results/m3..m5_params.json
# carry run_1 + run_2. Each script calibrates every run found in stage_4 and seeds
# its RNG per run, so run_1's numbers are expected to reproduce bit for bit.
set -u
LAB1=/c/Users/ptiu1/Documents/PhDStudy/INTR5100/lab1
PY="C:/Users/ptiu1/miniconda3/envs/intr5100/python.exe"
LOGS="$LAB1/.coding-dept/lab1-run2/logs"
cd "$LAB1" || exit 1
mkdir -p "$LOGS"
for script in m3_ga m4_pso m5_differentiable; do
  echo "===== $script ====="
  start=$(date +%s)
  "$PY" "src/calibrate/$script.py" > "$LOGS/$script.stdout.txt" 2>&1
  code=$?
  end=$(date +%s)
  echo "$script exit=$code elapsed=$((end - start))s"
  tail -n 3 "$LOGS/$script.stdout.txt"
done
echo "STEP_B_DONE"
