#!/bin/bash

LOG_FILE="run_all_Kd.log"

nohup bash -c '

echo "===== Kd All-Fold Pipeline Started at $(date) ====="

for FOLD in 1 2 3
do
  echo "--- FOLD ${FOLD} START ---"

  bash scripts/Kd_train.sh ${FOLD} 0 &&
  bash scripts/Kd_predict.sh ${FOLD} 0 &&
  bash scripts/Kd_eval.sh ${FOLD}

  echo "--- FOLD ${FOLD} DONE ---"
done

echo "===== Kd All-Fold Pipeline Finished at $(date) ====="

' > "$LOG_FILE" 2>&1 &

echo "Started run_all_Kd.sh with PID: $!"