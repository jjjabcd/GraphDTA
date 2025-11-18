#!/bin/bash

LOG_FILE="run_all_Ki.log"

nohup bash -c '

echo "===== Ki All-Fold Pipeline Started at $(date) ====="

for FOLD in 1 2 3
do
  echo "--- FOLD ${FOLD} START ---"

  bash scripts/Ki_train.sh ${FOLD} 1 &&
  bash scripts/Ki_predict.sh ${FOLD} 1 &&
  bash scripts/Ki_eval.sh ${FOLD}

  echo "--- FOLD ${FOLD} DONE ---"
done

echo "===== Ki All-Fold Pipeline Finished at $(date) ====="

' > "$LOG_FILE" 2>&1 &

echo "Started run_all_Ki.sh with PID: $!"