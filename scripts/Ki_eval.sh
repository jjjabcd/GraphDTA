#!/bin/bash

FOLD=$1
MODEL_NAME=${2:-GINConvNet}

if [ -z "$FOLD" ]; then
  echo "Usage: $0 [FOLD] [MODEL_NAME(optional)]"
  exit 1
fi

TASK_NAME="Ki"

OUT_DIR="./results_graph/${TASK_NAME}/${MODEL_NAME}/fold_${FOLD}"
PRED_CSV="${OUT_DIR}/predictions.csv"
METRICS_CSV="${OUT_DIR}/metrics.csv"

echo "=== Evaluating Fold ${FOLD} (${MODEL_NAME}) ==="
echo " Pred CSV : ${PRED_CSV}"
echo " Output   : ${METRICS_CSV}"

python eval.py \
    --task_name "${TASK_NAME}" \
    --pred_csv "${PRED_CSV}" \
    --out_csv "${METRICS_CSV}"

echo "=== Fold ${FOLD} Evaluation Finished ==="
