#!/bin/bash

FOLD=$1
GPU_ID=$2
MODEL_NAME=${3:-GINConvNet}

if [ -z "$FOLD" ] || [ -z "$GPU_ID" ]; then
  echo "Usage: $0 [FOLD] [GPU_ID] [MODEL_NAME(optional)]"
  exit 1
fi

TASK_NAME="Kd"

ROOT="/home/rlawlsgurjh/hdd/work/ChEMBL/data/processed/${TASK_NAME}/fold${FOLD}"
TEST_CSV="${ROOT}/test.csv"

OUT_DIR="./results_graph/${TASK_NAME}/${MODEL_NAME}/fold_${FOLD}"
MODEL_PATH="${OUT_DIR}/best_model.pt"
OUT_CSV="${OUT_DIR}/predictions.csv"

echo "=== Predicting Fold ${FOLD} with ${MODEL_NAME} on GPU ${GPU_ID} ==="
echo " Test CSV : ${TEST_CSV}"
echo " Model    : ${MODEL_PATH}"
echo " Output   : ${OUT_CSV}"

python predict.py \
    --task_name "${TASK_NAME}" \
    --model_name "${MODEL_NAME}" \
    --test_csv "${TEST_CSV}" \
    --model_path "${MODEL_PATH}" \
    --out_csv "${OUT_CSV}" \
    --cuda "${GPU_ID}"

echo "=== Fold ${FOLD} Prediction Finished ==="
