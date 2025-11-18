#!/bin/bash

FOLD=$1
GPU_ID=$2
MODEL_NAME=${3:-GINConvNet}

if [ -z "$FOLD" ] || [ -z "$GPU_ID" ]; then
  echo "Usage: $0 [FOLD] [GPU_ID] [MODEL_NAME(optional)]"
  exit 1
fi

TASK_NAME="Ki"

ROOT="/home/rlawlsgurjh/hdd/work/ChEMBL/data/processed/${TASK_NAME}/fold${FOLD}"
TRAIN_CSV="${ROOT}/train.csv"
VAL_CSV="${ROOT}/val.csv"

OUT_DIR="./results_graph/${TASK_NAME}/${MODEL_NAME}/fold_${FOLD}"

echo "=== Training ${TASK_NAME} Fold ${FOLD} with ${MODEL_NAME} on GPU ${GPU_ID} ==="

python train.py \
    --task_name "${TASK_NAME}" \
    --model_name "${MODEL_NAME}" \
    --train_csv "${TRAIN_CSV}" \
    --val_csv "${VAL_CSV}" \
    --out_dir "${OUT_DIR}" \
    --epochs 1000 \
    --batch_size 256 \
    --lr 0.0005 \
    --seed 42 \
    --patience 10 \
    --cuda "${GPU_ID}"

echo "=== Fold ${FOLD} training finished ==="
