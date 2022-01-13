#!/bin/bash

set -e;

POOLS_PATH=$1

AGG_PATH="data/agg.txt"
RAW_PATH="data/raw.txt"
BERT_PATH="clf"

echo "Aggregated records path: ${AGG_PATH}"
echo "Model path: ${BERT_PATH}"
echo ""

cd headline_cause;

echo "==== Pulling Toloka pools"
python3 -m crowd.aggregate \
    --agg-output ../$AGG_PATH \
    --raw-output ../$RAW_PATH \
    --pools-file ../$POOLS_PATH \
    --token "~/.toloka/personal_token"

echo "==== Training BERT"
CUDA_VISIBLE_DEVICES=1 python3 -m train_clf \
    --input-path ../$AGG_PATH \
    --out-dir ../$BERT_PATH;
echo "===="
