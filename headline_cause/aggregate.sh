#!/bin/bash

AGG_PATH="../data/agg.txt"
RAW_PATH="../data/raw.txt"
BERT_PATH="../clf"

echo "Aggregated records path: ${AGG_PATH}"
echo "Model path: ${BERT_PATH}"
echo ""

echo "==== Pulling Toloka pools"
python3.9 -m crowd.aggregate \
    --agg-output $AGG_PATH \
    --raw-output $RAW_PATH \
    --pools-file "../toloka/pools.txt" \
    --token "~/.toloka/personal_token"

echo "==== Training BERT"
CUDA_VISIBLE_DEVICES=1 python3 train_clf.py \
    --input-path $AGG_PATH \
    --out-dir $BERT_PATH;
echo "===="
