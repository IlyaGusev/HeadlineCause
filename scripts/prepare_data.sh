#!/bin/bash

set -e;

DOCS_PATH="data/gazeta_train_ts.jsonl"
FINAL_DOCS_PATH="data/gazeta_embeddings.jsonl"
PAIRS_PATH="data/gazeta_pairs.jsonl"

cd headline_cause;

CUDA_VISIBLE_DEVICES=0 python3 -m labse_docs \
    --input-path ../$DOCS_PATH \
    --output-path ../$FINAL_DOCS_PATH;

python3 sample_pairs.py \
    --docs-path ../$FINAL_DOCS_PATH \
    --output-path ../$PAIRS_PATH \
    --sample-rate 0.5 \
    --id-prefix gazeta;
