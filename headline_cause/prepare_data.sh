#!/bin/bash

DOCS_PATH="../data/lenta.jsonl"
FINAL_DOCS_PATH="../data/lenta_50000_embeddings.jsonl"
PAIRS_PATH="../data/lenta_50000_pairs.jsonl"

#CUDA_VISIBLE_DEVICES=0 python3.9 labse_docs.py \
#    --input-path $DOCS_PATH \
#    --output-path $FINAL_DOCS_PATH \
#    --ndocs 50000;

python3.9 sample_pairs.py \
    --docs-path $FINAL_DOCS_PATH \
    --output-path $PAIRS_PATH \
    --sample-rate 0.5 \
    --id-prefix lenta_v2;
