#!/bin/bash

NROWS=$1
if [ -z "$1" ]; then
    echo "Provide nrows"
    exit -1
fi

AGG_PATH="../data/agg.txt"
DOCS_PATH="../data/lenta_50000_embeddings.jsonl"
PAIRS_PATH="../data/lenta_50000_pairs.jsonl"
ANNOTATED_PAIRS_PATH="../data/lenta_50000_pairs_annotated.jsonl"
BERT_PATH="../clf"
SAMPLE_PATH="../data/sample.jsonl"

echo "==== Inferring BERT with Monte-Carlo Dropout"
CUDA_VISIBLE_DEVICES=1 python3.9 infer_clf.py \
    --input-path $PAIRS_PATH \
    --output-path $ANNOTATED_PAIRS_PATH \
    --model-path $BERT_PATH;

echo "==== Calculating KNN Density"
#python3.9 calc_density.py \
#    --docs-path $DOCS_PATH \
#    --embedding-key labse_embedding \
#    --input-path $ANNOTATED_PAIRS_PATH \
#    --output-path $ANNOTATED_PAIRS_PATH \
#    --k 20;

echo "==== Sorting by bald, filtering with RECS"
python3.9 filter_pairs.py \
    --input-path $ANNOTATED_PAIRS_PATH \
    --nrows $NROWS \
    --sort-field bald \
    --output-path $SAMPLE_PATH \
    --existing-path $AGG_PATH \
    --embedding-key labse_embedding \
    --docs-path $DOCS_PATH;
#    --use-recs;

echo "Output path: ${SAMPLE_PATH}"
echo "===="
