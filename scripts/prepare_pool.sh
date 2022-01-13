#!/bin/bash

NROWS=$1
if [ -z "$1" ]; then
    echo "Provide nrows"
    exit -1
fi

DOCS_PATH=$2
PAIRS_PATH=$3


AGG_PATH="data/agg.txt"
BERT_PATH="clf"

ANNOTATED_PAIRS_PATH="data/annotated_pairs.jsonl"
SAMPLE_PATH="data/sample.jsonl"

cd headline_cause;

echo "==== Inferring BERT with Monte-Carlo Dropout"
CUDA_VISIBLE_DEVICES=1 python3 -m active_learning.infer_clf \
    --input-path ../$PAIRS_PATH \
    --output-path ../$ANNOTATED_PAIRS_PATH \
    --model-path ../$BERT_PATH;

echo "==== Calculating KNN Density"
#python3 active_learning.calc_density \
#    --docs-path ../$DOCS_PATH \
#    --embedding-key labse_embedding \
#    --input-path ../$ANNOTATED_PAIRS_PATH \
#    --output-path ../$ANNOTATED_PAIRS_PATH \
#    --k 20;

echo "==== Sorting by bald, filtering with RECS"
python3 -m active_learning.filter_pairs\
    --input-path ../$ANNOTATED_PAIRS_PATH \
    --nrows $NROWS \
    --sort-field bald \
    --output-path ../$SAMPLE_PATH \
    --existing-path ../$AGG_PATH \
    --embedding-key labse_embedding \
    --docs-path ../$DOCS_PATH;
#    --use-recs;

echo "Output path: ${SAMPLE_PATH}"
echo "===="
