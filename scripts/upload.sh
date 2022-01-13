#!/bin/bash

set -e;

SAMPLE_PATH="data/sample.jsonl"
HONEY_PATH="toloka/ru/examples/honey.tsv"

echo "Honey path: ${HONEY_PATH}"
echo "Sample path: ${SAMPLE_PATH}"
echo ""

cd headline_cause;

echo "==== Uploading sample"
python3 -m crowd.upload \
    --input-path ../$SAMPLE_PATH \
    --honey-path ../$HONEY_PATH \
    --template-pool-id 30307627 \
    --overlap 10 \
    --seed 42;
echo "===="
