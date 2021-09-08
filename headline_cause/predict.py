import argparse

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np

from util import read_jsonl, write_jsonl


def get_batch(data, batch_size):
    start_index = 0
    while start_index < len(data):
        end_index = start_index + batch_size
        batch = data[start_index:end_index]
        yield batch
        start_index = end_index


def pipe_predict(data, pipe, batch_size=64):
    raw_preds = []
    for batch in tqdm(get_batch(data, batch_size), desc="predict"):
        raw_preds += pipe(batch)
    preds = np.array([int(max(labels, key=lambda x: x["score"])["label"][-1]) for labels in raw_preds])
    pp = np.array([[label["score"] for label in labels] for labels in raw_preds])
    return preds, pp


def predict(model_path, input_path, output_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=(0 if torch.cuda.is_available() else -1),
        return_all_scores=True
    )
    data = read_jsonl(input_path)
    test_pairs = [(r["left_title"], r["right_title"]) for r in data]
    labels, probs = pipe_predict(test_pairs, pipe)
    for r, l, p in zip(data, labels, probs):
        r["pred_label"] = int(l)
        r["pred_prob"] = float(p[l])
    write_jsonl(data, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    predict(**vars(args))
