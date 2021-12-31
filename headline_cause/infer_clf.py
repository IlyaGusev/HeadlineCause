import argparse
import json

import torch
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from util import write_jsonl, read_jsonl

def main(
    input_path,
    model_path,
    output_path,
    max_tokens=300,
    models_count=5
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=300)
    model.train() # Monte-Carlo Dropout
    records = read_jsonl(input_path)
    for r in tqdm(records):
        s1 = r["left_title"]
        s2 = r["right_title"]
        inputs = tokenizer(
            text=s1,
            text_pair=s2,
            add_special_tokens=True,
            max_length=max_tokens,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt"
        )
        inputs.to(device)
        with torch.no_grad():
            all_scores = []
            for _ in range(models_count):
                output = model(**inputs)
                logits = output.logits.squeeze(0).cpu().numpy()
                scores = softmax(logits)
                all_scores.append(scores.tolist())
            avg_scores = []
            class_count = len(all_scores[0])
            for c in range(class_count):
                c_score = float(np.mean([scores[c] for scores in all_scores]))
                avg_scores.append(c_score)
            entropy_over_avg = float(entropy(avg_scores))
            entropies = [float(entropy(scores)) for scores in all_scores]
            avg_entropy = float(np.mean(entropies))
            bald_score = entropy_over_avg - avg_entropy
            r["entropy"] = entropy_over_avg
            r["avg_entropy"] = avg_entropy
            r["bald"] = bald_score
            r["scores"] = avg_scores
    write_jsonl(output_path, records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))


