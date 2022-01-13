import argparse
import json

import torch
import numpy as np
from scipy.stats import entropy
from scipy.special import softmax
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from util import write_jsonl, read_jsonl, gen_batch


def main(
    input_path,
    model_path,
    output_path,
    max_tokens,
    models_count,
    batch_size
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)
    model.train() # Monte-Carlo Dropout

    output_records = []
    records = list(read_jsonl(input_path))
    for batch in tqdm(gen_batch(records, batch_size)):
        s1 = [r["left_title"] for r in batch]
        s2 = [r["right_title"] for r in batch]
        inputs = tokenizer(
            text=s1,
            text_pair=s2,
            add_special_tokens=True,
            max_length=max_tokens,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt"
        )
        inputs = inputs.to(device)

        num_labels = model.num_labels
        all_scores = torch.zeros((len(batch), models_count, num_labels))
        with torch.no_grad():
            for model_num in range(models_count):
                output = model(**inputs)
                logits = output.logits
                scores = torch.softmax(logits, dim=1).cpu()
                all_scores[:, model_num, :] = scores

        for sample_num in range(len(batch)):
            sample = batch[sample_num]
            sample_scores = all_scores[sample_num]

            avg_scores = torch.mean(sample_scores, dim=0).tolist()
            entropy_over_avg = float(entropy(avg_scores))

            entropies = [float(entropy(scores)) for scores in sample_scores]
            avg_entropy = float(np.mean(entropies))

            bald_score = entropy_over_avg - avg_entropy

            sample["entropy"] = entropy_over_avg
            sample["avg_entropy"] = avg_entropy
            sample["bald"] = bald_score
            sample["scores"] = avg_scores
            output_records.append(sample)
    write_jsonl(output_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--models-count", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=60)
    args = parser.parse_args()
    main(**vars(args))
