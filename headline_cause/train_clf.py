import argparse
import json
import random
from statistics import mean

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from scipy.stats import entropy
from sklearn.metrics import classification_report

from augment import augment
from util import read_jsonl


class PairsDataset(Dataset):
    def __init__(self, records, max_tokens, tokenizer, result_key):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.records = list()
        for r in records:
            inputs = self.embed_pair(r)
            inputs["labels"] = r["label"]
            self.records.append(inputs)

    def __len__(self):
        return len(self.records)

    def embed_pair(self, pair):
        inputs = self.tokenizer(
            text=pair["left_title"],
            text_pair=pair["right_title"],
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="max_length",
            truncation="longest_first",
            return_tensors="pt"
        )
        return {key: value.squeeze(0) for key, value in inputs.items()}

    def __getitem__(self, index):
        return self.records[index]


def calc_labels(records, result_key):
    full_result_mapping = {
        "bad": 0,
        "same": 1,
        "rel": 2,
        "left_right_cause": 3,
        "right_left_cause": 4,
        "left_right_refute": 5,
        "right_left_refute": 6
    }
    simple_result_mapping = {
        "not_cause": 0,
        "left_right": 1,
        "right_left": 2
    }
    result_mapping = simple_result_mapping if "simple" in result_key else full_result_mapping
    for r in records:
        r["label"] = result_mapping[r[result_key]]
    return records


def main(
    input_path,
    min_agreement,
    max_tokens,
    model_name,
    epochs,
    eval_steps,
    warmup_steps,
    lr,
    task,
    batch_size,
    grad_accum_steps,
    seed,
    out_dir
):
    res_key = "{}_result".format(task)
    agreement_key = "{}_agreement".format(task)

    records = read_jsonl(input_path)
    records = [r for r in records if r[agreement_key] > min_agreement]
    records = calc_labels(records, res_key)

    random.seed(seed)
    random.shuffle(records)

    val_border = int(len(records) * 0.8)
    test_border = int(len(records) * 0.9)
    train_records = records[:val_border]
    val_records = records[val_border:test_border]
    test_records = records[test_border:]
    train_records, val_records = augment(train_records, task), augment(val_records, task)

    labels_count = len({r[res_key] for r in train_records})
    assert labels_count >= 3

    print("Train records: ", len(train_records))
    print("Val records: ", len(val_records))
    print("Test records: ", len(test_records))
    print("Labels: ", labels_count)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    train_dataset = PairsDataset(train_records, max_tokens, tokenizer, res_key)
    val_dataset = PairsDataset(val_records, max_tokens, tokenizer, res_key)
    test_dataset = PairsDataset(test_records, max_tokens, tokenizer, res_key)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=labels_count)
    model = model.to(device)

    training_args = TrainingArguments(
        output_dir="checkpoints",
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=eval_steps,
        save_steps=eval_steps,
        warmup_steps=warmup_steps,
        learning_rate=lr,
        num_train_epochs=epochs,
        gradient_accumulation_steps=grad_accum_steps,
        report_to="none",
        load_best_model_at_end=True,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()

    model.save_pretrained(out_dir)
    train_dataset.tokenizer.save_pretrained(out_dir)

    y_true, y_pred = [], []
    true_entropies = []
    false_entropies = []
    with torch.no_grad():
        for item in test_dataset:
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            mask = item["attention_mask"].unsqueeze(0).to(device)
            label = item["labels"]
            outputs = model(input_ids, mask, return_dict=True)
            logits = outputs.logits[0]
            pred = torch.argmax(logits).item()

            y_pred.append(pred)
            y_true.append(label)

            scores = torch.softmax(logits, dim=0).cpu().numpy()
            ent = entropy(scores)
            if pred == label:
                true_entropies.append(ent)
            else:
                false_entropies.append(ent)
    print("Avg true entropy: {}".format(mean(true_entropies)))
    if false_entropies:
        print("Avg false entropy: {}".format(mean(false_entropies)))
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--min-agreement", type=float, default=0.69)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, choices=("full", "simple"), default="simple")
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--model-name", type=str, default="xlm-roberta-large")
    args = parser.parse_args()
    main(**vars(args))
