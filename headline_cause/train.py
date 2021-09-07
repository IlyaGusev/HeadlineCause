import argparse
import random
import os
import json
import copy
import random
from collections import Counter, defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from datasets import load_dataset
from tqdm import tqdm

from util import set_random_seed, read_jsonl


def make_symmetrical(records, prob, task):
    new_records = []
    for r in records:
        new_records.append(r)
        if random.random() <= prob:
            new_record = copy.copy(r)
            new_record["left_url"] = r["right_url"]
            new_record["right_url"] = r["left_url"]
            new_record["left_title"] = r["right_title"]
            new_record["right_title"] = r["left_title"]
            new_record["left_timestamp"] = r["right_timestamp"]
            new_record["right_timestamp"] = r["left_timestamp"]
            if task == "simple":
                mapping = {
                    1: 2,
                    2: 1
                }
            else:
                assert task == "full"
                mapping = {
                    3: 4,
                    4: 3,
                    5: 6,
                    6: 5
                }
            if r["label"] in mapping:
                new_record["label"] = mapping[r["label"]]
            new_record["is_inverted"] = 1
            new_records.append(new_record)
    return new_records


def checklist_add_typos(string, typos=1):
    string = list(string)
    swaps = np.random.choice(len(string) - 1, typos)
    for swap in swaps:
        tmp = string[swap]
        string[swap] = string[swap + 1]
        string[swap + 1] = tmp
    return ''.join(string)


def add_typos(records, prob):
    new_records = []
    for r in records:
        new_records.append(r)
        new_r = copy.copy(r)
        is_added = False
        if random.random() <= prob:
            new_r["left_title"] = str(checklist_add_typos(r["left_title"]))
            is_added = True
        if random.random() <= prob:
            new_r["right_title"] = str(checklist_add_typos(r["right_title"]))
            is_added = True
        if is_added:
            new_r["has_misspell"] = 1
            new_records.append(new_r)
    return new_records


def augment(records, task="simple"):
    records = make_symmetrical(records, 1.0, task)
    records = add_typos(records, 0.05)
    return records


class NewsPairsDataset(Dataset):
    def __init__(self, records, tokenizer, max_tokens):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.records = records

    def __len__(self):
        return len(self.records)

    def embed_record(self, record):
        inputs = self.tokenizer(
            text=record["left_title"],
            text_pair=record["right_title"],
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="max_length",
            truncation="longest_first",
            return_tensors='pt'
        )
        for key, value in inputs.items():
            value.squeeze_(0)
        return inputs

    def __getitem__(self, index):
        record = self.records[index]
        output = self.embed_record(record)
        label = record.get("label", None)
        if label is not None:
            output["labels"] = torch.tensor(label)
        return output

def main(
    task,
    out_dir,
    seed=32,
    use_augment=True,
    model_name = "xlm-roberta-large",
    max_tokens = 60,
    epochs = 4,
    eval_steps = 32,
    warmup_steps = 16,
    lr = 0.00002,
    batch_size = 8,
    grad_accum_steps = 16,
    patience = 3
):
    languages = ("ru", "en")
    assert task in ("simple", "full")
    set_random_seed(seed)

    datasets = dict()
    for lang in languages:
        dataset = load_dataset("IlyaGusev/headline_cause", "{}_{}".format(lang, task))
        datasets[lang] = {
            "train": list(dataset["train"]),
            "validation": list(dataset["validation"]),
            "test": list(dataset["test"])
        }

    labels_counter = Counter()
    for lang in languages:
        labels_counter += Counter([r["label"] for r in datasets[lang]["train"]])
    labels_count = len(labels_counter)

    if use_augment:
        for lang in languages:
            datasets[lang]["train"] = augment(datasets[lang]["train"])
            datasets[lang]["validation"] = augment(datasets[lang]["validation"])

    train_records = []
    val_records = []
    for lang in languages:
        print("{}:".format(lang))
        print(len(datasets[lang]["train"]))
        print(len(datasets[lang]["validation"]))
        print(len(datasets[lang]["test"]))
        for r in datasets[lang]["train"][:2]:
            print(r)
        print()
        train_records += datasets[lang]["train"]
        val_records += datasets[lang]["validation"]

    random.shuffle(train_records)

    tokenizer_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
    train_data = NewsPairsDataset(train_records, tokenizer, max_tokens)
    val_data = NewsPairsDataset(val_records, tokenizer, max_tokens)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=labels_count)
    model = model.to("cuda")

    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)]

    training_args = TrainingArguments(
        output_dir=out_dir,
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
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=callbacks
    )

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    model.eval()
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=0,
        return_all_scores=True
    )

    for lang in languages:
        y_true = np.array([r["label"] for r in datasets[lang]["test"]], dtype=np.int32)
        test_pairs = [(r["left_title"], r["right_title"]) for r in datasets[lang]["test"]]
        y_pred, y_pred_prob = pipe_predict(test_pairs, pipe)
        print("{}:".format(lang))
        print(classification_report(y_true, y_pred, digits=3))
        print(confusion_matrix(y_true, y_pred))
        print("Binary AUC: {}".format(roc_auc_score([int(l == 0) for l in y_true], [p[0] for p in y_pred_prob])))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=("simple", "full"), required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
