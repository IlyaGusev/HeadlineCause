import argparse
import copy
import random
from collections import Counter

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from datasets import load_dataset
from tqdm import tqdm

from augment import augment
from util import set_random_seed
from predict import pipe_predict

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
    seed,
    model_name,
    max_tokens,
    epochs,
    eval_steps,
    warmup_steps,
    lr,
    batch_size,
    grad_accum_steps,
    patience,
    use_augment = True
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
            datasets[lang]["train"] = augment(datasets[lang]["train"], task)
            datasets[lang]["validation"] = augment(datasets[lang]["validation"], task)

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
    if torch.cuda.is_available():
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
        device=(0 if torch.cuda.is_available() else -1),
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="xlm-roberta-large")
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    main(**vars(args))
