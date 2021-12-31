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

from util import read_jsonl


class PairsDataset(Dataset):
    def __init__(self, records, max_tokens, tokenizer, result_key="simple_result"):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.records = list()
        self.result_mapping = {
            "not_cause": 0,
            "left_right": 1,
            "right_left": 2
        }
        for pair in records:
            input_ids, mask = self.embed_pair(pair)
            self.records.append((input_ids, mask, self.result_mapping.get(pair[result_key])))

    def __len__(self):
        return len(self.records)

    def embed_pair(self, pair):
        inputs = self.tokenizer(
            text=pair["left_title"],
            text_pair=pair["right_title"],
            add_special_tokens=True,
            max_length=self.max_tokens,
            padding="max_length",
            truncation="longest_first"
        )
        return torch.LongTensor(inputs["input_ids"]), torch.FloatTensor(inputs["attention_mask"])

    def __getitem__(self, index):
        input_ids, attention_mask, label = self.records[index]
        output = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": [label]}
        return output


def main(
    input_path,
    min_agreement,
    max_tokens,
    model_name,
    epochs,
    eval_steps,
    warmup_steps,
    lr,
    batch_size,
    grad_accum_steps,
    seed,
    out_dir
):
    records = read_jsonl(input_path)
    records = [r for r in records if r["simple_agreement"] > min_agreement]

    random.seed(seed)
    random.shuffle(records)
    val_border = int(len(records) * 0.8)
    test_border = int(len(records) * 0.9)

    train_records = records[:val_border]
    random.shuffle(train_records)
    val_records = records[val_border:test_border]
    test_records = records[test_border:]
    assert len({r["simple_result"] for r in test_records}) >= 2

    print("Train records: ", len(train_records))
    print("Val records: ", len(val_records))
    print("Test records: ", len(test_records))

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    train_dataset = PairsDataset(train_records, max_tokens, tokenizer)
    val_dataset = PairsDataset(val_records, max_tokens, tokenizer)
    test_dataset = PairsDataset(test_records, max_tokens, tokenizer)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
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
            label = item["labels"][0]
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
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--min-agreement", type=float, default=0.69)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-05)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--model-name", type=str, default="DeepPavlov/rubert-base-cased")
    args = parser.parse_args()
    main(**vars(args))


