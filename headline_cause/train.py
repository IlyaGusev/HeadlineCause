import random
import os
import json
import copy
import random
from collections import Counter, defaultdict

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from tqdm import tqdm


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_jsonl(file_name, directory="data"):
    records = []
    path = os.path.join(directory, file_name)
    with open(path, "r") as r:
        for line in r:
            record = json.loads(line)
            records.append(record)
    return records


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


set_random_seed(21413)
ru_train_records = read_jsonl("simple_ru_train.jsonl")
ru_val_records = read_jsonl("simple_ru_val.jsonl")
ru_test_records = read_jsonl("simple_ru_test.jsonl")
ru_records = ru_train_records + ru_val_records + ru_test_records

en_train_records = read_jsonl("simple_en_train.jsonl")
en_val_records = read_jsonl("simple_en_val.jsonl")
en_test_records = read_jsonl("simple_en_test.jsonl")
en_records = en_train_records + en_val_records + en_test_records

ru_full_train_records = read_jsonl("full_ru_train.jsonl")
ru_full_val_records = read_jsonl("full_ru_val.jsonl")
ru_full_test_records = read_jsonl("full_ru_test.jsonl")
ru_full_records = ru_full_train_records + ru_full_val_records + ru_full_test_records

en_full_train_records = read_jsonl("full_en_train.jsonl")
en_full_val_records = read_jsonl("full_en_val.jsonl")
en_full_test_records = read_jsonl("full_en_test.jsonl")
en_full_records = en_full_train_records + en_full_val_records + en_full_test_records

ru_labels_counter = Counter([r["label"] for r in ru_records])
print(ru_labels_counter, sum(ru_labels_counter.values()))
en_labels_counter = Counter([r["label"] for r in en_records])
print(en_labels_counter, sum(en_labels_counter.values()))

ru_labels_counter_full = Counter([r["label"] for r in ru_full_records])
print(ru_labels_counter_full, sum(ru_labels_counter_full.values()))
en_labels_counter_full = Counter([r["label"] for r in en_full_records])
print(en_labels_counter_full, sum(en_labels_counter_full.values()))

labels_count = len(ru_labels_counter + en_labels_counter)
labels_count_full = len(ru_labels_counter_full + en_labels_counter_full)

ru_aug_train_records, ru_aug_val_records = augment(ru_train_records), augment(ru_val_records)
print("RU:")
print(len(ru_aug_train_records))
print(len(ru_aug_val_records))
print(len(ru_test_records))
for r in ru_aug_train_records[:2]:
    print(r)
print()

en_aug_train_records, en_aug_val_records = augment(en_train_records), augment(en_val_records)
print("EN:")
print(len(en_aug_train_records))
print(len(en_aug_val_records))
print(len(en_test_records))
for r in en_aug_train_records[:2]:
    print(r)

ru_full_aug_train_records, ru_full_aug_val_records = augment(ru_full_train_records, task="full"), augment(ru_full_val_records, task="full")
print("RU:")
print(len(ru_full_aug_train_records))
print(len(ru_full_aug_val_records))
print(len(ru_full_test_records))
for r in ru_full_aug_train_records[:2]:
    print(r)
print()

en_full_aug_train_records, en_full_aug_val_records = augment(en_full_train_records, task="full"), augment(en_full_val_records, task="full")
print("EN:")
print(len(en_full_aug_train_records))
print(max([r["right_timestamp"] for r in en_full_aug_train_records]))
print(len(en_full_aug_val_records))
print(max([r["right_timestamp"] for r in en_full_aug_val_records]))
print(len(en_full_test_records))
print(max([r["right_timestamp"] for r in en_full_test_records]))
for r in en_full_aug_train_records[:2]:
    print(r)

MODEL_NAME = "xlm-roberta-large"#@param {type:"string"}
TOKENIZER_NAME = MODEL_NAME
MAX_TOKENS = 60#@param {type:"number"}
EPOCHS = 4#@param {type:"number"}
EVAL_STEPS = 32#@param {type:"number"}
WARMUP_STEPS = 16#@param {type:"number"}
LR = 0.00002#@param {type:"number"}
BATCH_SIZE = 8#@param {type:"number"}
GRAD_ACCUM_STEPS = 16#@param {type:"number"}
PATIENCE = 3#@param {type:"number"}

train_records = ru_full_aug_train_records + en_full_aug_train_records
val_records = ru_full_aug_val_records + en_full_aug_val_records
random.shuffle(train_records)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, do_lower_case=False)
train_data = NewsPairsDataset(train_records, tokenizer, MAX_TOKENS)
val_data = NewsPairsDataset(val_records, tokenizer, MAX_TOKENS)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=labels_count_full)
model = model.to("cuda")

callbacks = [EarlyStoppingCallback(early_stopping_patience=PATIENCE)]

training_args = TrainingArguments(
    output_dir="checkpoints",
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=EVAL_STEPS,
    save_steps=EVAL_STEPS,
    warmup_steps=WARMUP_STEPS,
    learning_rate=LR,
    num_train_epochs=EPOCHS,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
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
