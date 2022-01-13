import random
import copy

import numpy as np


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
            if "left_timestamp" in r:
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
