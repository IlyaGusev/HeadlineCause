import argparse
from collections import defaultdict

from tqdm import tqdm

from util import read_jsonl, write_jsonl


def read_records(file_name, min_agreement=0.69, task="simple"):
    orig_records = read_jsonl(file_name)
    records = []
    for r in tqdm(orig_records):
        if float(r[task + "_agreement"]) < min_agreement:
            continue
        if task == "simple":
            mapping = {
                "not_cause": 0,
                "left_right": 1,
                "right_left": 2
            }
        else:
            assert task == "full"
            mapping = {
                "bad": 0,
                "same": 1,
                "rel": 2,
                "left_right_cause": 3,
                "right_left_cause": 4,
                "left_right_refute": 5,
                "right_left_refute": 6
            }
        result = r[task + "_result"]
        r["label"] = mapping[result]
        records.append(r)
    return records


def split_with_source(records, val_border=0.8, test_border=0.9):
    records_by_source = defaultdict(list)
    for r in tqdm(records):
        source = r["id"].split("_")[1]
        assert source in ("tg", "lenta")
        records_by_source[source].append(r)

    train_records, val_records, test_records = [], [], []
    for source, source_records in records_by_source.items():
        print(source, len(source_records))
        source_records.sort(key=lambda x: max(x["left_timestamp"], x["right_timestamp"]))
        val_border_idx = int(val_border * len(source_records))
        test_border_idx = int(test_border * len(source_records))
        train_records.extend(source_records[:val_border_idx])
        val_records.extend(source_records[val_border_idx:test_border_idx])
        test_records.extend(source_records[test_border_idx:])
    return train_records, val_records, test_records


def main(input_path, train, val, test, task):
    records = read_records(input_path, task=task)
    train_records, val_records, test_records = split_with_source(records)
    write_jsonl(train_records, train)
    write_jsonl(val_records, val)
    write_jsonl(test_records, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
