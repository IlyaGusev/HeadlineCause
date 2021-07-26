import argparse
import csv
from collections import Counter


def read_tsv(path):
    records = []
    with open(path, "r") as r:
        reader = csv.reader(r, delimiter="\t")
        header = next(reader)
        for row in reader:
            record = dict(zip(header, row))
            records.append(record)
    return records


def main(aggregated_path, raw_path, border):
    confidence_distribution = Counter()
    result_distribution = Counter()
    worker_distribution = Counter()

    agg_records = read_tsv(aggregated_path)
    raw_records = read_tsv(raw_path)

    for r in agg_records:
        confidence_distribution[r["confidence"]] += 1
        if float(r["confidence"]) >= border:
            result_distribution[r["result"]] += 1

    for r in raw_records:
        worker_distribution[r["worker_id"]] += 1

    print("CONFIDENCE:")
    for confidence, count in sorted(confidence_distribution.items(), reverse=True):
        print("{}\t{}".format(confidence, count))
    print()

    print("RESULT, border {}:".format(border))
    for result, count in sorted(result_distribution.items()):
        print("{}\t{}".format(count, result))
    print()

    print("WORKERS:")
    print("count\t{}".format(len(worker_distribution)))
    print("avg\t{}".format(sum(worker_distribution.values()) / len(worker_distribution)))
    print("max\t{}".format(max(worker_distribution.values())))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregated_path", type=str)
    parser.add_argument("raw_path", type=str)
    parser.add_argument("--border", type=float, default=0.8)
    args = parser.parse_args()
    main(**vars(args))
