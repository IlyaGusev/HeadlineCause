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


def main(aggregated_path, raw_path, min_confidence, min_votes_part):
    votes_distribution = Counter()
    result_distribution = Counter()
    worker_distribution = Counter()

    agg_records = read_tsv(aggregated_path)
    raw_records = read_tsv(raw_path)

    for r in agg_records:
        votes_distribution[r["mv_part_result_cause"]] += 1
        confidence_is_ok = float(r["ds_confidence_result_cause"]) >= min_confidence
        mv_part_is_ok = float(r["mv_part_result_cause"]) >= min_votes_part
        if confidence_is_ok and mv_part_is_ok:
            result_distribution[r["mv_result_cause"]] += 1

    for r in raw_records:
        worker_distribution[r["worker_id"]] += 1

    print("MV PART:")
    for votes, count in sorted(votes_distribution.items(), reverse=True):
        print("{}\t{}".format(votes, count))
    print()

    print("RESULT, min confidence {}:, min mv part: {}".format(min_confidence, min_votes_part))
    for result, count in sorted(result_distribution.items()):
        print("{}\t{}".format(count, result))
    print("{}\t{}".format(sum(result_distribution.values()), "all"))
    print()

    print("WORKERS:")
    print("count\t{}".format(len(worker_distribution)))
    print("avg\t{}".format(sum(worker_distribution.values()) / len(worker_distribution)))
    print("max\t{}".format(max(worker_distribution.values())))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregated_path", type=str)
    parser.add_argument("raw_path", type=str)
    parser.add_argument("--min-votes-part", type=float, default=0.7)
    parser.add_argument("--min-confidence", type=float, default=0.99)
    args = parser.parse_args()
    main(**vars(args))
