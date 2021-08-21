import argparse
import csv
from collections import Counter

from util import read_tsv


def main(aggregated_path, raw_path, min_votes_part):
    agg_records = read_tsv(aggregated_path)
    raw_records = read_tsv(raw_path)

    worker_distribution = Counter()
    for r in raw_records:
        worker_distribution[r["worker_id"]] += 1

    for res_key in ("result_cause", "result"):
        votes_distribution = Counter()
        result_distribution = Counter()
        for r in agg_records:
            votes = r["mv_part_{}".format(res_key)]
            result = r["mv_{}".format(res_key)]
            votes_distribution[votes] += 1
            mv_part_is_ok = float(votes) >= min_votes_part
            if mv_part_is_ok:
                result_distribution[result] += 1

        print("MV PART ({}):".format(res_key))
        for votes, count in sorted(votes_distribution.items(), reverse=True):
            print("{}\t{}".format(votes, count))
        print()

        print("RESULT, min MV part ({}): {}".format(res_key, min_votes_part))
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
    args = parser.parse_args()
    main(**vars(args))
