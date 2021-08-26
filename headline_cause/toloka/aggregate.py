import argparse
import os
import random
from collections import defaultdict, Counter

import toloka.client as toloka
from nltk.metrics.agreement import AnnotationTask

from util import get_key, write_tsv


def aggregate(records, res_key, overlap=10, min_part_alpha=0.7):
    results = defaultdict(list)
    for r in records:
        results[get_key(r)].append(r[res_key])

    for key, votes in results.items():
        random.shuffle(votes)
        results[key] = votes[:overlap]

    data = {get_key(r): r for r in records}
    votes_distribution = Counter()
    votes = dict()
    for key, res in results.items():
        res_count = Counter(res)
        overlap = len(res)
        res_win, votes_win = res_count.most_common(1)[0]
        votes_part = float(votes_win) / overlap
        votes_distribution[votes_part] += 1
        votes[key] = votes_part
        data[key].update({
            "mv_{}".format(res_key): res_win,
            "mv_part_{}".format(res_key): votes_part,
            "overlap": overlap
        })

    print("Aggregation ({}): ".format(res_key))
    total_samples = sum(votes_distribution.values())
    sum_agreement = 0
    for v, sample_count in sorted(votes_distribution.items(), reverse=True):
        print("{}: {}".format(v, sample_count))
        sum_agreement += sample_count * v
    print("Total: ", total_samples)
    print("Average agreement:", sum_agreement / total_samples)

    answers = [(r["worker_id"], get_key(r), r[res_key]) for r in records]
    t = AnnotationTask(data=answers)
    print("Krippendorff’s alpha: {}".format(t.alpha()))

    answers = [
        (r["worker_id"], get_key(r), r[res_key])
        for r in records if votes[get_key(r)] >= min_part_alpha
    ]
    t = AnnotationTask(data=answers)
    print("Krippendorff’s alpha, border {}: {}".format(min_part_alpha, t.alpha()))
    print()

    data = {key: r for key, r in data.items()}
    return data


def main(
    token,
    agg_output,
    raw_output,
    pools_file,
    input_fields,
):
    input_fields = input_fields.split(",")

    with open(os.path.expanduser(token), "r") as r:
        toloka_token = r.read().strip()
    toloka_client = toloka.TolokaClient(toloka_token, 'PRODUCTION')

    pool_ids = []
    with open(pools_file, "r") as r:
        for line in r:
            pool_id = line.strip()
            if not pool_id:
                continue
            pool_id = int(pool_id)
            pool_ids.append(pool_id)

    mapping = {
        "bad": "not_cause",
        "rel": "not_cause",
        "same": "not_cause",
        "left_right_cause": "left_right",
        "left_right_cancel": "left_right",
        "right_left_cause": "right_left",
        "right_left_cancel": "right_left"
    }
    records = []
    for pool_id in pool_ids:
        for assignment in toloka_client.get_assignments(pool_id=pool_id):
            solutions = assignment.solutions
            if not solutions:
                continue
            for task, solution in zip(assignment.tasks, solutions):
                known_solutions = task.known_solutions
                if known_solutions is not None:
                    continue
                input_values = task.input_values
                output_values = solution.output_values
                record = {
                    "result": output_values["result"],
                    "result_cause": mapping[output_values["result"]],
                    "worker_id": assignment.user_id,
                    "assignment_id": assignment.id
                }
                for field in input_fields:
                    record[field] = input_values[field]
                records.append(record)

    agg_records = aggregate(records, "result")
    agg_records_cause = aggregate(records, "result_cause")
    for key, r in agg_records.items():
        r.update(agg_records_cause[key])

    agg_records = list(agg_records.values())
    agg_records.sort(key=lambda x: (x["mv_part_result"], x["left_url"]), reverse=True)
    agg_header = [
        "mv_result", "mv_part_result",
        "mv_result_cause", "mv_part_result_cause",
    ]
    agg_header += input_fields
    write_tsv(agg_records, agg_header, agg_output)

    raw_records = records
    raw_header = ["result", "worker_id", "assignment_id"] + input_fields
    write_tsv(raw_records, raw_header, raw_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fields", type=str, default="left_title,right_title,left_url,right_url")
    parser.add_argument("--token", type=str, default="~/.toloka/token")
    parser.add_argument("--agg-output", type=str, required=True)
    parser.add_argument("--raw-output", type=str, required=True)
    parser.add_argument("--pools-file", type=str, required=True)
    args = parser.parse_args()
    random.seed(42)
    main(**vars(args))
