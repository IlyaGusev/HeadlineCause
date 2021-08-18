import argparse
import os
import csv
from collections import defaultdict, Counter

import pandas as pd
from crowdkit.aggregation import DawidSkene
import toloka.client as toloka


def aggregate(records, res_key):
    results = defaultdict(list)
    for r in records:
        results[r["id"]].append(r[res_key])

    data = {r["id"]: r for r in records}
    confidence_distribution = Counter()
    votes_distribution = Counter()
    for key, res in results.items():
        res_count = Counter(res)
        overlap = len(res)
        res_win, votes_win = res_count.most_common(1)[0]
        votes_part = float(votes_win) / overlap
        votes_distribution[votes_part] += 1
        data[key].update({
            "mv_{}".format(res_key): res_win,
            "mv_part_{}".format(res_key): votes_part,
            "overlap": overlap
        })

    answers = [(r["id"], r[res_key], r["worker_id"]) for r in records]
    answers_df = pd.DataFrame(answers, columns=["task", "label", "performer"])
    proba = DawidSkene(n_iter=20).fit_predict_proba(answers_df)
    labels = proba.idxmax(axis=1)
    index = list(proba.index)
    for key in index:
        label = labels[key]
        confidence = proba.loc[key, label]
        confidence_distribution[float(int(confidence * 10) / 10)] += 1
        data[key].update({
            "ds_{}".format(res_key): label,
            "ds_confidence_{}".format(res_key): confidence
        })

    print()
    print("Aggregation field:", res_key)
    print("Dawid-Skene: ")
    for confidence, sample_count in sorted(confidence_distribution.items(), reverse=True):
        print("{}: {}".format(confidence, sample_count))

    print()
    print("Majority vote: ")
    for votes, sample_count in sorted(votes_distribution.items(), reverse=True):
        print("{}: {}".format(votes, sample_count))

    data = {key: r for key, r in data.items()}
    return data


def write_tsv(records, header, path):
    with open(path, "w") as w:
        writer = csv.writer(w, delimiter="\t", quotechar='"')
        writer.writerow(header)
        for r in records:
            row = [r[key] for key in header]
            writer.writerow(row)


def main(
    token,
    agg_output,
    raw_output,
    pools_file,
    key_field,
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
                    key_field: input_values[key_field],
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
    agg_records.sort(key=lambda x : (x["mv_part_result"], str(x["id"])), reverse=True)
    agg_header = [
        key_field, "mv_result", "mv_part_result",
        "ds_result", "ds_confidence_result",
        "mv_result_cause", "mv_part_result_cause",
        "ds_result_cause", "ds_confidence_result_cause"
    ]
    agg_header += input_fields
    write_tsv(agg_records, agg_header, agg_output)

    raw_records = records
    raw_header = [key_field, "result", "worker_id", "assignment_id"] + input_fields
    write_tsv(raw_records, raw_header, raw_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-field", type=str, default="id")
    parser.add_argument("--input-fields", type=str, required=True)
    parser.add_argument("--token", type=str, default="~/.toloka/token")
    parser.add_argument("--agg-output", type=str, required=True)
    parser.add_argument("--raw-output", type=str, required=True)
    parser.add_argument("--pools-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
