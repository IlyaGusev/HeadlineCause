import argparse
import os
import csv
from collections import defaultdict, Counter

import toloka.client as toloka


def aggregate(records, border, res_key):
    results = defaultdict(list)
    for r in records:
        results[r["id"]].append(r[res_key])

    data = {r["id"]: r for r in records}
    confidence_distribution = Counter()
    for key, res in results.items():
        res_count = Counter(res)
        overlap = len(res)
        res_win, votes_win = res_count.most_common(1)[0]
        confidence = float(votes_win) / overlap
        confidence_distribution[confidence] += 1
        if confidence >= border:
            data[key]["confidence"] = confidence
            data[key]["overlap"] = overlap
            data[key][res_key] = res_win
        else:
            data.pop(key)
    for confidence, sample_count in sorted(confidence_distribution.items(), reverse=True):
        print("{}: {}".format(confidence, sample_count))
    data = list(data.values())
    data.sort(key=lambda x : (x["confidence"], int(x["id"])), reverse=True)
    return data


def write_tsv(records, header, path):
    with open(path, "w") as w:
        writer = csv.writer(w, delimiter="\t", quotechar='"')
        writer.writerow(header)
        for r in records:
            row = [r[key] for key in header]
            writer.writerow(row)


def main(
    border,
    token,
    agg_output,
    raw_output,
    pools_file,
    key_field,
    res_field,
    input_fields
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
                    res_field: output_values[res_field],
                    "worker_id": assignment.user_id,
                    "assignment_id": assignment.id
                }
                for field in input_fields:
                    record[field] = input_values[field]
                records.append(record)

    agg_records = aggregate(records, border, res_field)
    agg_header = [key_field, res_field, "confidence", "overlap"] + input_fields
    write_tsv(agg_records, agg_header, agg_output)

    raw_records = records
    raw_header = [key_field, res_field, "worker_id", "assignment_id"] + input_fields
    write_tsv(raw_records, raw_header, raw_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--border", type=float, default=0.0)
    parser.add_argument("--key-field", type=str, default="id")
    parser.add_argument("--res-field", type=str, default="result")
    parser.add_argument("--input-fields", type=str, required=True)
    parser.add_argument("--token", type=str, default="~/.toloka/token")
    parser.add_argument("--agg-output", type=str, required=True)
    parser.add_argument("--raw-output", type=str, required=True)
    parser.add_argument("--pools-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
