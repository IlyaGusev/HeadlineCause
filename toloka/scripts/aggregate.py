import argparse
import os
import csv
from collections import defaultdict, Counter

import toloka.client as toloka


def aggregate(records, border):
    results = defaultdict(list)
    for r in records:
        results[r["id"]].append(r["result"])

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
            data[key]["res"] = res_win
        else:
            data.pop(key)
    for confidence, sample_count in sorted(confidence_distribution.items(), reverse=True):
        print("{}: {}".format(confidence, sample_count))
    return data


def main(border, token, output, pools_file):
    pool_ids = []
    with open(pools_file, "r") as r:
        for line in r:
            pool_id = line.strip()
            if not pool_id:
                continue
            pool_id = int(pool_id)
            pool_ids.append(pool_id)

    with open(os.path.expanduser(token), "r") as r:
        toloka_token = r.read().strip()

    toloka_client = toloka.TolokaClient(toloka_token, 'PRODUCTION')
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
                key = input_values["id"]
                records.append({
                    "id": input_values["id"],
                    "left_title": input_values["left_title"],
                    "right_title": input_values["right_title"],
                    "left_url": input_values["left_url"],
                    "right_url": input_values["right_url"],
                    "worker_id": assignment.user_id,
                    "result": solution.output_values["result"],
                    "assignment_id": assignment.id
                })

    agg_records = aggregate(records, border)
    with open(output, "w") as w:
        header = ["left_title", "right_title", "result", "confidence", "id", "left_url", "right_url"]
        writer = csv.writer(w, delimiter="\t", quotechar='"')
        writer.writerow(header)
        for _, r in agg_records.items():
            row = [r[key] for key in header]
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--border", type=float, default=0.0)
    parser.add_argument("--token", type=str, default="~/.toloka/token")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--pools-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
