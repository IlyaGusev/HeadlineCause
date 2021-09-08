import argparse
import os
from collections import defaultdict, Counter

import toloka.client as toloka
from nltk.metrics.agreement import AnnotationTask
from crowdkit.aggregation import DawidSkene
import pandas as pd

from util import get_key, write_jsonl


def unquote(title):
    if title[0] == title[-1] == '"':
        fixed_title = title[1:-1]
    else:
        fixed_title = title
    fixed_title = fixed_title.replace('""', '"')
    return title


def aggregate(records, res_key, overlap=10, min_agreement=0.7):
    results = defaultdict(list)
    records.sort(key=lambda x: x["assignment_id"])
    for r in records:
        results[get_key(r)].append(r[res_key + "_result"])

    for key, votes in results.items():
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
            "{}_result".format(res_key): res_win,
            "{}_agreement".format(res_key): votes_part
        })

    answers = [(str(hash(get_key(r))), r[res_key + "_result"], r["worker_id"]) for r in records]
    answers_df = pd.DataFrame(answers, columns=["task", "label", "performer"])
    proba = DawidSkene(n_iter=20).fit_predict_proba(answers_df)
    labels = proba.idxmax(axis=1)
    for key in data:
        ds_key = str(hash(key))
        label = labels[ds_key]
        confidence = proba.loc[ds_key, label]
        data[key].update({
            "{}_ds_result".format(res_key): label,
            "{}_ds_confidence".format(res_key): confidence
        })

    print("Aggregation ({}): ".format(res_key))
    total_samples = sum(votes_distribution.values())
    sum_agreement = 0
    for v, sample_count in sorted(votes_distribution.items(), reverse=True):
        print("{}: {}".format(v, sample_count))
        sum_agreement += sample_count * v
    print("Total: ", total_samples)
    print("Average agreement:", sum_agreement / total_samples)

    answers = [(r["worker_id"], get_key(r), r[res_key + "_result"]) for r in records]
    t = AnnotationTask(data=answers)
    print("Krippendorff’s alpha: {}".format(t.alpha()))

    answers = [
        (r["worker_id"], get_key(r), r[res_key + "_result"])
        for r in records if votes[get_key(r)] >= min_agreement
    ]
    t = AnnotationTask(data=answers)
    print("Krippendorff’s alpha, border {}: {}".format(min_agreement, t.alpha()))
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

    results_mapping = {
        "left_right_cancel": "left_right_refute",
        "right_left_cancel": "right_left_refute"
    }

    simple_mapping = {
        "bad": "not_cause",
        "rel": "not_cause",
        "same": "not_cause",
        "left_right_cause": "left_right",
        "left_right_refute": "left_right",
        "right_left_cause": "right_left",
        "right_left_refute": "right_left"
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
                result = output_values["result"]
                result = results_mapping.get(result, result)
                record = {
                    "full_result": result,
                    "simple_result": simple_mapping[result],
                    "worker_id": assignment.user_id,
                    "assignment_id": assignment.id
                }
                for field in input_fields:
                    record[field] = input_values[field]
                record["left_title"] = unquote(record["left_title"])
                record["right_title"] = unquote(record["right_title"])
                records.append(record)

    agg_records = aggregate(records, "full")
    agg_records_simple = aggregate(records, "simple")
    for key, r in agg_records.items():
        r.update(agg_records_simple[key])

    agg_records = list(agg_records.values())
    agg_records.sort(key=lambda x: (x["full_agreement"], x["left_url"]), reverse=True)
    agg_header = [
        "full_result", "full_agreement", "full_ds_result", "full_ds_confidence",
        "simple_result", "simple_agreement", "simple_ds_result", "simple_ds_confidence"
    ]
    agg_header += input_fields
    agg_records = [{key: r[key] for key in agg_header} for r in agg_records]
    write_jsonl(agg_records, agg_output)

    raw_records = records
    raw_header = ["full_result", "worker_id", "assignment_id"] + input_fields
    raw_records = [{key: r[key] for key in raw_header} for r in raw_records]
    write_jsonl(raw_records, raw_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-fields", type=str, default="left_title,right_title,left_url,right_url")
    parser.add_argument("--token", type=str, default="~/.toloka/token")
    parser.add_argument("--agg-output", type=str, required=True)
    parser.add_argument("--raw-output", type=str, required=True)
    parser.add_argument("--pools-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
