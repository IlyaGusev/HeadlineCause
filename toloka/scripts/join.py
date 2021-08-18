import re
import csv
import argparse
import json
from datetime import datetime


def read_tsv(file_name):
    records = []
    with open(file_name, "r") as r:
        reader = csv.reader(r, delimiter="\t")
        header = next(reader)
        for row in reader:
            record = dict(zip(header, row))
            records.append(record)
    return records


def main(markup, docs, output, min_votes_part, min_confidence):
    records = read_tsv(markup)
    docs = {r["url"]: r for r in read_tsv(docs)}
    fixed_records = []
    for r in records:
        rid = r["id"]
        if float(r["ds_confidence_result_cause"]) < min_confidence:
            continue
        if float(r["mv_part_result_cause"]) < min_votes_part:
            continue
        if r["left_url"] not in docs:
            print("Bad url: {}".format(r["left_url"]))
            continue
        if r["right_url"] not in docs:
            print("Bad url: {}".format(r["right_url"]))
            continue
        r["left_timestamp"] = docs[r["left_url"]]["timestamp"]
        r["right_timestamp"] = docs[r["right_url"]]["timestamp"]
        if "_" not in r["id"] or "single" in r["id"]:
            r["id"] = "lenta_" + r["id"]

        if r["mv_result_cause"] == "left_right" and r["left_timestamp"] > r["right_timestamp"]:
            print("Bad timestamps: {}".format(r["id"]))
            continue
        if r["mv_result_cause"] == "right_left" and r["right_timestamp"] > r["left_timestamp"]:
            print("Bad timestamps: {}".format(r["id"]))
            continue

        fixed_records.append(r)

    with open(output, "w") as w:
        writer = csv.writer(w, delimiter="\t", quotechar='"')
        header = (
            "id", "left_title", "right_title",
            "left_url", "right_url",
            "left_timestamp", "right_timestamp",
            "ds_confidence_result", "ds_result",
            "ds_confidence_result_cause", "ds_result_cause",
            "mv_part_result", "mv_result",
            "mv_part_result_cause", "mv_result_cause"
        )
        writer.writerow(header)
        for r in fixed_records:
            writer.writerow([r[key] for key in header])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markup", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--docs", type=str, required=True)
    parser.add_argument("--min-votes-part", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()
    main(**vars(args))
