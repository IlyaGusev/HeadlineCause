import re
import csv
import argparse
import json
from datetime import datetime

from util import read_tsv, get_key

def main(markup, docs, output):
    records = read_tsv(markup)
    docs = {r["url"]: r for r in read_tsv(docs)}
    fixed_records = []
    for r in records:
        rid = r["id"]
        if r["left_url"] not in docs:
            print("Bad url: {}".format(r["left_url"]))
            continue
        if r["right_url"] not in docs:
            print("Bad url: {}".format(r["right_url"]))
            continue
        r["left_timestamp"] = docs[r["left_url"]]["timestamp"]
        r["right_timestamp"] = docs[r["right_url"]]["timestamp"]
        if "_" not in rid or "single" in rid:
            rid = "lenta_" + rid
            r["id"] = rid

        if r["mv_result_cause"] == "left_right" and r["left_timestamp"] > r["right_timestamp"]:
            print("Bad timestamps: {}".format(rid))
            continue
        if r["mv_result_cause"] == "right_left" and r["right_timestamp"] > r["left_timestamp"]:
            print("Bad timestamps: {}".format(rid))
            continue

        fixed_records.append(r)

    with open(output, "w") as w:
        writer = csv.writer(w, delimiter="\t", quotechar='"')
        header = (
            "id", "left_title", "right_title",
            "left_url", "right_url",
            "left_timestamp", "right_timestamp",
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
    args = parser.parse_args()
    main(**vars(args))
