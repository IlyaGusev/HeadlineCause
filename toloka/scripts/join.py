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


def get_date(url):
    dates = re.findall(r"\d\d\d\d\/\d\d\/\d\d", url)
    date = str(next(iter(dates), None))
    return int(datetime.strptime(date, "%Y/%m/%d").timestamp())


def main(markup, output, tg_data, lenta_data):
    records = read_tsv(markup)
    tg_records = {r["id"]: r for r in read_tsv(tg_data)}
    lenta_records = {r["id"]: r for r in read_tsv(lenta_data)}
    fixed_records = []
    for r in records:
        rid = r["id"]
        if rid not in tg_records and rid not in lenta_records:
            print("Record not found: {}".format(rid))
            continue
        r["confidence"] = float(r["confidence"])
        if r["confidence"] < 0.69:
            continue
        is_tg = rid in tg_records
        orig_r = tg_records[rid] if is_tg else lenta_records[rid]

        is_inverted = orig_r["left_url"] == r["right_url"]
        if is_inverted:
            assert orig_r["left_url"] == r["right_url"]
            assert orig_r["right_url"] == r["left_url"]
        else:
            assert orig_r["left_url"] == r["left_url"]
            assert orig_r["right_url"] == r["right_url"]
        if is_tg:
            left_timestamp = orig_r["left_timestamp"]
            right_timestamp = orig_r["right_timestamp"]
        else:
            left_timestamp = get_date(orig_r["left_url"])
            right_timestamp = get_date(orig_r["right_url"])
        r["left_timestamp"] = right_timestamp if is_inverted else left_timestamp
        r["right_timestamp"] = left_timestamp if is_inverted else right_timestamp

        if not is_tg:
            r["id"] = "lenta_" + r["id"]

        if r["result"].startswith("left_right") and r["left_timestamp"] > r["right_timestamp"]:
            print("Bad timestamps: {}".format(r["id"]))
            continue
        if r["result"].startswith("right_left") and r["right_timestamp"] > r["left_timestamp"]:
            print("Bad timestamps: {}".format(r["id"]))
            continue

        fixed_records.append(r)

    with open(output, "w") as w:
        writer = csv.writer(w, delimiter="\t", quotechar='"')
        header = (
            "id", "left_title", "right_title",
            "left_url", "right_url", "left_timestamp",
            "right_timestamp", "confidence", "result",
            "overlap"
        )
        writer.writerow(header)
        for r in fixed_records:
            writer.writerow([r[key] for key in header])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markup", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tg-data", type=str, required=True)
    parser.add_argument("--lenta-data", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
