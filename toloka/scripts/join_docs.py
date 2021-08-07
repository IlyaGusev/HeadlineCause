import re
import csv
import argparse
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


def main(output, tg_data, lenta_data=None):
    records = {r["id"]: r for r in read_tsv(tg_data)}
    if lenta_data:
        lenta_records = {"lenta_" + r["id"]: r for r in read_tsv(lenta_data)}
        for rid, r in lenta_records.items():
            r["left_timestamp"] = get_date(r["left_url"])
            r["right_timestamp"] = get_date(r["right_url"])
            records[rid] = r
    docs = dict()
    for rid, r in records.items():
        docs[r["left_url"]] = {
            "title": r["left_title"],
            "timestamp": r["left_timestamp"],
            "url": r["left_url"]
        }
        docs[r["right_url"]] = {
            "title": r["right_title"],
            "timestamp": r["right_timestamp"],
            "url": r["right_url"]
        }
    docs = list(docs.values())
    with open(output, "w") as w:
        writer = csv.writer(w, delimiter="\t", quotechar='"')
        header = (
            "title", "url", "timestamp"
        )
        writer.writerow(header)
        for r in docs:
            writer.writerow([r[key] for key in header])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tg-data", type=str, required=True)
    parser.add_argument("--lenta-data", type=str, default=None)
    args = parser.parse_args()
    main(**vars(args))
