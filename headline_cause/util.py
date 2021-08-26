import json
import csv
from urllib.parse import urlparse


def write_tsv(records, header, path):
    with open(path, "w") as w:
        writer = csv.writer(w, delimiter="\t", quotechar='"')
        writer.writerow(header)
        for r in records:
            row = [r[key] for key in header]
            writer.writerow(row)


def read_tsv(file_name):
    records = []
    with open(file_name, "r") as r:
        reader = csv.reader(r, delimiter="\t")
        header = next(reader)
        for row in reader:
            record = dict(zip(header, row))
            records.append(record)
    return records


def write_jsonl(records, path):
    with open(path, "w") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(file_name):
    records = []
    with open(file_name, "r") as r:
        for line in r:
            record = json.loads(line)
            records.append(record)
    return records


def get_host(url):
    return '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(url))


def get_key(record):
    return tuple(sorted((record["left_url"], record["right_url"])))
