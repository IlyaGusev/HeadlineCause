import csv
import sys
import random

records = []
header = ("id", "has_lenta_link", "has_cluster_link", "first_title", "second_title", "first_url", "second_url")
with open(sys.argv[1], "r") as r:
    reader = csv.reader(r, delimiter="\t")
    for row in reader:
        r = dict(zip(header, row))
        has_lenta_link = int(r.pop("has_lenta_link")) == 1
        has_cluster_link = int(r.pop("has_cluster_link")) == 1
        if not has_lenta_link:
            continue
        if random.random() < 0.5:
            r["left_title"], r["right_title"] = r.pop("second_title"), r.pop("first_title")
            r["left_url"], r["right_url"] = r.pop("second_url"), r.pop("first_url")
        else:
            r["left_title"], r["right_title"] = r.pop("first_title"), r.pop("second_title")
            r["left_url"], r["right_url"] = r.pop("first_url"), r.pop("second_url")
        records.append(r)

with open(sys.argv[2], "w") as w:
    writer = csv.writer(w, delimiter="\t")
    header = ("id", "left_title", "right_title", "left_url", "right_url")
    writer.writerow(header)
    for r in records:
        writer.writerow([r[key] for key in header])
