import random
import sys
import csv
import json
from urllib.parse import urlparse


def get_host(url):
    return '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(url))

language = sys.argv[3]
records = []
with open(sys.argv[1], "r") as rf:
    for line in rf:
        r = json.loads(line)
        if r["from_language"] != language or r["to_language"] != language:
            continue
        #if r["distance"] > 0.3:
        #    continue
        if False:
            bert_label = r.get("bert_label", None)
            bert_confidence = r.get("bert_confidence", None)
            if bert_label is not None and bert_label == 0:
                continue
            if bert_confidence is not None and bert_confidence < 0.8:
                continue
        rnd = random.random() < 0.5
        left_title = r.get("from_title", r.get("left_title"))
        right_title = r.get("to_title", r.get("right_title"))

        #if len(left_title) > 130 or len(right_title) > 130:
        #    continue
        #if len(left_title) < 40 or len(right_title) < 40:
        #    continue
        #bad_chars = (";", ":", "!", "?") 
        #has_bad_chars = False
        #for ch in bad_chars:
        #    if ch in left_title or ch in right_title:
        #        has_bad_chars = True
        #if has_bad_chars:
        #    continue

        left_url = r.get("from_url", r.get("left_url"))
        right_url = r.get("to_url", r.get("right_url"))
        #if get_host(left_url) != get_host(right_url): 
        #    continue

        left_timestamp = r.get("from_timestamp", r.get("left_timestamp"))
        right_timestamp = r.get("to_timestamp", r.get("right_timestamp"))

        #if abs(left_timestamp - right_timestamp) < 7200:
        #    continue

        records.append({
            "id": "tg_" + str(r["id"]),
            "left_title": left_title if rnd else right_title,
            "right_title": right_title if rnd else left_title,
            "left_url": left_url if rnd else right_url,
            "right_url": right_url if rnd else left_url,
            "left_timestamp": left_timestamp if rnd else right_timestamp,
            "right_timestamp": right_timestamp if rnd else left_timestamp,
        })

with open(sys.argv[2], "w") as w:
    writer = csv.writer(w, delimiter="\t")
    header = (
        "id", "left_title", "right_title",
        "left_url", "right_url",
        "left_timestamp", "right_timestamp"
    )
    writer.writerow(header)
    for r in records:
        writer.writerow([r[key] for key in header])
