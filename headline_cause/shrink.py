import sys
import json

from util import write_jsonl, read_jsonl

records_file = sys.argv[1]
output_file = sys.argv[2]

records = read_jsonl(records_file)
has_workers = "worker_id" in records[0]
if has_workers:
    workers = list({r["worker_id"] for r in records})
    worker2num = {wid: i for i, wid in enumerate(workers)}

for r in records:
    r.pop("left_title")
    r.pop("right_title")
    if has_workers:
        r["worker_id"] = worker2num[r["worker_id"]]

records.sort(key=lambda x: x["left_url"])
write_jsonl(records, output_file)
