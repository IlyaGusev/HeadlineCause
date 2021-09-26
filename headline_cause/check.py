import sys
from util import read_jsonl

records_file = sys.argv[1]
docs_file = sys.argv[2]

records = read_jsonl(records_file)
docs = read_jsonl(docs_file)
urls = {r["url"] for r in docs}

for record in records:
    assert record["left_url"] in urls
    assert record["right_url"] in urls

print("OK")
