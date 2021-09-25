import json
import sys

docs_file = sys.argv[1]
authors_file = sys.argv[2]
raw_file = sys.argv[3]
output_file = sys.argv[4]

def read_jsonl(file_name):
    records = []
    with open(file_name) as r:
        for line in r:
            record = json.loads(line)
            records.append(record)
    return records

docs = read_jsonl(docs_file)
authors = read_jsonl(authors_file)
records = read_jsonl(raw_file)
used_urls = {r["left_url"] for r in records}
used_urls = used_urls.union({r["right_url"] for r in records})

authors = {r["url"]: r["authors"] for r in authors}
filtered_docs = dict()
bad_authors_cnt = 0
for doc in docs:
    url = doc["url"]
    if url not in used_urls:
        continue
    doc["authors"] = authors.get(url, None)
    if not doc["authors"]:
        bad_authors_cnt += 1
    filtered_docs[url] = doc
print("Bad authors:", bad_authors_cnt)

assert len(used_urls) == len(filtered_docs), "{} vs {}".format(len(used_urls), len(filtered_docs))
with open(output_file, "w") as w:
    for url, doc in filtered_docs.items():
        w.write(json.dumps(doc, ensure_ascii=False).strip() + "\n")
