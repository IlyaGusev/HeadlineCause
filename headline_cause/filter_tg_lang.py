import json
import sys

with open(sys.argv[1], "r") as r, open(sys.argv[2], "w") as w:
    for line in r:
        record = json.loads(line)
        from_language = record["from_language"]
        to_language = record["to_language"]
        if from_language != "en" or to_language != "en":
            continue
        w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
