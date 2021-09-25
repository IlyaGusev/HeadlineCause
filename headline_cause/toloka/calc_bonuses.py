import sys
from collections import Counter

from util import read_jsonl

raw_files = sys.argv[1], sys.argv[2]
workers = Counter()
for f in raw_files:
    workers.update([r["worker_id"] for r in read_jsonl(f)])

overall = 0
with open("bonuses.tsv", "w") as w:
    for worker_id, count in workers.most_common():
        bonus = count * 0.003 * 1.5
        overall += bonus
        w.write("{}\t{}\t{}\n".format(worker_id, bonus, "-"))
print(f"Overall: {overall}$")
