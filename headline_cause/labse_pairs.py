import sys
import json
import copy

from scipy.spatial.distance import cosine
from tqdm import tqdm

from labse import LaBSE
from util import read_jsonl, write_jsonl


input_path = sys.argv[1]
output_path = sys.argv[2]

records = read_jsonl(input_path)
titles = set()
for record in tqdm(records):
    titles.add(record["left_title"])
    titles.add(record["right_title"])

titles = list(titles)
print(len(titles))

model = LaBSE()
embeddings = model(titles)
title2index = {title: index for index, title in enumerate(titles)}

with open(output_path, "w") as w:
    for r in tqdm(records):
        left_title = r["left_title"]
        right_title = r["right_title"]
        new_record = copy.copy(r)
        left_embedding = embeddings[title2index[left_title]]
        right_embedding = embeddings[title2index[right_title]]
        new_record["labse_embedding"] = [float(v) for v in left_embedding] + [float(v) for v in right_embedding]
        new_record["labse_cosine_distance"] = cosine(left_embedding, right_embedding)
        w.write(json.dumps(new_record, ensure_ascii=False).strip() + "\n")
