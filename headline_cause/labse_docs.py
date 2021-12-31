import argparse
import json
import copy

from tqdm import tqdm

from labse import LaBSE
from util import read_jsonl, write_jsonl, gen_batch


def main(
    input_path,
    output_path,
    ndocs,
    batch_size
):
    model = LaBSE()
    batch_size = 8
    docs = read_jsonl(input_path)
    docs.sort(key=lambda x: x["timestamp"])
    docs = docs[-ndocs:]
    with open(output_path, "w") as w:
        for batch in tqdm(gen_batch(docs, batch_size)):
            titles = [r["title"] for r in batch]
            embeddings = model(titles, batch_size)
            for r, embedding in zip(batch, embeddings):
                r["labse_embedding"] = [float(v) for v in embedding]
                w.write(json.dumps(r, ensure_ascii=False).strip() + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--ndocs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    main(**vars(args))
