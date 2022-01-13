import argparse

from annoy import AnnoyIndex
import numpy as np
from tqdm import tqdm

from util import read_jsonl, write_jsonl


def main(
    docs_path,
    input_path,
    output_path,
    embedding_key,
    k
):
    print("Reading docs...")
    docs = list(read_jsonl(docs_path))
    embedding_size = len(docs[0][embedding_key])
    docs = {doc["url"]: doc for doc in docs}

    print("Reading records...")
    index = AnnoyIndex(embedding_size * 2, 'angular')
    records = read_jsonl(input_path)
    for i, r in enumerate(tqdm(records)):
        doc1 = docs[r["left_url"]]
        doc2 = docs[r["right_url"]]
        embedding = doc1[embedding_key] + doc2[embedding_key]
        index.add_item(i, embedding)

    print("Building index...")
    index.build(1000)

    print("Inferring density...")
    for i, r in enumerate(tqdm(records)):
        neighbours, distances = index.get_nns_by_item(i, k, include_distances=True)
        similarities = -(np.power(distances, 2) / 2.0 - 1.0)
        avg_sim = np.mean(similarities)
        records[i]["density"] = avg_sim
        if "bald" in records[i]:
            records[i]["bald_density"] = records[i]["bald"] * avg_sim
        if "entropy" in records[i]:
            records[i]["entropy_density"] = records[i]["entropy"] * avg_sim

    print("Writing results...")
    write_jsonl(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs-path', type=str, required=True)
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--embedding-key', type=str, default="labse_embedding")
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()
    main(**vars(args))
