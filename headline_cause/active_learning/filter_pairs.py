import argparse
import random

from scipy.spatial.distance import cosine
from tqdm import tqdm

from crowd.util import get_key
from util import read_jsonl, write_jsonl


def main(
    input_path,
    docs_path,
    output_path,
    existing_path,
    sort_field,
    nrows,
    seed,
    use_recs,
    embedding_key
):
    print("Reading records...")
    records = read_jsonl(input_path)

    if existing_path:
        print("Filtering existing records...")
        existing_keys = {get_key(r) for r in read_jsonl(existing_path)}
        print("...{} existing keys".format(len(existing_keys)))
        filtered_records = [r for r in records if get_key(r) not in existing_keys]
        print("...{} records filtered".format(len(records) - len(filtered_records)))
        print("...{} records remained".format(len(filtered_records)))
        records = filtered_records

    random.seed(seed)
    random.shuffle(records)
    if sort_field:
        print("Sorting by {}...".format(sort_field))
        records.sort(key=lambda x: x[sort_field], reverse=True)

    # Redundancy Elimination by Cosine Similarity
    if use_recs:
        print("Using RECS...")
        docs = {d["url"]: d for d in read_jsonl(docs_path)}
        filtered_records = list()
        initial_threshold = 1.0
        threshold_step = (initial_threshold - 0.7) / (nrows * 1.5)
        threshold = initial_threshold
        for r in tqdm(records):
            d1 = docs[r["left_url"]]
            d2 = docs[r["right_url"]]
            embedding1 = d1[embedding_key] + d2[embedding_key]
            r["embedding"] = embedding1
            max_sim = 0.0
            for r2 in filtered_records:
                embedding2 = r2["embedding"]
                sim = cosine(embedding1, embedding2)
                max_sim = max(max_sim, sim)
            if max_sim < threshold:
                filtered_records.append(r)
                threshold -= threshold_step
                if len(filtered_records) == nrows:
                    break
        records = [r for r in filtered_records]
        for r in records:
            r.pop("embedding")
        print("...{} records after RECS, {} final threshold".format(len(records), threshold))

    records = records[:nrows]

    print("Writing results...")
    write_jsonl(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--docs-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--existing-path", type=str, default=None)
    parser.add_argument("--nrows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-recs", action="store_true", default=False)
    parser.add_argument("--embedding-key", type=str, default=None)
    parser.add_argument("--sort-field", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))


