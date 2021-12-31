import argparse

import numpy as np
from faiss import Kmeans, IndexFlatL2

from util import read_jsonl, write_jsonl


def main(
    input_path,
    output_path,
    n_clusters
):
    records = read_jsonl(input_path)
    assert records
    print("{} samples read".format(len(records)))

    d = len(records[0]["left_embedding"]) * 2
    features_matrix = np.zeros((len(records), d), dtype=np.float32)
    for i, record in enumerate(records):
        features_matrix[i] = record["left_embedding"] + record["right_embedding"]
    print("Matrix {}x{}".format(*features_matrix.shape))

    clustering = Kmeans(d=d, k=n_clusters, verbose=True, nredo=5, niter=30)
    clustering.train(features_matrix)
    index = IndexFlatL2(d)
    index.add(features_matrix)
    center_points = index.search(clustering.centroids, 1)[1]
    center_points = np.squeeze(center_points)

    best_records = [records[index] for index in center_points]
    print("{} samples selected".format(len(best_records)))
    write_jsonl(best_records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--n-clusters', type=int, required=True)
    args = parser.parse_args()
    main(**vars(args))
