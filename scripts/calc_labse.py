import sys
import json
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds/norms


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def labse_get_embeddings(sentences, encoder, preprocessor, batch_size=4):
    def get_embeddings(texts):
        return normalization(encoder(preprocessor(texts))["default"])
    embeddings = np.zeros((len(sentences), 768))
    current_index = 0
    for batch in tqdm(gen_batch(sentences, batch_size)):
        batch_embeddings = get_embeddings(batch)
        embeddings[current_index:current_index+batch_size, :] = batch_embeddings
        current_index += batch_size
    return embeddings


def main(input_path, output_path):
    PREPROCESSOR = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2"
    MODEL = "https://tfhub.dev/google/LaBSE/2"
    labse_preprocessor = hub.KerasLayer(PREPROCESSOR)
    labse_encoder = hub.KerasLayer(MODEL)

    sentences = []
    with open(input_path, "r") as r:
        for line in r:
            sentences.append(line.strip())
    embeddings = labse_get_embeddings(sentences, labse_encoder, labse_preprocessor)

    with open(output_path, "w") as w:
        for embedding in embeddings:
            w.write(json.dumps(embedding.tolist()) + "\n")

main("input.txt", "output.txt")
