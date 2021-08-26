import sys
import json

from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from toloka.util import read_jsonl, write_jsonl


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


def _get_embeddings(texts, encoder, preprocessor):
    return normalization(encoder(preprocessor(texts))["default"])


def labse_get_embeddings(sentences, encoder, preprocessor, batch_size=64):
    embeddings = np.zeros((len(sentences), 768))
    current_index = 0
    for batch in tqdm(gen_batch(sentences, batch_size)):
        batch_embeddings = _get_embeddings(batch, encoder, preprocessor)
        embeddings[current_index:current_index+batch_size, :] = batch_embeddings
        current_index += batch_size
    return embeddings


def labse_get_embedding(sentence, encoder, preprocessor):
    return _get_embeddings([sentence], encoder, preprocessor)[0]


PREPROCESSOR = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2"
MODEL = "https://tfhub.dev/google/LaBSE/2"
labse_preprocessor = hub.KerasLayer(PREPROCESSOR)
labse_encoder = hub.KerasLayer(MODEL)

input_path = sys.argv[1]
output_path = sys.argv[2]

records = read_jsonl(input_path)
titles = set()
for record in records:
    titles.add(record["left_title"])
    titles.add(record["right_title"])
titles = list(titles)

embeddings = labse_get_embeddings(titles, labse_encoder, labse_preprocessor)
embeddings = {title: embedding for title, embedding in zip(titles, embeddings)}

for r in records:
    left_title = r["left_title"]
    right_title = r["right_title"]
    left_embedding = embeddings[left_title]
    right_embedding = embeddings[right_title]
    distance = cosine(left_embedding, right_embedding)
    r["labse_cosine_distance"] = distance

write_jsonl(records, output_path)
