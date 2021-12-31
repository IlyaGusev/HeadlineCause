import gc
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tqdm import tqdm

from util import gen_batch

DEFAULT_ENCODER_PATH = "https://tfhub.dev/google/LaBSE/2"
DEFAULT_PREPROCESSOR_PATH = "https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2"


def normalization(embeds):
    norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
    return embeds / norms


class LaBSE:
    def __init__(self, encoder_path=DEFAULT_ENCODER_PATH, preprocessor_path=DEFAULT_PREPROCESSOR_PATH):
        self.preprocessor = hub.KerasLayer(preprocessor_path)
        self.model = hub.KerasLayer(encoder_path)

    def embed_batch(self, texts):
        return normalization(self.model(self.preprocessor(texts))["default"])

    def labse_get_embeddings(self, sentences, batch_size):
        embeddings = np.zeros((len(sentences), 768))
        current_index = 0
        for batch in gen_batch(sentences, batch_size):
            batch_embeddings = self.embed_batch(batch)
            embeddings[current_index:current_index+batch_size, :] = batch_embeddings
            current_index += batch_size
            tf.keras.backend.clear_session()
            gc.collect()
        return embeddings

    def __call__(self, sentences, batch_size=16):
        return self.labse_get_embeddings(sentences, batch_size)
