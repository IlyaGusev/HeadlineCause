import argparse
import hashlib
import random
from collections import Counter

import spacy
from tqdm import tqdm
from nltk.stem.snowball import SnowballStemmer

from util import read_jsonl, write_jsonl
from crowd.util import get_key


BAD_POS = ("PREP", "NPRO", "CONJ", "PRCL", "NUMR", "PRED", "INTJ", "PUNCT", "CCONJ", "ADP")


def tokenize(text, spacy_model, stemmer):
    text = text.replace("\xa0", " ")
    analysis = spacy_model(text)
    tokens = [token.lemma_ for token in analysis if token.pos_ not in BAD_POS]
    tokens = [stemmer.stem(token) for token in tokens]
    tokens = [token for token in tokens if len(token) > 2]
    return tokens


def calc_ngrams(text, n=3):
    text = text.replace("\xa0", " ").lower()
    ngrams = []
    for word in text.split():
        ngrams.extend([word[i:i+n] for i in range(len(word)-n)])
    return ngrams


def pair_from_docs(doc1, doc2, id_prefix):
    r = {
        "left_title": doc1["title"],
        "right_title": doc2["title"],
        "left_url": doc1["url"],
        "right_url": doc2["url"],
        "left_timestamp": doc1["timestamp"],
        "right_timestamp": doc2["timestamp"]
    }
    key = get_key(r)
    md5 = hashlib.md5((key[0] + key[1]).encode('utf-8')).hexdigest()
    r["id"] = id_prefix + "_" + md5
    return r


def main(
    docs_path,
    output_path,
    id_prefix,
    sample_rate,
    min_common_ngrams,
    stop_ngrams_count,
    min_common_tokens,
    stop_tokens_count,
    max_timestamp_diff,
    max_left_doc_samples,
    ndocs
):
    docs = read_jsonl(docs_path)
    docs.sort(key=lambda x: x["timestamp"])
    if ndocs:
        docs = docs[-ndocs:]
    random.shuffle(docs)
    print("{} documents read".format(len(docs)))

    ngrams_cnt = Counter()
    for doc in tqdm(docs):
        ngrams = calc_ngrams(doc["title"])
        ngrams_cnt.update(ngrams)
        doc["ngrams"] = set(ngrams)
    common_ngrams = {ngram for ngram, _ in ngrams_cnt.most_common(stop_ngrams_count)}
    print(common_ngrams)
    for doc in docs:
        doc["ngrams"] = doc["ngrams"].difference(common_ngrams)

    spacy_model = spacy.load("ru_core_news_md")
    stemmer = SnowballStemmer("russian")
    tokens_cnt = Counter()
    for doc in tqdm(docs):
        tokens = tokenize(doc["title"], spacy_model, stemmer)
        tokens_cnt.update(tokens)
        doc["tokens"] = set(tokens)
    common_tokens = {token for token, _ in tokens_cnt.most_common(stop_tokens_count)}
    print(common_tokens)
    for doc in docs:
        doc["tokens"] = doc["tokens"].difference(common_tokens)

    n = len(docs)
    true_n_pairs = int(n * (n - 1) // 2)
    pbar = tqdm(total=true_n_pairs)
    all_count = 0
    filtered_count = 0
    records = []
    for doc1_num, doc1 in enumerate(docs):
        left_doc_samples_count = 0
        for doc2_num in range(doc1_num + 1, len(docs)):
            pbar.update(1)
            doc2 = docs[doc2_num]

            all_count += 1
            if max_timestamp_diff:
                if abs(doc1["timestamp"] - doc2["timestamp"]) > max_timestamp_diff:
                    filtered_count += 1
                    continue

            if min_common_ngrams:
                intersection = doc1["ngrams"].intersection(doc2["ngrams"])
                if len(intersection) < min_common_ngrams:
                    filtered_count += 1
                    continue

            if min_common_tokens:
                intersection = doc1["tokens"].intersection(doc2["tokens"])
                if len(intersection) < min_common_tokens:
                    filtered_count += 1
                    continue

            if random.random() > sample_rate:
                continue

            left_doc_samples_count += 1

            if random.random() < 0.5:
                doc1, doc2 = doc2, doc1

            record = pair_from_docs(doc1, doc2, id_prefix)
            records.append(record)
            if max_left_doc_samples and left_doc_samples_count >= max_left_doc_samples:
                break

    pbar.close()
    print("{}% pairs filtered".format(filtered_count / all_count * 100.0))
    print("{} pairs sampled".format(len(records)))

    write_jsonl(records, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--sample-rate', type=float, required=True)
    parser.add_argument('--min-common-ngrams', type=int, default=0)
    parser.add_argument('--stop-ngrams-count', type=int, default=50)
    parser.add_argument('--min-common-tokens', type=int, default=0)
    parser.add_argument('--stop-tokens-count', type=int, default=100)
    parser.add_argument('--max-timestamp-diff', type=int, default=86400*365)
    parser.add_argument('--max-left-doc-samples', type=int, default=4)
    parser.add_argument('--ndocs', type=int, default=None)
    parser.add_argument('--id-prefix', type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))

