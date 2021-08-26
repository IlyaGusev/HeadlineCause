import argparse
from collections import defaultdict, Counter
from util import read_tsv

import spacy


def get_normalized_verbs(s, spacy_model, language):
    doc = spacy_model(s)
    negation_words = {
        "ru": ("не", "передумать", "отменить", "отказаться"),
        "en": ("not", )
    }
    negation_tokens = [tok for tok in doc if tok.dep_ == 'neg' or tok.lemma_ in negation_words[language]]
    negated_tokens = [child.lemma_ for token in negation_tokens for child in token.children]
    verbs = []
    for token in doc:
        if token.is_stop:
            continue
        if token.pos_ == "VERB":
            if token.lemma_ in negated_tokens:
                verbs.append(("не_" if language == "ru" else "not_") + token.lemma_)
            else:
                verbs.append(token.lemma_)
    return tuple(verbs)


def pairs2verbs(pairs, spacy_model, lang):
    return tuple([
        (get_normalized_verbs(s1, spacy_model, lang), get_normalized_verbs(s2, spacy_model, lang))
        for s1, s2 in pairs
    ])


def verbs2stats(pairs):
    left_verbs = []
    right_verbs = []
    same_verbs = []
    pairs_verbs = []
    neg_pairs_verbs = []
    for vv1, vv2 in pairs:
        left_verbs.extend(vv1)
        right_verbs.extend(vv2)
        same_verbs.extend(list(set(vv1) & set(vv2)))
        for v1 in vv1:
            for v2 in vv2:
                pairs_verbs.append((v1, v2))
                contains_russian_neg = v1.startswith('не_') or v2.startswith('не_')
                contains_english_neg = v1.startswith('not_') or v2.startswith('not_')
                if (v1 in v2 or v2 in v1) and (contains_russian_neg or contains_english_neg):
                    neg_pairs_verbs.append((v1, v2))
    print('\nleft verbs stats')
    print('\t'+'\n\t'.join(map(str, Counter(left_verbs).most_common(5))))
    print('\nright verbs stats')
    print('\t'+'\n\t'.join(map(str, Counter(right_verbs).most_common(5))))
    print('\nsame verbs stats')
    print('\t'+'\n\t'.join(map(str, Counter(same_verbs).most_common(5))))
    print('\npair verbs stats')
    print('\t'+'\n\t'.join(map(str, Counter(pairs_verbs).most_common(5))))
    print('\nneg pair verbs stats')
    print('\t'+'\n\t'.join(map(str, Counter(neg_pairs_verbs).most_common(5))))


def save_gdf(file_name, pairs):
    nodes = Counter()
    edges = Counter()
    for vv1, vv2 in pairs:
        for v1 in vv1:
            nodes.update([v1, ])
        for v2 in vv2:
            nodes.update([v2, ])
        for v1 in vv1:
            for v2 in vv2:
                edges.update([(v1, v2)])
    with open(file_name, 'w', encoding='utf-8') as fh:
        print('nodedef>name VARCHAR,label VARCHAR,cnt DOUBLE', file=fh)
        for n, c in nodes.items():
            print(f'{n},{n},{c}', file=fh)
        print('edgedef>node1 VARCHAR,node2 VARCHAR,directed BOOLEAN,label VARCHAR', file=fh)
        for nn, l in edges.items():
            n1, n2 = nn
            print(f'{n1},{n2},true,{l}', file=fh)


def main(markup_path, language):
    records = read_tsv(markup_path)
    results = defaultdict(list)
    for r in records:
        result = r["full_result"]
        if float(r["full_agreement"]) < 0.69:
            continue
        left_title, right_title = r["left_title"], r["right_title"]
        if result.startswith("right"):
            left_title, right_title = right_title, left_title
        label = r["full_result"].replace("_", "").replace("left", "").replace("right", "")
        if label == "bad":
            continue
        results[label].append((left_title, right_title))
    verbs = dict()
    spacy_model = spacy.load("en_core_web_sm" if language == "en" else "ru_core_news_sm")
    for label, pairs in results.items():
        verbs[label] = pairs2verbs(pairs, spacy_model, language)
        print()
        print(label)
        verbs2stats(verbs[label])
        save_gdf(language + "_" + label + ".gdf", verbs[label])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("markup_path", type=str)
    parser.add_argument("--language", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
