import argparse
import random
import csv
import json
import spacy
from nltk.metrics.distance import edit_distance

from util import get_host, read_jsonl


def normalize_entity(text, spacy_model):
    text = " ".join([token.lemma_ for token in spacy_model(text)])
    return text.lower().replace(".", "")


def main(
    input_path,
    output_path,
    language,
    max_distance,
    min_distance,
    bert_min_confidence,
    bert_labels,
    max_length,
    min_length,
    remove_bad_chars,
    filter_same_hosts,
    min_timestamp_diff,
    remove_same_locations
):
    nlp = spacy.load("en_core_web_sm" if language == "en" else "ru_core_news_sm")
    bert_labels = [int(bert_label) for bert_label in bert_labels.split(",")]
    records = read_jsonl(input_path)

    filtered_records = []
    for r in records:
        if "from_language" in r and r["from_language"] != language:
            continue
        if "to_language" in r and r["to_language"] != language:
            continue
        if "distance" in r and r["distance"] > max_distance:
            continue
        if "distance" in r and r["distance"] < min_distance:
            continue
        bert_label = r.get("bert_label", None)
        bert_confidence = r.get("bert_confidence", None)
        if bert_label is not None and bert_label not in bert_labels:
            continue
        if bert_confidence is not None and bert_confidence < bert_min_confidence:
            continue
        rnd = random.random() < 0.5
        left_title = r.get("from_title", r.get("left_title"))
        right_title = r.get("to_title", r.get("right_title"))

        if len(left_title) > max_length or len(right_title) > max_length:
            continue
        if len(left_title) < min_length or len(right_title) < min_length:
            continue
        if remove_bad_chars:
            bad_chars = (";", ":", "!", "?")
            has_bad_chars = False
            for ch in bad_chars:
                if ch in left_title or ch in right_title:
                    has_bad_chars = True
            if has_bad_chars:
                continue

        left_url = r.get("from_url", r.get("left_url"))
        right_url = r.get("to_url", r.get("right_url"))
        if filter_same_hosts and get_host(left_url) != get_host(right_url):
            continue

        left_timestamp = int(r.get("from_timestamp", r.get("left_timestamp")))
        right_timestamp = int(r.get("to_timestamp", r.get("right_timestamp")))

        if abs(left_timestamp - right_timestamp) < min_timestamp_diff:
            continue

        if remove_same_locations:
            left_entities = nlp(left_title).ents
            right_entities = nlp(right_title).ents
            loc_left_entities = [e for e in left_entities if e.label_ in ("LOC", "GPE")]
            loc_right_entities = [e for e in right_entities if e.label_ in ("LOC", "GPE")]
            loc_left_entities = [normalize_entity(e.text, nlp) for e in loc_left_entities]
            loc_right_entities = [normalize_entity(e.text, nlp) for e in loc_right_entities]
            synonyms = {
                "рф": "россия",
                "луганск": "лнр",
                "ставрополье": "ставропольский край",
                "белоруссия": "беларусь",
                "кремль": "москва",
                "краснодар": "краснодарский край",
                "сша": "америка",
                "киеве": "украина",
                "киев": "украина"
            }
            loc_intersection = set()
            for le in loc_left_entities:
                for re in loc_right_entities:
                    le = synonyms.get(le, le)
                    re = synonyms.get(re, re)
                    if le in re or edit_distance(le, re) < 3:
                        loc_intersection.add(le)
                    elif re in le or edit_distance(le, re) < 3:
                        loc_intersection.add(re)
            if not loc_left_entities or not loc_right_entities or loc_intersection:
                continue
            print(loc_left_entities, loc_right_entities)

        filtered_records.append({
            "id": str(r["id"]),
            "left_title": left_title if rnd else right_title,
            "right_title": right_title if rnd else left_title,
            "left_url": left_url if rnd else right_url,
            "right_url": right_url if rnd else left_url,
            "left_timestamp": left_timestamp if rnd else right_timestamp,
            "right_timestamp": right_timestamp if rnd else left_timestamp,
        })

    with open(output_path, "w") as w:
        writer = csv.writer(w, delimiter="\t")
        header = (
            "id", "left_title", "right_title",
            "left_url", "right_url",
            "left_timestamp", "right_timestamp"
        )
        writer.writerow(header)
        for r in filtered_records:
            writer.writerow([r[key] for key in header])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--max-distance", type=float, default=1.0)
    parser.add_argument("--min-distance", type=float, default=0.0)
    parser.add_argument("--bert-min-confidence", type=float, default=0.0)
    parser.add_argument("--bert-labels", type=str, default="0,1,2")
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--min-length", type=int, default=40)
    parser.add_argument("--remove-bad-chars", action="store_true", default=False)
    parser.add_argument("--filter-same-hosts", action="store_true", default=False)
    parser.add_argument("--remove-same-locations", action="store_true", default=False)
    parser.add_argument("--min-timestamp-diff", type=int, default=-1)
    args = parser.parse_args()
    main(**vars(args))
