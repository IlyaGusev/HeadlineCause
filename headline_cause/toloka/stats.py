import argparse
from collections import Counter

from util import read_tsv, read_jsonl, get_host


def normalize_entity(text, nlp):
    text = " ".join([token.lemma_ for token in nlp(text)])
    return text.lower().replace(".", "")


def main(
    aggregated_path,
    raw_path,
    min_votes_part,
    language,
    enable_spacy,
    docs_path
):
    agg_records = read_tsv(aggregated_path)
    raw_records = read_tsv(raw_path)

    worker_distribution = Counter()
    for r in raw_records:
        worker_distribution[r["worker_id"]] += 1

    for res_key in ("result_cause", "result"):
        votes_distribution = Counter()
        result_distribution = Counter()
        for r in agg_records:
            votes = r["mv_part_{}".format(res_key)]
            result = r["mv_{}".format(res_key)]
            votes_distribution[votes] += 1
            mv_part_is_ok = float(votes) >= min_votes_part
            if mv_part_is_ok:
                result_distribution[result] += 1

        print("MV PART ({})".format(res_key))
        for votes, count in sorted(votes_distribution.items(), reverse=True):
            print("{}\t{}".format(votes, count))
        print()

        print("RESULT, min MV part ({}), {}".format(res_key, min_votes_part))
        for result, count in sorted(result_distribution.items()):
            print("{}\t{}".format(count, result))
        print("{}\t{}".format(sum(result_distribution.values()), "all"))
        print()

    print("WORKERS")
    print("count\t{}".format(len(worker_distribution)))
    print("avg\t{}".format(sum(worker_distribution.values()) / len(worker_distribution)))
    print("max\t{}".format(max(worker_distribution.values())))
    print()

    if enable_spacy:
        import spacy
        nlp = spacy.load("en_core_web_sm" if language == "en" else "ru_core_news_sm")
        left_loc_distribution = Counter()
        right_loc_distribution = Counter()

    same_host_count = 0
    link_count = 0
    bad_ts_count = 0
    bad_ts_mv_count = 0
    bad_ts_mv_sh_count = 0
    left_host_distribution = Counter()
    right_host_distribution = Counter()
    docs = {r["url"]: r for r in read_jsonl(docs_path)}
    lenta_timestamps = []
    tg_timestamps = []
    for r in agg_records:
        left_url = r["left_url"]
        right_url = r["right_url"]
        left_host = get_host(left_url)
        right_host = get_host(right_url)
        if left_host == right_host:
            same_host_count += 1

        is_lenta = left_host == right_host == "https://lenta.ru/"
        left_host_distribution[left_host] += 1
        right_host_distribution[right_host] += 1

        left_doc = docs[left_url]
        right_doc = docs[right_url]
        left_ts = left_doc["timestamp"]
        right_ts = right_doc["timestamp"]
        timestamps = lenta_timestamps if is_lenta else tg_timestamps
        timestamps.append(max(left_ts, right_ts))
        votes = r["mv_part_result_cause"]
        result = r["mv_result_cause"]
        mv_part_is_ok = float(votes) >= min_votes_part
        if result == "left_right" and left_ts > right_ts:
            bad_ts_count += 1
            if mv_part_is_ok:
                bad_ts_mv_count += 1
                if left_host == right_host:
                    bad_ts_mv_sh_count += 1
        if result == "right_left" and right_ts > left_ts:
            bad_ts_count += 1
            if mv_part_is_ok:
                bad_ts_mv_count += 1
                if left_host == right_host:
                    bad_ts_mv_sh_count += 1
        link_count += int(right_url in left_doc["links"] or left_url in right_doc["links"])
        if enable_spacy:
            left_entities = nlp(r["left_title"]).ents
            right_entities = nlp(r["right_title"]).ents
            loc_labels = ("LOC", "GPE")
            loc_left_entities = [normalize_entity(e.text, nlp) for e in left_entities if e.label_ in loc_labels]
            left_loc_distribution.update(loc_left_entities)
            loc_right_entities = [normalize_entity(e.text, nlp) for e in right_entities if e.label_ in loc_labels]
            right_loc_distribution.update(loc_right_entities)

    timestamps.sort()
    print("OTHER STATS")
    print("80 percintile ts\t{}".format(timestamps[int(len(tg_timestamps) * 8 // 10)]))
    print("90 percintile ts\t{}".format(timestamps[int(len(tg_timestamps) * 9 // 10)]))
    print("records count\t{}".format(len(agg_records)))
    print("link count\t{}".format(link_count))
    print("same host count\t{}".format(same_host_count))
    print("bad ts count\t{}".format(bad_ts_count))
    print("bad ts count (high agreement)\t{}".format(bad_ts_mv_count))
    print("bad ts count (high agreement, same host)\t{}".format(bad_ts_mv_sh_count))
    print("top 5 left hosts:")
    for host, count in left_host_distribution.most_common(5):
        print("{}\t{}".format(count, host))
    print("top 5 right hosts:")
    for host, count in right_host_distribution.most_common(5):
        print("{}\t{}".format(count, host))
    if enable_spacy:
        print("top 5 left locations:")
        for loc, count in left_loc_distribution.most_common(5):
            print("{}\t{}".format(count, loc))
        print("top 5 right locations:")
        for loc, count in right_loc_distribution.most_common(5):
            print("{}\t{}".format(count, loc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregated_path", type=str)
    parser.add_argument("raw_path", type=str)
    parser.add_argument("docs_path", type=str)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--min-votes-part", type=float, default=0.7)
    parser.add_argument("--enable-spacy", action="store_true", default=False)
    args = parser.parse_args()
    main(**vars(args))
