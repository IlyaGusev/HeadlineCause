import csv
import argparse

from util import read_tsv, get_host, read_jsonl


def main(markup, docs, output, language):
    records = read_tsv(markup)
    docs = {r["url"]: r for r in read_jsonl(docs)}
    fixed_records = []
    bad_ts_count = 0
    bad_url_count = 0
    for r in records:
        left_url = r["left_url"]
        right_url = r["right_url"]
        if left_url not in docs:
            print("Bad url: {}".format(left_url))
            bad_url_count += 1
            continue
        if right_url not in docs:
            print("Bad url: {}".format(right_url))
            bad_url_count += 1
            continue
        left_doc = docs[left_url]
        right_doc = docs[right_url]
        left_ts = left_doc["timestamp"]
        right_ts = right_doc["timestamp"]
        r["left_timestamp"] = left_ts
        r["right_timestamp"] = right_ts
        if r["mv_result_cause"] == "left_right" and left_ts > right_ts:
            print("Bad timestamps: {}, {}".format(left_url, right_url))
            bad_ts_count += 1
            continue
        if r["mv_result_cause"] == "right_left" and right_ts > left_ts:
            print("Bad timestamps: {}, {}".format(left_url, right_url))
            bad_ts_count += 1
            continue
        is_lenta = get_host(left_url) == get_host(right_url) == "https://lenta.ru/"
        r["id"] = language + "_" + ("tg_" if not is_lenta else "lenta_") + str(len(fixed_records))

        r["has_link"] = int(right_url in left_doc["links"] or left_url in right_doc["links"])
        r["full_agreement"] = r.pop("mv_part_result")
        r["full_result"] = r.pop("mv_result")
        r["simple_agreement"] = r.pop("mv_part_result_cause")
        r["simple_result"] = r.pop("mv_result_cause")
        mapping = {
            "left_right_cancel": "left_right_refute",
            "right_left_cancel": "right_left_refute"
        }
        r["full_result"] = mapping.get(r["full_result"], r["full_result"])
        fixed_records.append(r)

    print("Bad url count: {}".format(bad_url_count))
    print("Bad ts count: {}".format(bad_ts_count))
    with open(output, "w") as w:
        writer = csv.writer(w, delimiter="\t", quotechar='"')
        header = (
            "id", "left_title", "right_title",
            "left_url", "right_url",
            "left_timestamp", "right_timestamp",
            "full_result", "full_agreement",
            "simple_result", "simple_agreement",
            "has_link"
        )
        writer.writerow(header)
        for r in fixed_records:
            writer.writerow([r[key] for key in header])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--markup", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--docs", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
