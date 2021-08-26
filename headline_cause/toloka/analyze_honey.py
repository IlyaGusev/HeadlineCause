import argparse
import os
from collections import Counter

import toloka.client as toloka


def main(
    token,
    pools_file,
    key_field,
    res_field,
):
    with open(os.path.expanduser(token), "r") as r:
        toloka_token = r.read().strip()
    toloka_client = toloka.TolokaClient(toloka_token, 'PRODUCTION')

    pool_ids = []
    with open(pools_file, "r") as r:
        for line in r:
            pool_id = line.strip()
            if not pool_id:
                continue
            pool_id = int(pool_id)
            pool_ids.append(pool_id)

    honey_correct_count = Counter()
    honey_all_count = Counter()
    for pool_id in pool_ids:
        for assignment in toloka_client.get_assignments(pool_id=pool_id):
            solutions = assignment.solutions
            if not solutions:
                continue
            for task, solution in zip(assignment.tasks, solutions):
                known_solutions = task.known_solutions
                if known_solutions is None:
                    continue
                input_values = task.input_values
                output_values = solution.output_values
                true_result = known_solutions[0].output_values[res_field]
                pred_result = output_values[res_field]
                honey_id = input_values["id"]
                honey_all_count[honey_id] += 1
                if true_result == pred_result:
                    honey_correct_count[honey_id] += 1
    for honey_id, all_count in sorted(honey_all_count.items()):
        correct_count = honey_correct_count[honey_id]
        print(honey_id, correct_count / all_count * 100.0, correct_count, all_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-field", type=str, default="id")
    parser.add_argument("--res-field", type=str, default="result")
    parser.add_argument("--token", type=str, default="~/.toloka/token")
    parser.add_argument("--pools-file", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))
