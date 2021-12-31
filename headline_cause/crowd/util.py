import hashlib


def get_key(record):
    return tuple(sorted((record["left_url"], record["right_url"])))


def get_pool(pool_id, toloka_client):
    records = []
    for assignment in toloka_client.get_assignments(pool_id=pool_id):
        solutions = assignment.solutions
        if not solutions:
            continue
        for task, solution in zip(assignment.tasks, solutions):
            known_solutions = task.known_solutions
            if known_solutions is not None:
                continue
            input_values = task.input_values
            output_values = solution.output_values
            record = {
                "worker_id": assignment.user_id,
                "assignment_id": assignment.id
            }
            record.update(input_values)
            record.update(output_values)
            records.append(record)
    return records


def read_markup(markup_path):
    records = []
    with open(markup_path, "r") as r:
        header = next(r).strip().split("\t")
        header = [f.split(":")[-1] for f in header]
        for line in r:
            fields = line.strip().split("\t")
            record = dict(zip(header, fields))
            records.append(record)
    return records
