"""Exclude BABILong QA1 evaluation rows copied into a shared few-shot affix."""

import random

BABILONG_SOURCE = ("RMT-team/babilong-1k-samples", "0k", "qa1")


def same_split(pool_path, pool_name, pool_split, eval_source=BABILONG_SOURCE):
    pool = (str(pool_path).strip(), str(pool_name).strip(), str(pool_split).strip())
    return pool == tuple(eval_source)


def qa1_key(row):
    return (str(row["input"]).strip(), str(row["question"]).strip())


def sampled_indices(row_count, shots, seed):
    return random.Random(int(seed)).sample(list(range(row_count)), min(int(shots), row_count))


def exclusion_plan(rows, selected_indices):
    selected = [int(index) for index in selected_indices]
    selected_keys = {qa1_key(rows[index]) for index in selected}
    excluded = [index for index, row in enumerate(rows) if qa1_key(row) in selected_keys]
    return {"sampled_indices": selected, "excluded_indices": excluded,
            "evaluated_docs": len(rows) - len(excluded), "dataset_rows": len(rows)}


def verified_plan(rows, shots, seed, actual_sampled_indices):
    expected = sampled_indices(len(rows), shots, seed)
    actual = [int(index) for index in actual_sampled_indices]
    if actual != expected:
        raise ValueError(f"Few-shot sample changed: expected {expected}, got {actual}")
    return exclusion_plan(rows, actual)
