"""Seed-specific BABILong QA1 evaluation filtering for the lm-eval task."""

import json
import os

from babilong_exclusion import exclusion_plan, sampled_indices


def process_docs(dataset):
    encoded = os.environ.get("ACACHE_BABILONG_EXCLUSION")
    if not encoded:
        return dataset
    settings = json.loads(encoded)
    selected = sampled_indices(len(dataset), settings["shots"], settings["seed"])
    plan = exclusion_plan(dataset, selected)
    if not set(selected) <= set(plan["excluded_indices"]):
        raise RuntimeError("BABILong sampled docs were not excluded")
    excluded = set(plan["excluded_indices"])
    kept = [index for index in range(len(dataset)) if index not in excluded]
    docs = dataset.add_column("_babilong_source_index", list(range(len(dataset)))).select(kept)
    print(f"BABILONG_EXCLUSION {json.dumps(plan, sort_keys=True)}", flush=True)
    return docs
