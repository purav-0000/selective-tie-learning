#!/usr/bin/env python
"""Prepare immutable causal PKU sources shared by direct-emoji campaigns."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from stl.llm.provenance import digest_json  # noqa: E402
from stl.llm.factorized_hotel import write_jsonl  # noqa: E402
from stl.llm.provenance import atomic_json  # noqa: E402
from stl.llm.real_data.pku_saferlhf import (  # noqa: E402
    DATASET_ID, DATASET_REVISION, select_sources,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    train, test, audit = select_sources(
        ds["train"], ds["test"], strict_rows=7000,
        minimum_unsafe_severity=1, require_better_agreement=True,
        require_same_response_source=True, selection_strategy="uniform",
        selection_seed=0)
    if len(train) != 7000 or len(test) < 1:
        raise RuntimeError("unexpected PKU source cardinality")
    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "train_sources.jsonl", train)
    write_jsonl(args.output / "test_sources.jsonl", test)
    manifest = {
        "dataset_id": DATASET_ID, "dataset_revision": DATASET_REVISION,
        "strict_source_rows": 5000, "selection_probe_source_rows": 500,
        "reserve_source_rows": 500, "energy_source_rows": 1000,
        "test_source_rows": len(test),
        "partitions": {"strict": [0, 5000], "selection": [5000, 5500],
                       "reserve": [5500, 6000], "energy": [6000, 7000]},
        "train_digest": digest_json(train), "test_digest": digest_json(test),
        "subset_audit": audit,
    }
    atomic_json(args.output / "manifest.json", manifest)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
