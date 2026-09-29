#!/usr/bin/env python
"""Run independently resumable endpoint indices from staged score bundles."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))

from stl.nonlinear.campaign import (  # noqa: E402
    atomic_json, atomic_torch_save, config_from_dict, endpoint_identifier,
    run_endpoint, validate_reloaded_endpoint,
)


ENDPOINT_INITIALIZATIONS = ("common_initialization", "strict_checkpoint")


def endpoint_foundation(foundation: dict, initialization: str) -> dict:
    """Select the common or strict initialization stored in a foundation."""
    if initialization not in ENDPOINT_INITIALIZATIONS:
        raise ValueError(f"unknown endpoint initialization: {initialization}")
    if initialization == "common_initialization":
        return foundation
    continued = dict(foundation)
    continued["initial_state"] = foundation["strict_state"]
    continued["initial_digest"] = foundation["strict_digest"]
    return continued


def load_config(path: Path, width: int, device: str):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.pop("name", None)
    raw.pop("endpoint_initialization", None)
    raw.pop("output_root", None)
    widths = tuple(raw.pop("widths"))
    if width not in widths:
        raise ValueError(f"width {width} is not present in the configuration")
    raw.update({"width": width, "device": device})
    return config_from_dict(raw)


def enumerate_specs(cfg) -> list[dict]:
    output = []
    for seed in cfg.seeds:
        for operator in cfg.operators:
            for budget in cfg.budgets:
                if budget == cfg.n_candidates:
                    method = cfg.methods[0]
                    output.append({
                        "seed": seed, "operator": operator,
                        "budget": budget, "method": method,
                        "draw": 0 if method == "UTL" else None})
                    continue
                for method in cfg.methods:
                    draws = range(cfg.utl_draws) if method == "UTL" else [None]
                    for draw in draws:
                        output.append({
                            "seed": seed, "operator": operator,
                            "budget": budget, "method": method, "draw": draw})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--task-indices", type=int, nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--endpoint-initialization", choices=ENDPOINT_INITIALIZATIONS,
        help="optional override of the protocol in the configuration")
    args = parser.parse_args()
    experiment = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg = load_config(args.config, args.width, args.device)
    initialization = (args.endpoint_initialization
                      or experiment.get("endpoint_initialization",
                                        "common_initialization"))
    root = args.output_root or (
        ROOT/experiment.get("output_root", "results/nonlinear/paper_aligned")
        /f"w{cfg.width}")
    specs = enumerate_specs(cfg)
    cache = {}
    for index in args.task_indices:
        spec = specs[index]
        seed = spec["seed"]
        if seed not in cache:
            foundation = torch.load(
                args.input_root/"foundations"/f"seed_{seed:02d}.pt",
                map_location="cpu", weights_only=False)
            scores = torch.load(
                args.input_root/"scores"/f"seed_{seed:02d}.pt",
                map_location="cpu", weights_only=False)
            cache[seed] = foundation, scores
        foundation, scores = cache[seed]
        endpoint_source = endpoint_foundation(
            foundation, initialization)
        identifier = endpoint_identifier(
            spec["operator"], spec["method"], spec["budget"], spec["draw"])
        destination = (root/"endpoints"/spec["operator"]
                       /f"seed_{seed:02d}"/f"{identifier}.pt")
        if destination.exists():
            print(json.dumps({"reused": index, "identifier": identifier}),
                  flush=True)
            continue
        endpoint_spec = {name: item for name, item in spec.items() if name != "seed"}
        value = run_endpoint(cfg, endpoint_source, scores, **endpoint_spec)
        value["endpoint_initialization"] = initialization
        value["foundation_initial_digest"] = foundation["initial_digest"]
        value["endpoint_start_digest"] = endpoint_source["initial_digest"]
        value["full_pool_shared"] = spec["budget"] == cfg.n_candidates
        value["task_index"] = index
        value["reload_validation"] = validate_reloaded_endpoint(cfg, foundation, value)
        atomic_torch_save(value, destination)
        compact = {key: value[key] for key in (
            "task_index", "identifier", "seed", "width", "operator",
            "method", "draw", "budget", "state_digest", "selected_hash",
            "runtime_seconds", "peak_vram_bytes", "metrics", "fitting",
            "reload_validation", "endpoint_initialization",
            "foundation_initial_digest", "endpoint_start_digest")}
        atomic_json(compact, destination.with_suffix(".json"))
        print(json.dumps({"completed": index, "identifier": identifier,
                          "runtime_seconds": value["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
