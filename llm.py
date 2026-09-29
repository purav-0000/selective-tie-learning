#!/usr/bin/env python
"""Run and aggregate the paper's hotel and emoji LLM experiments."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "experiments/llm/counterfactual_operator"
CONFIGS = {
    ("llama1b", "hotels", "fresh"): EXPERIMENT / "configs/hotels.yaml",
    ("llama1b", "emoji", "fresh"): EXPERIMENT / "configs/emoji.yaml",
    ("llama1b", "hotels", "continued"): EXPERIMENT / "configs/hotels_continued.yaml",
    ("llama1b", "emoji", "continued"): EXPERIMENT / "configs/emoji_continued.yaml",
    ("qwen3-4b", "hotels", "fresh"): EXPERIMENT / "configs/hotels_qwen3_4b.yaml",
    ("qwen3-4b", "emoji", "fresh"): EXPERIMENT / "configs/emoji_qwen3_4b.yaml",
}
def _run(command: list[str], environment: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=ROOT, env=environment)
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            + " ".join(command))


def _prepare_pku(runtime_root: Path, environment: dict[str, str]) -> None:
    destination = runtime_root / "inputs/pku-safe-causal-sources-v1"
    manifest = destination / "manifest.json"
    if manifest.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        str(EXPERIMENT / "prepare_pku_sources.py"),
        "--output", str(destination),
    ], environment)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Reproduce the paper's LLM campaign on one CUDA GPU. "
                     "The default runs both fresh protocols and aggregates them."))
    parser.add_argument("--gpu", type=int, default=0,
                        help="physical CUDA device (default: 0)")
    parser.add_argument("--model", choices=("llama1b", "qwen3-4b"),
                        default="llama1b",
                        help="model campaign (default: llama1b)")
    parser.add_argument("--datasets", nargs="+", choices=("hotels", "emoji"),
                        default=["hotels", "emoji"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="seed subset (default: every configured seed)")
    parser.add_argument("--continued", action="store_true",
                        help="run the strict-checkpoint appendix protocol")
    parser.add_argument("--foundation-only", action="store_true",
                        help="stop after strict training and shared scoring")
    parser.add_argument("--endpoint-phase", action="append", default=[],
                        metavar="METHOD_dDRAW_kBUDGET",
                        help="run only a configured endpoint after the foundation")
    parser.add_argument("--runtime-root", type=Path,
                        default=ROOT / "results/runtime",
                        help="model/data cache and immutable input directory")
    args = parser.parse_args()
    registered_seeds = [0] if args.model == "qwen3-4b" else [0, 1, 2]
    seeds = args.seeds if args.seeds is not None else registered_seeds

    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("datasets must be unique")
    if len(seeds) != len(set(seeds)) or not set(seeds) <= set(registered_seeds):
        parser.error(f"seeds must be unique members of {set(registered_seeds)}")
    if args.model == "qwen3-4b" and args.continued:
        parser.error("the submitted Qwen control is fresh training with seed 0")
    if args.foundation_only and args.endpoint_phase:
        parser.error("--foundation-only and --endpoint-phase are mutually exclusive")

    runtime_root = args.runtime_root.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["STL_RUNTIME_ROOT"] = str(runtime_root)
    if "emoji" in args.datasets:
        _prepare_pku(runtime_root, environment)

    mode = "continued" if args.continued else "fresh"
    with tqdm(total=len(args.datasets), desc="LLM campaigns", unit="dataset") as bar:
        for dataset in args.datasets:
            config = CONFIGS[(args.model, dataset, mode)]
            command = [
                sys.executable, str(EXPERIMENT / "run.py"),
                "--config", str(config),
                "--seeds", ",".join(map(str, seeds)),
                "--gpu", str(args.gpu),
            ]
            if args.foundation_only:
                command.append("--foundation-only")
            else:
                for phase in args.endpoint_phase:
                    command.extend(("--endpoint-phase", phase))
            _run(command, environment)
            bar.update(1)
            bar.set_postfix_str(dataset)

    complete = (
        set(args.datasets) == {"hotels", "emoji"}
        and set(seeds) == set(registered_seeds)
        and not args.foundation_only and not args.endpoint_phase
    )
    if complete:
        aggregate_mode = "qwen_fresh" if args.model == "qwen3-4b" else mode
        output = (ROOT / "results/llm/counterfactual_operator" /
                  f"paper_{args.model}_{mode}/aggregate")
        _run([
            sys.executable, str(EXPERIMENT / "aggregate.py"),
            "--local-root", str(ROOT / "results/llm/counterfactual_operator"),
            "--training-mode", aggregate_mode,
            "--output", str(output),
        ], environment)
        print(f"Aggregate tables and figures: {output}")
    else:
        print("Subset/check complete; aggregation is reserved for the full "
              "two-dataset, three-seed campaign.")


if __name__ == "__main__":
    main()
