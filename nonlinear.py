#!/usr/bin/env python
"""Run, validate, and plot the paper's nonlinear reward-model experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from tqdm.auto import tqdm
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiments/nonlinear/config.yaml"
CONTINUED_CONFIG = ROOT / "experiments/nonlinear/config_continued.yaml"


def resolve_device(configured: str, override: str | None) -> str:
    """Resolve runtime placement without changing scientific configuration."""
    requested = str(override or configured)
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _stream(command: list[str], on_line=None) -> None:
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert process.stdout is not None
    tail: list[str] = []
    for line in process.stdout:
        line = line.rstrip()
        tail.append(line)
        tail = tail[-30:]
        if on_line is not None:
            on_line(line)
    return_code = process.wait()
    if return_code:
        detail = "\n".join(tail)
        raise RuntimeError(
            f"Command failed with exit code {return_code}:\n"
            f"{' '.join(command)}\n{detail}"
        )


def _endpoint_specs(config: dict, seeds: list[int], *, quick: bool = False) -> list[dict]:
    specs = []
    for seed in config["seeds"]:
        for operator in config["operators"]:
            for budget in config["budgets"]:
                if budget == config["n_candidates"]:
                    method = config["methods"][0]
                    specs.append({"seed": seed, "operator": operator,
                                  "budget": budget, "method": method, "draw": 0})
                    continue
                for method in config["methods"]:
                    draws = range(config["utl_draws"]) if method == "UTL" else [None]
                    for draw in draws:
                        specs.append({"seed": seed, "operator": operator,
                                      "budget": budget, "method": method,
                                      "draw": draw})
    selected = [dict(index=index, **spec) for index, spec in enumerate(specs)
                if int(spec["seed"]) in seeds]
    if not quick:
        return selected
    representative_budget = 500 if 500 in config["budgets"] else config["budgets"][0]
    return [spec for spec in selected if (
        (spec["budget"] == representative_budget
         and (spec["method"] != "UTL" or spec["draw"] == 0))
        or spec["budget"] == config["n_candidates"]
    )]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's nonlinear MLP campaign.")
    parser.add_argument("--config", type=Path,
                        help="YAML config (defaults to the paper config)")
    parser.add_argument("--continued", action="store_true",
                        help="run the strict-checkpoint appendix protocol")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="optional seed subset; all paper seeds run by default")
    parser.add_argument(
        "--quick", action="store_true",
        help=("run seed 0 with all selectors at k=500 and the shared full-pool "
              "endpoint; intended only as an end-to-end functional check"),
    )
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument(
        "--device",
        help=("runtime override: auto, cpu, cuda, or an explicit torch device; "
              "the config value is used when omitted"),
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help=("CPU math threads per process. The paper results used 4; "
              "change only for a non-bitwise functional run."),
    )
    parser.add_argument("--output-root", type=Path,
                        help="width-specific result directory")
    parser.add_argument("--foundation-root", type=Path,
                        help="width-specific fresh-run foundation directory")
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be positive")
    # These variables are consumed when the stage subprocesses import NumPy,
    # BLAS, and PyTorch. Pinning them here removes an otherwise hidden source
    # of CPU reduction-order differences.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)

    if args.config and args.continued:
        parser.error("choose either --config or --continued")
    config_path = (args.config or
                   (CONTINUED_CONFIG if args.continued else DEFAULT_CONFIG))
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.width not in map(int, config["widths"]):
        parser.error(f"width {args.width} is not present in {config['widths']}")
    seeds = (args.seeds or
             ([int(config["seeds"][0])] if args.quick else
              [int(seed) for seed in config["seeds"]]))
    unknown = sorted(set(seeds) - set(map(int, config["seeds"])))
    if unknown:
        parser.error(f"seeds not present in the config: {unknown}")
    if args.quick and len(seeds) != 1:
        parser.error("--quick accepts exactly one seed")
    device = resolve_device(str(config.get("device", "cpu")), args.device)
    default_output = (ROOT / "results/nonlinear/quick_check" / f"w{args.width}"
                      if args.quick else
                      ROOT / config["output_root"] / f"w{args.width}")
    output = args.output_root.resolve() if args.output_root else default_output
    foundation_root = args.foundation_root.resolve() if args.foundation_root else output
    if args.continued and args.foundation_root is None:
        foundation_root = ROOT / "results/nonlinear/paper_aligned" / f"w{args.width}"
    specs = _endpoint_specs(config, seeds, quick=args.quick)

    total = len(seeds) + len(specs) + 2
    with tqdm(total=total, desc="Nonlinear", unit="stage") as bar:
        if not args.continued or foundation_root == output:
            prepare = [
                sys.executable, str(ROOT / "experiments/nonlinear/prepare.py"),
                "--config", str(config_path), "--device", device,
                "--output-root", str(foundation_root), "--cells",
                *(f"{args.width}:{seed}" for seed in seeds),
            ]
            foundation_done = 0
            def foundation_progress(line: str) -> None:
                nonlocal foundation_done
                if line.startswith("[cell] completed "):
                    foundation_done += 1
                    bar.update(1)
                    bar.set_postfix_str(line.removeprefix("[cell] completed "))
            _stream(prepare, foundation_progress)
            if foundation_done < len(seeds):
                bar.update(len(seeds) - foundation_done)
        else:
            missing = [seed for seed in seeds if not (
                foundation_root / "foundations" / f"seed_{seed:02d}.pt").is_file()]
            if missing:
                raise FileNotFoundError(
                    f"continued training needs fresh foundations for seeds {missing}: "
                    f"run python nonlinear.py first")
            bar.update(len(seeds))
            bar.set_postfix_str("reused fresh foundations")

        grid = [
            sys.executable, str(ROOT / "experiments/nonlinear/grid.py"),
            "--config", str(config_path), "--input-root", str(foundation_root),
            "--output-root", str(output), "--width", str(args.width),
            "--device", device, "--task-indices",
            *(str(spec["index"]) for spec in specs),
        ]
        endpoint_done: set[int] = set()
        def endpoint_progress(line: str) -> None:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                return
            index = value.get("completed", value.get("reused"))
            if isinstance(index, int) and index not in endpoint_done:
                endpoint_done.add(index)
                bar.update(1)
                bar.set_postfix_str(value.get("identifier", f"cell {index}"))
        _stream(grid, endpoint_progress)
        if len(endpoint_done) < len(specs):
            bar.update(len(specs) - len(endpoint_done))

        validation = [
            sys.executable, str(ROOT / "experiments/nonlinear/validate.py"),
            "--config", str(config_path), "--root", str(output),
            "--widths", str(args.width), "--seeds", *map(str, seeds),
        ]
        if not args.quick:
            validation.append("--require-complete")
        if foundation_root != output:
            validation += ["--foundation-root", str(foundation_root)]
        _stream(validation)
        bar.update(1)
        bar.set_postfix_str("validation passed")

        assembly = [
            sys.executable, str(ROOT / "experiments/nonlinear/assemble.py"),
            "--config", str(config_path), "--width", str(args.width),
            "--root", str(output), "--seeds", *map(str, seeds),
        ]
        if foundation_root != output:
            assembly += ["--foundation-root", str(foundation_root)]
        _stream(assembly)
        bar.update(1)
        bar.set_postfix_str("figures assembled")

    print(f"Results: {output}")
    print(f"Figures: {output / 'aggregate/figures'}")


if __name__ == "__main__":
    main()
