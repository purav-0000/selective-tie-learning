#!/usr/bin/env python
"""Run, validate, and plot the paper's log-linear experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from tqdm.auto import tqdm
import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "experiments/linear/config.yaml"
CONTINUED_CONFIG = ROOT / "experiments/linear/config_continued.yaml"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's complete log-linear campaign.")
    parser.add_argument("--config", type=Path,
                        help="YAML config (defaults to the paper config)")
    parser.add_argument("--continued", action="store_true",
                        help="run the strict-checkpoint appendix protocol")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="optional seed subset for a shorter exact run")
    parser.add_argument("--workers", type=int,
                        default=min(5, os.cpu_count() or 1))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.config and args.continued:
        parser.error("choose either --config or --continued")
    config_path = (args.config or
                   (CONTINUED_CONFIG if args.continued else DEFAULT_CONFIG))
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    unknown = sorted(set(seeds) - set(map(int, config["seeds"])))
    if unknown:
        parser.error(f"seeds not present in the config: {unknown}")
    output = (args.output_root.resolve() if args.output_root else
              ROOT / config["output_root"])

    run_command = [
        sys.executable, str(ROOT / "experiments/linear/run.py"),
        "--config", str(config_path), "--output-root", str(output),
        "--workers", str(max(1, args.workers)), "--seeds",
        *map(str, seeds),
    ]
    if args.overwrite:
        run_command.append("--overwrite")

    completed_seeds: set[int] = set()
    with tqdm(total=len(seeds) + 2, desc="Log-linear", unit="stage") as bar:
        def progress(line: str) -> None:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                return
            seed = value.get("seed")
            if seed in seeds and "endpoints" in value and seed not in completed_seeds:
                completed_seeds.add(seed)
                bar.update(1)
                bar.set_postfix_str(f"seed {seed} complete")

        _stream(run_command, progress)
        if len(completed_seeds) < len(seeds):
            bar.update(len(seeds) - len(completed_seeds))

        validation = [
            sys.executable, str(ROOT / "experiments/linear/validate.py"),
            "--config", str(config_path), "--root", str(output),
            "--seeds", *map(str, seeds), "--require-complete",
        ]
        _stream(validation)
        bar.update(1)
        bar.set_postfix_str("validation passed")

        assembly = [
            sys.executable, str(ROOT / "experiments/linear/assemble.py"),
            "--config", str(config_path), "--root", str(output),
            "--seeds", *map(str, seeds),
        ]
        _stream(assembly)
        bar.update(1)
        bar.set_postfix_str("figures assembled")

    print(f"Results: {output}")
    print(f"Figures: {output / 'aggregate/figures'}")


if __name__ == "__main__":
    main()
