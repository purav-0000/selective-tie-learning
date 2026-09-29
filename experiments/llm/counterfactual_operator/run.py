#!/usr/bin/env python
"""Run a resumable hotel or emoji counterfactual campaign."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from stl.llm.counterfactual_campaign import score, train_endpoint, train_strict  # noqa: E402
from stl.llm.provenance import atomic_json, digest_json  # noqa: E402


def file_digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"cyclic base_config chain: {chain}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping: {path}")
    overlay = dict(raw)
    base_value = overlay.pop("base_config", None)
    if base_value is None:
        return overlay
    base = Path(str(base_value))
    if not base.is_absolute():
        local = path.parent / base
        base = local if local.is_file() else ROOT / base
    return _deep_merge(_load_config_tree(base, (*stack, path)), overlay)


def _validate_config(raw: dict) -> None:
    required_sections = {
        "campaign", "dataset", "data", "selection", "training", "model",
        "scoring", "evaluation", "output",
    }
    missing = required_sections - raw.keys()
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")
    if raw["dataset"] not in ("hotel_one_sided", "pku_direct_emoji"):
        raise ValueError("unsupported counterfactual campaign")
    selection = raw["selection"]
    methods = selection.get("methods")
    budgets = selection.get("budgets")
    seeds = selection.get("seeds")
    if methods != ["utl", "pm", "ha"]:
        raise ValueError("methods must be [utl, pm, ha]")
    if budgets != [100, 200, 300, 400, 500, 750, 1000, 1500, 2000]:
        raise ValueError("unexpected LLM budget grid")
    qwen_control = raw["model"]["id"] == "Qwen/Qwen3-4B-Instruct-2507"
    expected_seeds = [0] if qwen_control else [0, 1, 2]
    expected_utl_draws = 1 if qwen_control else 2
    if seeds != expected_seeds:
        raise ValueError(f"LLM seeds must be {expected_seeds}")
    if int(selection.get("utl_draws", 0)) != expected_utl_draws:
        raise ValueError(
            f"campaign requires {expected_utl_draws} UTL draw(s)")
    if selection.get("score_packet") != "exact_symmetric_average":
        raise ValueError("scoring must use the exact symmetric packet")
    target_size = selection.get("target_candidate_pool_size")
    if target_size is not None and (int(target_size) < max(budgets) or
                                    int(target_size) > int(raw["data"]["strict_rows"])):
        raise ValueError("invalid targeted-selector candidate panel")
    if selection.get("utl_candidate_pool") != "panel":
        raise ValueError("UTL and targeted selectors must use the same panel")
    if raw["training"]["packet"] != "single_prerealized_uniform":
        raise ValueError("endpoint training must use realized ties")
    if raw["training"].get("endpoint_initialization", "common_pretrained") not in (
            "common_pretrained", "strict_checkpoint"):
        raise ValueError("unsupported endpoint initialization")
    environments = raw["evaluation"].get("environments")
    if (not isinstance(environments, list) or
            not {"P", "Q_clean", "Q_adv"} <= set(environments)):
        raise ValueError("evaluation must include P, Q_clean, and Q_adv")
    if environments != ["P", "Q_clean", "Q_adv"]:
        raise ValueError("evaluation environments must be P, Q_clean, Q_adv")
    if raw["model"]["arithmetic"] != "unquantized_bf16":
        raise ValueError("active LLM campaigns require unquantized bf16")
    energy_anchors = int(raw["evaluation"].get("energy_anchors", 0))
    energy_redraws = int(raw["evaluation"].get("energy_redraws", 0))
    if energy_anchors < 100 or energy_redraws < 2:
        raise ValueError("endpoint energy requires >=100 anchors and >=2 redraws")
    if float(raw["training"].get("max_gradient_norm", 0.0)) <= 0:
        raise ValueError("max_gradient_norm must be positive")
    if raw["scoring"].get("selection_probe_size") != 500:
        raise ValueError("LLM scoring uses one fixed 500-row probe")
    if "probe_sizes" in raw["scoring"] or "criteria" in raw["scoring"]:
        raise ValueError("probe-size sweeps are not part of this campaign")
    relative_root = Path(str(raw["output"].get("relative_root", "")))
    if (relative_root.is_absolute() or ".." in relative_root.parts or
            not relative_root.parts or relative_root.parts[0] != "results"):
        raise ValueError("output.relative_root must remain under results/")


def load_config(path: Path) -> dict:
    raw = _load_config_tree(path)
    _validate_config(raw)
    return raw


def endpoint_phases(raw: dict) -> list[str]:
    phases = []
    for budget in raw["selection"]["budgets"]:
        for method in raw["selection"]["methods"]:
            draws = (range(raw["selection"]["utl_draws"])
                     if method == "utl" else range(1))
            phases.extend(f"{method}_d{draw}_k{budget}" for draw in draws)
    return phases


def phases(raw: dict, foundation_only: bool = False,
           endpoint_subset: set[str] | None = None) -> list[str]:
    """Return the ordered foundation and endpoint phases for one seed."""
    foundation = ["strict", "scores"]
    if foundation_only:
        return foundation
    endpoints = endpoint_phases(raw)
    if endpoint_subset is not None:
        endpoints = [phase for phase in endpoints if phase in endpoint_subset]
    return foundation + endpoints


def _phase_parts(phase: str) -> tuple[str, int, int]:
    method, draw, budget = phase.split("_")
    return method, int(draw.removeprefix("d")), int(budget.removeprefix("k"))


def _run_phase(raw: dict, seed: int, runtime_root: Path, seed_root: Path,
               completed: dict, phase: str, attempt: Path, update) -> None:
    if phase == "strict":
        train_strict(raw, seed, runtime_root, attempt)
    elif phase == "scores":
        strict = Path(completed["strict"]["path"])
        score(raw, seed, runtime_root, strict, attempt, update)
    else:
        method, draw, budget = _phase_parts(phase)
        strict = Path(completed["strict"]["path"])
        scores = Path(completed["scores"]["path"])
        train_endpoint(raw, seed, runtime_root, strict, scores, attempt,
                       method, draw, budget)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--seeds", help="comma-separated parent shard seeds")
    p.add_argument("--seed", type=int, help="one child phase seed")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument(
        "--foundation-only", action="store_true",
        help="run strict training and shared candidate scoring only")
    p.add_argument(
        "--endpoint-phase", action="append", default=[],
        metavar="METHOD_dDRAW_kBUDGET",
        help=("after preparing the shared foundation, run only this configured "
              "endpoint phase; repeat to select more than one endpoint"),
    )
    p.add_argument("--worker-phase")
    p.add_argument("--attempt", type=Path)
    p.add_argument("--seed-root", type=Path)
    args = p.parse_args()
    raw = load_config(args.config)
    all_endpoint_phases = set(endpoint_phases(raw))
    if args.endpoint_phase and args.foundation_only:
        raise ValueError(
            "endpoint-phase cannot be combined with foundation-only")
    if len(args.endpoint_phase) != len(set(args.endpoint_phase)):
        raise ValueError("duplicate endpoint-phase")
    unknown_endpoint_phases = sorted(set(args.endpoint_phase) - all_endpoint_phases)
    if unknown_endpoint_phases:
        raise ValueError(
            f"unknown endpoint phases: {unknown_endpoint_phases}")
    endpoint_subset = set(args.endpoint_phase) if args.endpoint_phase else None
    runtime_root = Path(os.environ.get(
        "STL_RUNTIME_ROOT", str(ROOT / "results/runtime")))
    runtime_root.mkdir(parents=True, exist_ok=True)
    code_files = [Path(__file__).resolve(), *sorted(
        (ROOT / "src/stl/llm").rglob("*.py"),
        key=lambda path: path.relative_to(ROOT).as_posix())]
    implementation = digest_json([(p.relative_to(ROOT).as_posix(), file_digest(p)) for p in code_files])

    if args.worker_phase:
        if args.seed is None or args.attempt is None or args.seed_root is None:
            raise ValueError("child phase requires seed, attempt, and seed-root")
        completed_path = args.seed_root / "completed.json"
        completed = json.loads(completed_path.read_text()) if completed_path.exists() else {}
        expected = {"campaign": raw["campaign"], "config": digest_json(raw),
                    "implementation": implementation, "seed": args.seed}
        args.attempt.mkdir(parents=True, exist_ok=False)
        _run_phase(raw, args.seed, runtime_root, args.seed_root, completed,
                   args.worker_phase, args.attempt, lambda name: print(f"[gate] {name}", flush=True))
        atomic_json(args.attempt / "success.json", expected)
        return 0

    if not args.seeds:
        raise ValueError("parent runner requires --seeds")
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != len(set(seeds)) or not set(seeds).issubset(raw["selection"]["seeds"]):
        raise ValueError("invalid shard seeds")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    status = ROOT / "results/llm/bundle_status" / (
        f"{raw['campaign']}_s{'-'.join(map(str,seeds))}.json")
    assigned = (0 if args.foundation_only
                else len(seeds) * len(
                    all_endpoint_phases if endpoint_subset is None
                    else endpoint_subset))
    state = {"campaign": raw["campaign"], "seeds": seeds, "state": "queued",
             "assigned_cells": assigned, "completed_cells": [], "current_cell": "waiting_for_gpu"}

    def update(current: str | None) -> None:
        state["current_cell"] = current
        atomic_json(status, state)
        print(f"[counterfactual] current={current}; cells={len(state['completed_cells'])}/{assigned}", flush=True)

    state["state"] = "running"
    update("starting_shard")
    for seed in seeds:
        seed_root = ROOT / raw["output"]["relative_root"] / f"seed{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        completed_path = seed_root / "completed.json"
        completed = json.loads(completed_path.read_text()) if completed_path.exists() else {}
        expected = {"campaign": raw["campaign"], "config": digest_json(raw),
                    "implementation": implementation, "seed": seed}
        contract = seed_root / "contract.json"
        if contract.exists() and json.loads(contract.read_text()) != expected:
            raise ValueError("stale seed resume contract")
        atomic_json(contract, expected)
        atomic_json(seed_root / "effective_config.json", raw)
        for phase in phases(raw, args.foundation_only, endpoint_subset):
            if phase in completed:
                marker = Path(completed[phase]["path"]) / "success.json"
                if not marker.is_file() or json.loads(marker.read_text()) != expected:
                    raise ValueError("invalid completed phase provenance")
                if phase in all_endpoint_phases:
                    state["completed_cells"].append(f"s{seed}:{phase}")
                continue
            attempts = seed_root / "attempts" / phase
            attempts.mkdir(parents=True, exist_ok=True)
            attempt_number = len(list(attempts.iterdir())) + 1
            attempt = attempts / f"attempt_{attempt_number:03d}"
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
                       "--gpu", str(args.gpu), "--seed", str(seed), "--worker-phase", phase,
                       "--attempt", str(attempt.resolve()), "--seed-root", str(seed_root.resolve())]

            if shutil.disk_usage(runtime_root).free < 4 * 1024**3:
                raise RuntimeError("less than 4 GiB free; preserving results and stopping")
            state["state"] = "running"; update(f"s{seed}:{phase}")
            completed_process = subprocess.run(command, cwd=ROOT)
            if completed_process.returncode:
                raise RuntimeError(
                    f"phase {phase} failed with exit code "
                    f"{completed_process.returncode}; rerun the same command "
                    "to resume from the last committed phase")
            marker = attempt / "success.json"
            if not marker.is_file():
                raise RuntimeError("phase child did not commit success")
            if json.loads(marker.read_text(encoding="utf-8")) != expected:
                raise RuntimeError("phase child committed mismatched provenance")
            completed[phase] = {"path": str(attempt.resolve()), **expected}
            atomic_json(completed_path, completed)
            if phase in all_endpoint_phases:
                state["completed_cells"].append(f"s{seed}:{phase}")
                update(f"s{seed}:{phase}:committed")
    state["state"] = "completed"; update(None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
