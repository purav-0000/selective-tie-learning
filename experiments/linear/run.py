#!/usr/bin/env python
"""Run winner-preserving log-linear Bradley--Terry tie selection."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))

from stl.linear.campaign import (  # noqa: E402
    METHODS, OPERATORS, compute_scores, endpoint_metrics,
    make_linear_bt_dgp, solve_dgp, solve_single_packet)
from stl.linear.population_metrics import trained_component_metrics  # noqa: E402


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _implementation_digest() -> str:
    digest = hashlib.sha256()
    for path in (
        ROOT/"src/stl/linear/objective.py",
        ROOT/"src/stl/linear/optimization.py",
        ROOT/"src/stl/linear/campaign.py",
        ROOT/"src/stl/linear/population_metrics.py",
        ROOT/"src/stl/nonlinear/dgp.py",
        Path(__file__).resolve(),
    ):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent/f".{path.name}.{os.getpid()}.tmp"
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(_jsonable(value), handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    os.replace(temporary, path)


def _load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _complete(path: Path, config_digest: str, implementation_digest: str,
              identity: dict) -> bool:
    if not path.is_file():
        return False
    try:
        value = _load(path)
    except (OSError, EOFError, json.JSONDecodeError):
        return False
    return (
        value.get("state") == "completed"
        and value.get("config_digest") == config_digest
        and value.get("implementation_digest") == implementation_digest
        and all(value.get(key) == expected for key, expected in identity.items())
        and value.get("metrics", {}).get("finite", True)
    )


def _validate(config: dict) -> None:
    if tuple(config["methods"]) != METHODS or tuple(config["operators"]) != OPERATORS:
        raise ValueError("method/operator grid is immutable")
    if not 1 <= int(config["n_candidates"]) <= int(config["n_strict"]):
        raise ValueError("invalid candidate source subsample")
    if not np.isclose(float(config["theory_packet_weight"]),
                      1.0 / int(config["n_strict"]), rtol=0.0, atol=1e-15):
        raise ValueError("theory packet must have exactly 1/N weight")
    if config["candidate_bank"] != "both_options_from_all_strict_pairs":
        raise ValueError("candidate donor bank must contain all strict options")
    if config["reference_coordinates"] != "theta_minus_theta_ref_with_zero_origin":
        raise ValueError("log-linear theta must be reference-adjusted")
    if config["training"] not in (
            "fresh_zero_initialized_dense_newton_realized_ties",
            "continued_strict_initialized_dense_newton_realized_ties"):
        raise ValueError("unknown endpoint initialization/training protocol")
    if config["candidate_operator"] != "strict_winner_vs_empirical_nuisance_redraw":
        raise ValueError("candidate must retain the displayed winner")
    if config["training_packet"] != "single_prerealized_uniform":
        raise ValueError("endpoint training must use fixed realized tie labels")
    if config["scoring_packet"] != "exact_symmetric_average":
        raise ValueError("selection must score exact symmetric packets")
    if config["packet_source"] != "grad_logical_symmetric_packet_over_N":
        raise ValueError("packet source must match one-shot influence theory")
    if config["arithmetic"] != "float64":
        raise ValueError("log-linear campaign requires float64")


def _make_dgp(config: dict, seed: int):
    keys = (
        "d_c", "d_s", "block_condition_number", "r_min", "r_max",
        "gamma_c", "gamma_s", "beta_teacher", "beta_dpo", "n_strict",
        "n_candidates")
    return make_linear_bt_dgp(seed=seed, **{key: config[key] for key in keys})


def _solve(config: dict, dgp, operator: str, selected: np.ndarray,
           *, starting_theta: np.ndarray | None = None):
    solve = solve_dgp(
        dgp, operator, selected,
        starting_theta=starting_theta,
        tolerance=float(config["gradient_tolerance"]),
        maximum_iterations=int(config["maximum_newton_iterations"]))
    if not solve.converged:
        raise RuntimeError(f"full-DPO solve failed: {solve.gradient_norm:.3e}")
    if solve.hessian_eigenvalues[0] <= 0.0:
        raise RuntimeError("endpoint Hessian is not positive definite")
    return solve


def _endpoint(
    *, config_digest: str, implementation_digest: str, seed: int,
    operator: str, method: str, draw: int | None, budget: int,
    selected: np.ndarray, solve, dgp, strict_theta: np.ndarray, runtime: float,
    initialization: str,
) -> dict:
    return {
        "state": "completed", "config_digest": config_digest,
        "implementation_digest": implementation_digest,
        "seed": seed, "operator": operator, "method": method,
        "draw": draw, "budget": budget,
        "initialization": initialization,
        "initialization_digest": _array_digest(
            np.zeros_like(strict_theta)
            if initialization == "theta_zero" else strict_theta),
        "training": "exact_full_batch_dpo_realized_ties",
        "logical_packet_weight": f"1/({len(dgp.strict.c_w)}+{budget})",
        "training_packet": "single_prerealized_uniform",
        "scoring_packet": "exact_symmetric_average",
        "selected_indices": selected,
        "selection_digest": _array_digest(np.sort(selected)),
        "theta": solve.theta,
        "runtime_seconds": runtime,
        "iterations": solve.iterations,
        "backtracks": solve.backtracks,
        "metrics": endpoint_metrics(dgp, solve, strict_theta),
    }


def run_seed(config: dict, output: str, seed: int,
             overwrite: bool) -> dict:
    started = perf_counter()
    config_digest = _digest(config)
    implementation_digest = _implementation_digest()
    root = Path(output)/f"seed_{seed:02d}"
    dgp = _make_dgp(config, seed)
    strict = _solve(config, dgp, "resampling", np.empty(0, dtype=np.int64))
    scores = compute_scores(dgp, strict)
    maximum_residual = max(
        diagnostic["oracle_residual"]
        for diagnostic in scores.diagnostics["operators"].values()
    )
    if maximum_residual > 1e-10:
        raise RuntimeError(f"oracle solve residual is too large: {maximum_residual:.3e}")

    pool = dgp.pools["resampling"]
    if len(np.unique(pool.source_indices)) != len(pool.rows):
        raise RuntimeError("candidate sources are not distinct")
    if not np.array_equal(pool.anchor_c, dgp.strict.c_w[pool.source_indices]):
        raise RuntimeError("winner causal content changed")
    expected_bank = dgp.strict.nuisance_bank
    if not np.array_equal(pool.nuisance_bank, expected_bank):
        raise RuntimeError("donor bank does not contain all strict options")
    if not 0.4 <= pool.a_wins.mean() <= 0.6:
        raise RuntimeError("realized orientation is unexpectedly imbalanced")
    strict_metrics = endpoint_metrics(dgp, strict, strict.theta)
    strict_energy = strict_metrics["energy"]["spurious_energy"]
    endpoint_initialization = (
        "theta_zero"
        if config["training"] == "fresh_zero_initialized_dense_newton_realized_ties"
        else "strict_checkpoint")

    foundation = {
        "state": "completed", "config_digest": config_digest,
        "implementation_digest": implementation_digest, "seed": seed,
        "strict_theta": strict.theta,
        "strict_metrics": strict_metrics,
        "strict_gradient_norm": strict.gradient_norm,
        "strict_hessian_eigenvalues": strict.hessian_eigenvalues,
        "dgp_diagnostics": dgp.diagnostics,
        "strict_digest": _array_digest(dgp.strict.strict_deltas),
        "nuisance_bank_digest": _array_digest(pool.nuisance_bank),
        "candidate_source_indices": pool.source_indices,
        "candidate_donor_indices": pool.donor_index,
        "candidate_orientation_a_wins": pool.a_wins,
        "candidate_digests": {
            operator: _array_digest(dgp.pools[operator].rows)
            for operator in OPERATORS},
        "orientation_digest": _array_digest(dgp.pools["resampling"].a_wins),
        "candidate_anchor": "preferred_causal_latent",
        "resampling_bank": "both_options_from_all_strict_pairs",
        "reference_coordinates": "theta_minus_theta_ref_with_zero_origin",
        "theory_packet_weight": float(config["theory_packet_weight"]),
        "candidate_operator": "strict_winner_vs_empirical_nuisance_redraw",
        "training_packet": "single_prerealized_uniform",
        "scoring_packet": "exact_symmetric_average",
        "population_statistics_used_for_selection": True,
        "endpoint_initialization": endpoint_initialization,
    }
    _atomic_json(root/"foundation.json.gz", foundation)
    score_payload = {
        "state": "completed", "config_digest": config_digest,
        "implementation_digest": implementation_digest, "seed": seed,
        "scores": scores.scores,
        "rankings": scores.rankings, "diagnostics": scores.diagnostics,
    }
    _atomic_json(root/"scores.json.gz", score_payload)

    theory_count = len(pool.rows)
    for index in range(theory_count):
        margin = float(scores.scores["resampling"]["PM"][index])
        oracle = float(scores.scores["resampling"]["Oracle"][index])
        conditional = float(scores.diagnostics["operators"]["resampling"][
            "predicted_conditional_influence"][index])
        for packet in ("symmetric", "realized"):
            path = root/"theory"/packet/f"j{index:04d}.json.gz"
            identity = {"seed": seed, "operator": "resampling",
                        "candidate_index": index, "packet": packet}
            if not overwrite and _complete(
                    path, config_digest, implementation_digest, identity):
                continue
            packet_started = perf_counter()
            solve = solve_single_packet(
                dgp, "resampling", index, strict,
                symmetric=(packet == "symmetric"),
                tolerance=float(config["gradient_tolerance"]),
                maximum_iterations=int(config["maximum_newton_iterations"]))
            if not solve.converged or solve.hessian_eigenvalues[0] <= 0:
                raise RuntimeError(f"single-packet {packet} retrain failed: {index}")
            energy = trained_component_metrics(
                solve.theta, strict.theta, dgp)["E_S"]
            _atomic_json(path, {
                "state": "completed", "config_digest": config_digest,
                "implementation_digest": implementation_digest,
                "seed": seed, "operator": "resampling",
                "candidate_index": index, "packet": packet,
                "source_index": int(pool.source_indices[index]),
                "donor_index": int(pool.donor_index[index]),
                "zero_contrast": bool(np.all(pool.rows[index] == 0.0)),
                "a_wins": bool(pool.a_wins[index]),
                "packet_weight_relative_to_strict_mean": 1.0/len(dgp.strict.c_w),
                "absolute_margin": margin,
                "oracle_predicted_reduction": oracle,
                "conditional_predicted_reduction": conditional,
                "observed_spurious_energy_reduction": strict_energy-energy,
                "spurious_energy": energy,
                "gradient_norm": solve.gradient_norm,
                "hessian_minimum_eigenvalue": float(solve.hessian_eigenvalues[0]),
                "runtime_seconds": perf_counter()-packet_started,
                "metrics": {"finite": bool(np.isfinite(energy))},
            })
        if (index+1) % 25 == 0 or index+1 == theory_count:
            print(json.dumps({"seed": seed, "theory_candidates": index+1,
                              "theory_total": theory_count}), flush=True)

    budgets = list(map(int, config["budgets"]))
    draws = int(config["utl_draws"])
    completed = 0
    full_solve = None
    def selected_solve(selected: np.ndarray):
        nonlocal full_solve
        starting_theta = (
            None if endpoint_initialization == "theta_zero" else strict.theta)
        if len(selected) == len(pool.rows):
            if full_solve is None:
                full_solve = _solve(
                    config, dgp, "resampling", selected,
                    starting_theta=starting_theta)
            return full_solve
        return _solve(
            config, dgp, "resampling", selected,
            starting_theta=starting_theta)
    for operator in OPERATORS:
        for method in ("PM", "Oracle"):
            ranking = scores.rankings[operator][method]
            for budget in budgets:
                selected = ranking[:budget]
                path = root/operator/method/f"k{budget:05d}.json.gz"
                identity = {"seed": seed, "operator": operator,
                            "method": method, "draw": None, "budget": budget}
                if overwrite or not _complete(
                        path, config_digest, implementation_digest, identity):
                    endpoint_started = perf_counter()
                    solve = selected_solve(selected)
                    _atomic_json(path, _endpoint(
                        config_digest=config_digest,
                        implementation_digest=implementation_digest,
                        seed=seed, operator=operator, method=method, draw=None,
                        budget=budget, selected=selected, solve=solve, dgp=dgp,
                        strict_theta=strict.theta,
                        initialization=endpoint_initialization,
                        runtime=perf_counter()-endpoint_started))
                completed += 1
        for draw in range(draws):
            ranking = np.random.default_rng(
                np.random.SeedSequence([seed, 91_007, draw])).permutation(
                    len(dgp.pools[operator].rows))
            for budget in budgets:
                selected = ranking[:budget]
                path = root/operator/"UTL"/f"draw{draw}_k{budget:05d}.json.gz"
                identity = {"seed": seed, "operator": operator,
                            "method": "UTL", "draw": draw, "budget": budget}
                if overwrite or not _complete(
                        path, config_digest, implementation_digest, identity):
                    endpoint_started = perf_counter()
                    solve = selected_solve(selected)
                    _atomic_json(path, _endpoint(
                        config_digest=config_digest,
                        implementation_digest=implementation_digest,
                        seed=seed, operator=operator, method="UTL", draw=draw,
                        budget=budget, selected=selected, solve=solve, dgp=dgp,
                        strict_theta=strict.theta,
                        initialization=endpoint_initialization,
                        runtime=perf_counter()-endpoint_started))
                completed += 1
    return {
        "seed": seed, "endpoints": completed,
        "strict_gradient_norm": strict.gradient_norm,
        "strict_accuracy_P": foundation["strict_metrics"]["accuracy"]["P"],
        "strict_accuracy_Q_mixed": foundation["strict_metrics"]["accuracy"]["Q_mixed"],
        "oracle_maximum_residual": maximum_residual,
        "theory_candidates": theory_count,
        "runtime_seconds": perf_counter()-started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate(config)
    output = args.output_root or ROOT/config["output_root"]
    seeds = args.seeds if args.seeds else list(map(int, config["seeds"]))
    summaries = []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds))) as executor:
        futures = {executor.submit(
            run_seed, config, str(output), seed, args.overwrite): seed
            for seed in seeds}
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    print(json.dumps({
        "state": "completed", "output": str(output),
        "summaries": sorted(summaries, key=lambda row: row["seed"]),
    }, indent=2))


if __name__ == "__main__":
    main()
