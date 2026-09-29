#!/usr/bin/env python
"""Validate provenance, convergence, and completeness of the linear theory run."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import yaml


def _load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.root or Path(config["output_root"])
    seeds = args.seeds if args.seeds else config["seeds"]
    failures: list[str] = []
    provenance: set[tuple[str, str]] = set()
    counts = {"seeds": 0, "theory": 0, "endpoints": 0}
    expected_endpoints = len(config["budgets"])*(len(config["methods"])-1+config["utl_draws"])
    expected_initialization = (
        "theta_zero"
        if config["training"] == "fresh_zero_initialized_dense_newton_realized_ties"
        else "strict_checkpoint")
    for seed in seeds:
        base = root/f"seed_{seed:02d}"
        try:
            foundation = _load(base/"foundation.json.gz")
            scores = _load(base/"scores.json.gz")
        except (OSError, EOFError, ValueError) as error:
            failures.append(f"seed {seed}: foundation/scores missing or invalid: {error}")
            continue
        counts["seeds"] += 1
        pair = (foundation["config_digest"], foundation["implementation_digest"])
        provenance.add(pair)
        if (scores["config_digest"], scores["implementation_digest"]) != pair:
            failures.append(f"seed {seed}: score provenance mismatch")
        if foundation["strict_gradient_norm"] > config["gradient_tolerance"]:
            failures.append(f"seed {seed}: strict nonstationarity")
        if min(foundation["strict_hessian_eigenvalues"]) <= 0:
            failures.append(f"seed {seed}: strict Hessian not positive definite")
        source = np.asarray(foundation["candidate_source_indices"])
        donor = np.asarray(foundation["candidate_donor_indices"])
        orientation = np.asarray(foundation["candidate_orientation_a_wins"])
        if len(source) != config["n_candidates"] or len(np.unique(source)) != len(source):
            failures.append(f"seed {seed}: candidate source subsample")
        if (len(donor) != len(source) or np.any(donor < 0)
                or np.any(donor >= 2*config["n_strict"])):
            failures.append(f"seed {seed}: donor bank indexing")
        if not .4 <= orientation.mean() <= .6:
            failures.append(f"seed {seed}: orientation imbalance")
        if foundation["dgp_diagnostics"]["nuisance_bank_rows"] != 2*config["n_strict"]:
            failures.append(f"seed {seed}: wrong donor bank")
        diagnostic = scores["diagnostics"]["operators"]["resampling"]
        if diagnostic["oracle_residual"] > 1e-10:
            failures.append(f"seed {seed}: oracle residual")
        if not np.isfinite(diagnostic["packet_moment_relative_error"]):
            failures.append(f"seed {seed}: packet moment")
        for packet in ("symmetric", "realized"):
            paths = sorted((base/"theory"/packet).glob("j*.json.gz"))
            counts["theory"] += len(paths)
            if args.require_complete and len(paths) != config["n_candidates"]:
                failures.append(f"seed {seed}/{packet}: {len(paths)} theory cells")
            for path in paths:
                value = _load(path)
                if (value["config_digest"], value["implementation_digest"]) != pair:
                    failures.append(f"{path}: provenance")
                if value["packet"] != packet or value["packet_weight_relative_to_strict_mean"] != 1/config["n_strict"]:
                    failures.append(f"{path}: packet/weight")
                if value["gradient_norm"] > config["gradient_tolerance"] or value["hessian_minimum_eigenvalue"] <= 0:
                    failures.append(f"{path}: retrain convergence")
                if packet == "realized" and value["a_wins"] != bool(orientation[value["candidate_index"]]):
                    failures.append(f"{path}: orientation")
        endpoints = sorted(
            path for operator in config["operators"]
            for path in (base/operator).glob("*/*.json.gz"))
        counts["endpoints"] += len(endpoints)
        if args.require_complete and len(endpoints) != expected_endpoints:
            failures.append(f"seed {seed}: {len(endpoints)} endpoint cells")
        full_pool_energy = []
        full_pool_accuracy = []
        for path in endpoints:
            value = _load(path)
            if (value["config_digest"], value["implementation_digest"]) != pair:
                failures.append(f"{path}: provenance")
            selected = np.asarray(value["selected_indices"])
            if len(selected) != value["budget"] or len(np.unique(selected)) != len(selected):
                failures.append(f"{path}: selected set")
            if (value["initialization"] != expected_initialization
                    or value["training_packet"] != "single_prerealized_uniform"):
                failures.append(f"{path}: training protocol")
            metrics = value["metrics"]
            if metrics["gradient_norm"] > config["gradient_tolerance"] or metrics["hessian_minimum_eigenvalue"] <= 0:
                failures.append(f"{path}: endpoint convergence")
            if any(not 0 <= x <= 1 for x in metrics["accuracy"].values()):
                failures.append(f"{path}: accuracy range")
            energy = metrics["energy"]
            if not np.isclose(energy["normalized_spurious_energy"], energy["spurious_energy"]/energy["total_energy"]):
                failures.append(f"{path}: energy normalization")
            if value["budget"] == config["n_candidates"]:
                full_pool_energy.append(energy["spurious_energy"])
                full_pool_accuracy.append(metrics["accuracy"]["Q_mixed"])
        if len(full_pool_energy) > 1 and (np.ptp(full_pool_energy) > 1e-9 or np.ptp(full_pool_accuracy) > 1e-9):
            failures.append(f"seed {seed}: k=500 selectors disagree")
    if len(provenance) > 1:
        failures.append("mixed campaign provenance")
    result = {"passed": not failures, "counts": counts, "failures": failures}
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
