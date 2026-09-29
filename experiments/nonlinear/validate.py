#!/usr/bin/env python
"""Validate foundations, scores, and optionally every completed endpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))

from stl.nonlinear.campaign import (  # noqa: E402
    config_digest, config_from_dict, implementation_digest,
)


def load_config(path: Path, width: int):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.pop("name", None)
    raw.pop("endpoint_initialization", None)
    raw.pop("output_root", None)
    widths = tuple(raw.pop("widths"))
    if width not in widths:
        raise ValueError("width is not present in the configuration")
    raw["width"] = width
    return config_from_dict(raw)


def validate_width(root: Path, cfg, require_complete: bool, *,
                   seeds: list[int] | None = None,
                   foundation_root: Path | None = None,
                   endpoint_initialization: str = "common_initialization") -> dict:
    failures = []
    source_root = root if foundation_root is None else foundation_root
    foundations, scores = {}, {}
    implementation_mismatches: list[int] = []
    selected_seeds = list(cfg.seeds if seeds is None else seeds)
    for seed in selected_seeds:
        foundation_path = source_root/"foundations"/f"seed_{seed:02d}.pt"
        score_path = source_root/"scores"/f"seed_{seed:02d}.pt"
        if not foundation_path.exists() or not score_path.exists():
            if require_complete:
                failures.append(f"seed {seed}: missing foundation or scores")
            continue
        foundation = torch.load(foundation_path, map_location="cpu", weights_only=False)
        score = torch.load(score_path, map_location="cpu", weights_only=False)
        foundations[seed], scores[seed] = foundation, score
        if foundation["config_digest"] != config_digest(cfg):
            failures.append(f"seed {seed}: config digest mismatch")
        if score["implementation_digest"] != foundation["implementation_digest"]:
            failures.append(f"seed {seed}: foundation/score implementation mismatch")
        if foundation["implementation_digest"] != implementation_digest():
            implementation_mismatches.append(seed)
        if score["strict_digest"] != foundation["strict_digest"]:
            failures.append(f"seed {seed}: strict digest mismatch")
        latent = foundation["data"]["strict_latents"]
        if len(latent["strict"]["phi_w"]) != cfg.n_strict:
            failures.append(f"seed {seed}: wrong strict count")
        pools = foundation["data"]["pools"]
        pool = pools["resampling"]
        source = np.asarray(pool["source_indices"])
        if (len(source) != cfg.n_candidates
                or len(np.unique(source)) != cfg.n_candidates):
            failures.append(f"seed {seed}: invalid candidate source subsample")
        if len(pool["nuisance_bank"]) != 2*cfg.n_strict:
            failures.append(f"seed {seed}: donor bank does not cover strict set")
        curvature = np.asarray(foundation["data"]["curvature_indices"])
        if (len(curvature) != cfg.n_curvature
                or len(np.unique(curvature)) != cfg.n_curvature
                or curvature.min() < 0 or curvature.max() >= cfg.n_strict):
            failures.append(f"seed {seed}: invalid curvature subsample")
        expected_bank = torch.cat((latent["s_w"], latent["s_l"]), dim=0)
        if not torch.equal(pool["nuisance_bank"], expected_bank):
            failures.append(f"seed {seed}: donor bank content mismatch")
        if any(len(item["first"]) != cfg.n_candidates for item in pools.values()):
            failures.append(f"seed {seed}: wrong candidate count")
        donors = pools["resampling"]["donor_index"]
        if int(donors.min()) < 0 or int(donors.max()) >= 2*cfg.n_strict:
            failures.append(f"seed {seed}: donor index outside both-side bank")
        if not torch.equal(
                pools["resampling"]["first"],
                latent["strict"]["phi_w"][torch.as_tensor(source)]):
            failures.append(f"seed {seed}: candidate does not retain strict winner")
        if pools["resampling"]["a_wins"].dtype != torch.bool:
            failures.append(f"seed {seed}: missing fixed realized orientation")
        realized_fraction = float(pools["resampling"]["a_wins"].float().mean())
        if not 0.45 <= realized_fraction <= 0.55:
            failures.append(f"seed {seed}: realized orientations are imbalanced")
        if foundation.get("optimizer") != cfg.optimizer:
            failures.append(f"seed {seed}: strict optimizer mismatch")
        if foundation.get("reference_margin") != "implicit_zero":
            failures.append(f"seed {seed}: reference margin is not zero")
        hashes = foundation["data"]["split_hashes"]
        if len(set(hashes.values())) != len(hashes):
            failures.append(f"seed {seed}: split hashes are not disjoint")
        curvature = score["diagnostics"].get("curvature")
        if "Influence" in cfg.methods:
            if not curvature or curvature.get("solver") != "woodbury_cholesky":
                failures.append(f"seed {seed}: missing Woodbury GGN diagnostics")
            else:
                worst = max(
                    item["relative_residual"]
                    for item in curvature["systems"].values())
                if worst > cfg.ggn_solve_tolerance:
                    failures.append(
                        f"seed {seed}: Woodbury residual {worst:.3g}")
        for operator in cfg.operators:
            for method in (value for value in cfg.methods if value != "UTL"):
                vector = np.asarray(score["scores"][operator][method])
                ranking = np.asarray(score["rankings"][operator][method])
                if len(vector) != cfg.n_candidates or not np.isfinite(vector).all():
                    failures.append(f"seed {seed}: bad {operator}/{method} scores")
                if len(np.unique(ranking)) != cfg.n_candidates:
                    failures.append(f"seed {seed}: bad {operator}/{method} ranking")

    endpoint_count = 0
    per_seed_operator = sum(
        1 if budget == cfg.n_candidates else sum(
            cfg.utl_draws if method == "UTL" else 1
            for method in cfg.methods)
        for budget in cfg.budgets)
    expected = len(selected_seeds)*len(cfg.operators)*per_seed_operator
    for path in root.glob("endpoints/*/seed_*/*.pt"):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if int(value["seed"]) not in selected_seeds:
            continue
        endpoint_count += 1
        seed = int(value["seed"])
        if seed not in foundations:
            failures.append(f"{path.name}: missing foundation")
            continue
        expected_start = (
            foundations[seed]["initial_digest"]
            if endpoint_initialization == "common_initialization"
            else foundations[seed]["strict_digest"])
        if (value["initial_digest"] != expected_start
                or value.get("endpoint_initialization",
                             "common_initialization") != endpoint_initialization):
            failures.append(f"{path.name}: wrong endpoint initialization")
        if value.get("training_packet") != "single_prerealized_uniform":
            failures.append(f"{path.name}: wrong tie training packet")
        if value.get("scoring_packet") != "exact_symmetric_average":
            failures.append(f"{path.name}: wrong scoring packet")
        if value.get("optimizer") != cfg.optimizer:
            failures.append(f"{path.name}: endpoint optimizer mismatch")
        if value.get("reference_margin") != "implicit_zero":
            failures.append(f"{path.name}: reference margin is not zero")
        selected = np.asarray(value["selected_indices"])
        if len(selected) != value["budget"] or len(np.unique(selected)) != len(selected):
            failures.append(f"{path.name}: repeated/wrong selected indices")
        if not value.get("reload_validation", {}).get("passed"):
            failures.append(f"{path.name}: reload validation absent")
        for metric in value["metrics"]["accuracy"].values():
            if not np.isfinite(metric):
                failures.append(f"{path.name}: nonfinite accuracy")
        energy = value["metrics"].get("energy", {})
        required_energy = {
            "counterfactual_energy",
            "anchors", "redraws_per_anchor", "heldout_ties",
        }
        if (not required_energy <= set(energy)
                or not all(np.isfinite(energy[key]) for key in required_energy)
                or energy["counterfactual_energy"] < 0.0
                or energy["anchors"] != cfg.n_energy_probe
                or energy["redraws_per_anchor"] != cfg.energy_redraws
                or energy["heldout_ties"] != (
                    cfg.n_energy_probe * cfg.energy_redraws)):
            failures.append(f"{path.name}: invalid endpoint energy")
    full_pool = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in root.glob("endpoints/*/seed_*/*.pt")
        if (f"k{cfg.n_candidates:05d}" in path.stem
            and int(path.parent.name.removeprefix("seed_")) in selected_seeds)]
    by_seed: dict[int, list[str]] = {}
    for value in full_pool:
        by_seed.setdefault(int(value["seed"]), []).append(value["state_digest"])
    for seed, digests in by_seed.items():
        if len(digests) != 1:
            failures.append(f"seed {seed}: expected one shared full-pool endpoint")
    if require_complete and endpoint_count != expected:
        failures.append(f"expected {expected} endpoints, found {endpoint_count}")
    if implementation_mismatches:
        failures.append(
            "implementation digest mismatch for seeds "
            + ",".join(map(str, implementation_mismatches)))
    result = {
        "width": cfg.width, "foundations": len(foundations),
        "scores": len(scores), "endpoints": endpoint_count,
        "expected_endpoints": expected,
        "implementation_status": "current",
        "current_implementation_digest": implementation_digest(),
        "failures": failures,
        "passed": not failures,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--widths", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--foundation-root", type=Path)
    parser.add_argument(
        "--endpoint-initialization",
        choices=("common_initialization", "strict_checkpoint"),
        help="optional override of the protocol in the configuration")
    args = parser.parse_args()
    experiment = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    widths = args.widths or list(map(int, experiment["widths"]))
    root = args.root or ROOT/experiment.get(
        "output_root", "results/nonlinear/paper_aligned")
    initialization = (args.endpoint_initialization
                      or experiment.get("endpoint_initialization",
                                        "common_initialization"))
    def width_root(base: Path, width: int) -> Path:
        """Accept either a campaign root or one explicit width directory."""
        if any((base/name).exists() for name in (
                "foundations", "scores", "endpoints")):
            return base
        return base/f"w{width}"
    results = [validate_width(
        width_root(root, width), load_config(args.config, width),
        args.require_complete,
        seeds=args.seeds,
        foundation_root=(width_root(args.foundation_root, width)
                         if args.foundation_root else None),
        endpoint_initialization=initialization)
        for width in widths]
    print(json.dumps(results, indent=2))
    if not all(value["passed"] for value in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
