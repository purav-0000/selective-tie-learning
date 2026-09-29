"""Foundations, matched scoring, fresh endpoints, and provenance."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

from .dgp import ENVIRONMENTS, GenericNonlinearDGP, LatentPairBatch, PreferenceBatch
from .training import (
    clone_state, configure_fp32, expected_bt_accuracy, fit_from_initialization,
    load_state, make_model, state_digest, strict_fit_diagnostics, tensor_digest,
)
from .config import ResamplingInfluenceConfig
from .data import (
    PacketPool, PreferenceLatents, ProbeBank, make_packet_pools,
    make_probe_bank, sample_evaluations, sample_preference_latents,
)
from .dependence import counterfactual_energy
from .scoring import compute_matched_scores


def config_from_dict(values: dict) -> ResamplingInfluenceConfig:
    valid = {item.name for item in fields(ResamplingInfluenceConfig)}
    unknown = set(values)-valid
    if unknown:
        raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
    normalized = dict(values)
    for name in ("budgets", "methods", "operators", "seeds"):
        if name in normalized:
            normalized[name] = tuple(normalized[name])
    return ResamplingInfluenceConfig(**normalized)


def config_digest(cfg: ResamplingInfluenceConfig) -> str:
    values = cfg.to_dict()
    # Device placement is runtime provenance, not scientific configuration.
    # Excluding it lets a CUDA-produced foundation validate after CPU reload.
    values.pop("device", None)
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def implementation_digest() -> str:
    digest = hashlib.sha256()
    directories = [Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
    retained = {"dgp.py", "training.py", "scoring.py"}
    paths = list(directories[0].glob("*.py"))
    paths.extend(path for path in directories[1].glob("*.py") if path.name in retained)
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(path.relative_to(directories[1]).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def stream_seed(seed: int, name: str) -> int:
    payload = f"nonlinear-resampling-influence-v1:{seed}:{name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63-1)


def atomic_torch_save(value: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, destination)


def atomic_json(value: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)


def _preference_dict(batch: PreferenceBatch) -> dict:
    return {name: getattr(batch, name).detach().cpu()
            for name in ("phi_w", "phi_l", "q_w", "q_l")}


def _preference(value: dict, device: str, dtype=None) -> PreferenceBatch:
    values = []
    for name in ("phi_w", "phi_l", "q_w", "q_l"):
        tensor = value[name].to(device)
        values.append(tensor.to(dtype) if dtype is not None else tensor)
    return PreferenceBatch(*values)


def _latents_dict(batch: PreferenceLatents) -> dict:
    return {
        "strict": _preference_dict(batch.strict),
        **{name: getattr(batch, name).detach().cpu()
           for name in ("c_w", "s_w", "q_w", "c_l", "s_l", "q_l")},
    }


def _pool_dict(pool: PacketPool) -> dict:
    return {name: getattr(pool, name).detach().cpu() for name in (
        "first", "second", "anchor_c", "donor_index", "a_wins",
        "source_indices", "nuisance_bank")}


def _pool(value: dict, device: str, dtype=None) -> PacketPool:
    continuous = []
    for name in ("first", "second", "anchor_c"):
        tensor = value[name].to(device)
        continuous.append(tensor.to(dtype) if dtype is not None else tensor)
    bank = value["nuisance_bank"].to(device)
    if dtype is not None:
        bank = bank.to(dtype)
    return PacketPool(
        *continuous, value["donor_index"].clone(), value["a_wins"].clone(),
        value["source_indices"].clone(), bank)


def _probe_dict(probe: ProbeBank) -> dict:
    return {name: getattr(probe, name).detach().cpu() for name in (
        "resampling_first", "resampling_second", "resampling_redraws",
        "anchor_c", "nuisance_bank", "donor_indices")}


def _probe(value: dict, device: str, dtype=None) -> ProbeBank:
    values = []
    for name in (
        "resampling_first", "resampling_second", "resampling_redraws",
        "anchor_c", "nuisance_bank",
    ):
        tensor = value[name].to(device)
        values.append(tensor.to(dtype) if dtype is not None else tensor)
    return ProbeBank(*values, value["donor_indices"].clone())


def _evaluation_dict(batch: LatentPairBatch) -> dict:
    return {name: getattr(batch, name).detach().cpu()
            for name in ("phi_a", "phi_b", "q_a", "q_b")}


def _evaluation(value: dict, device: str) -> LatentPairBatch:
    return LatentPairBatch(*(value[name].to(device)
                             for name in ("phi_a", "phi_b", "q_a", "q_b")))


def build_data_bundle(cfg: ResamplingInfluenceConfig, seed: int) -> dict:
    dgp = GenericNonlinearDGP(
        seed=stream_seed(seed, "geometry"), d_c=cfg.d_c, d_s=cfg.d_s,
        block_condition_number=cfg.block_condition_number,
        r_min=cfg.r_min, r_max=cfg.r_max,
        gamma_c=cfg.gamma_c, gamma_s=cfg.gamma_s,
        beta_teacher=cfg.beta_teacher, mixer_hidden=cfg.mixer_hidden,
        mixer_seed=cfg.mixer_seed, device=cfg.device)
    strict_latents = sample_preference_latents(
        dgp, cfg.n_strict, seed=stream_seed(seed, "strict"))
    curvature_generator = torch.Generator(device="cpu").manual_seed(
        stream_seed(seed, "curvature_subsample"))
    curvature_indices = torch.randperm(
        cfg.n_strict, generator=curvature_generator)[:cfg.n_curvature].sort().values
    pools = make_packet_pools(
        dgp, strict_latents, n_candidates=cfg.n_candidates,
        seed=stream_seed(seed, "candidate_donors"))
    selection_latents = sample_preference_latents(
        dgp, cfg.n_selection_probe, seed=stream_seed(seed, "selection_bank"))
    energy_latents = sample_preference_latents(
        dgp, cfg.n_energy_probe, seed=stream_seed(seed, "energy_bank"))
    strict_nuisance_bank = strict_latents.nuisance_bank
    selection = make_probe_bank(
        dgp, selection_latents, nuisance_bank=strict_nuisance_bank, redraws=2,
        seed=stream_seed(seed, "selection_donors"))
    energy = make_probe_bank(
        dgp, energy_latents, nuisance_bank=strict_nuisance_bank,
        redraws=cfg.energy_redraws,
        seed=stream_seed(seed, "energy_donors"))
    evaluations = sample_evaluations(
        dgp, tuple(ENVIRONMENTS), cfg.n_evaluation,
        lambda name: stream_seed(seed, f"evaluation:{name}"))

    strict_value = _latents_dict(strict_latents)
    pool_values = {name: _pool_dict(value) for name, value in pools.items()}
    selection_value, energy_value = _probe_dict(selection), _probe_dict(energy)
    evaluation_values = {name: _evaluation_dict(value)
                         for name, value in evaluations.items()}
    hashes = {
        "strict": tensor_digest(*strict_value["strict"].values()),
        "strict_latents": tensor_digest(*(
            strict_value[name] for name in
            ("c_w", "s_w", "q_w", "c_l", "s_l", "q_l"))),
        **{f"pool:{name}": tensor_digest(*value.values())
           for name, value in pool_values.items()},
        "selection_probe": tensor_digest(*selection_value.values()),
        "energy_probe": tensor_digest(*energy_value.values()),
        **{f"evaluation:{name}": tensor_digest(*value.values())
           for name, value in evaluation_values.items()},
    }
    return {
        "seed": seed,
        "config_digest": config_digest(cfg),
        "implementation_digest": implementation_digest(),
        "strict_latents": strict_value,
        "curvature_indices": curvature_indices,
        "pools": pool_values,
        "selection_probe": selection_value,
        "energy_probe": energy_value,
        "evaluation": evaluation_values,
        "split_hashes": hashes,
        "stream_seeds": {name: stream_seed(seed, name) for name in (
            "geometry", "strict", "candidate_donors", "selection_bank",
            "selection_donors", "energy_bank", "energy_donors",
            "initialization", "strict_training_schedule",
            "curvature_subsample")},
        "geometry": {
            "A": dgp.A, "B": dgp.B, "R": dgp.R,
            "u_c": dgp.u_c, "u_s": dgp.u_s,
            "diagnostics": dgp.geometry_diagnostics(),
            "moments": {name: dgp.option_moments(name) for name in ENVIRONMENTS},
        },
        "candidate_protocol": {
            "source_pairs": cfg.n_candidates,
            "donor_bank_vectors": 2*cfg.n_strict,
            "donor_bank": cfg.candidate_bank,
            "zero_contrast_ties": int(torch.all(
                pools["resampling"].first == pools["resampling"].second,
                dim=1).sum().item()),
            "realized_a_fraction": float(
                pools["resampling"].a_wins.float().mean().item()),
        },
    }


def materialize_bundle(bundle: dict, device: str, *, dtype=None) -> dict:
    strict_value = bundle["strict_latents"]["strict"]
    strict = _preference(strict_value, device, dtype)
    curvature_indices = bundle["curvature_indices"].to(device)
    curvature_strict = PreferenceBatch(*(
        getattr(strict, name)[curvature_indices]
        for name in ("phi_w", "phi_l", "q_w", "q_l")))
    return {
        "strict": strict,
        "curvature_strict": curvature_strict,
        "pools": {name: _pool(value, device, dtype)
                  for name, value in bundle["pools"].items()},
        "selection_probe": _probe(bundle["selection_probe"], device, dtype),
        "energy_probe": _probe(bundle["energy_probe"], device, dtype),
        "evaluation": {name: _evaluation(value, device)
                       for name, value in bundle["evaluation"].items()},
    }


def evaluate_checkpoint(
    cfg: ResamplingInfluenceConfig, model: torch.nn.Module, data: dict,
) -> dict:
    energy = counterfactual_energy(
        model, data["energy_probe"], cfg.beta_dpo)
    return {
        "accuracy": {name: expected_bt_accuracy(model, batch, cfg.beta_teacher)
                     for name, batch in data["evaluation"].items()},
        "energy": energy,
    }


def build_foundation(cfg: ResamplingInfluenceConfig, seed: int) -> dict:
    configure_fp32()
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(cfg.device))
    started = time.perf_counter()
    bundle = build_data_bundle(cfg, seed)
    data = materialize_bundle(bundle, cfg.device)
    model = make_model(cfg, stream_seed(seed, "initialization"))
    initial_state = clone_state(model)
    strict_model, fitting = fit_from_initialization(
        cfg, initial_state, data["strict"],
        schedule_seed=stream_seed(seed, "strict_training_schedule"))
    strict_state = clone_state(strict_model)
    return {
        "format_version": 1,
        "kind": "nonlinear_resampling_influence_foundation",
        "seed": seed,
        "config": cfg.to_dict(),
        "config_digest": config_digest(cfg),
        "implementation_digest": implementation_digest(),
        "data": bundle,
        "initial_state": initial_state,
        "initial_digest": state_digest(initial_state),
        "strict_state": strict_state,
        "strict_digest": state_digest(strict_state),
        "strict_fitting": fitting,
        "strict_diagnostics": {
            **strict_fit_diagnostics(strict_model, data["strict"], cfg.beta_dpo),
            **evaluate_checkpoint(cfg, strict_model, data),
        },
        "reference_margin": "implicit_zero",
        "optimizer": cfg.optimizer,
        "runtime_seconds": time.perf_counter()-started,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(torch.device(cfg.device))
            if cfg.device.startswith("cuda") else 0),
    }


def score_foundation(cfg: ResamplingInfluenceConfig, foundation: dict) -> dict:
    if foundation["config_digest"] != config_digest(cfg):
        raise ValueError("foundation/config digest mismatch")
    if foundation["implementation_digest"] != implementation_digest():
        raise ValueError("foundation/implementation digest mismatch")
    data = materialize_bundle(
        foundation["data"], cfg.device, dtype=torch.float64)
    model = make_model(cfg, 0).double()
    load_state(model, {name: value.to(cfg.device)
                       for name, value in foundation["strict_state"].items()})
    started = time.perf_counter()
    result = compute_matched_scores(
        cfg, model, data["curvature_strict"], data["pools"],
        data["selection_probe"], seed=foundation["seed"])
    utl_rankings = {}
    for draw in range(cfg.utl_draws):
        rng = np.random.default_rng(stream_seed(
            foundation["seed"], f"utl_ranking:{draw}"))
        utl_rankings[str(draw)] = rng.permutation(cfg.n_candidates)
    return {
        "format_version": 1,
        "kind": "nonlinear_resampling_influence_scores",
        "seed": foundation["seed"],
        "config_digest": config_digest(cfg),
        "implementation_digest": implementation_digest(),
        "strict_digest": foundation["strict_digest"],
        "scores": result.scores,
        "rankings": result.rankings,
        "utl_rankings": utl_rankings,
        "diagnostics": result.diagnostics,
        "runtime_seconds": time.perf_counter()-started,
    }


def endpoint_identifier(
    operator: str, method: str, budget: int, draw: int | None = None,
) -> str:
    suffix = f"_draw{draw}" if method == "UTL" else ""
    method_name = method.lower().replace("-", "_")
    return f"{operator}_{method_name}{suffix}_k{budget:05d}"


def run_endpoint(
    cfg: ResamplingInfluenceConfig, foundation: dict, scores: dict, *,
    operator: str, method: str, budget: int, draw: int | None = None,
) -> dict:
    if operator not in cfg.operators or method not in cfg.methods:
        raise ValueError("unknown operator or method")
    if budget not in cfg.budgets:
        raise ValueError("budget is not present in the configuration")
    if method == "UTL":
        if draw is None or not 0 <= draw < cfg.utl_draws:
            raise ValueError("UTL requires a valid draw")
        ranking = scores["utl_rankings"][str(draw)]
    else:
        if draw is not None:
            raise ValueError("targeted methods do not take a draw")
        ranking = scores["rankings"][operator][method]
    # The selector determines a set, not an optimizer-specific row order.
    # Sorting gives every method the same schedule mapping for the same set
    # and forces exact agreement when the full pool is selected.
    selected = np.sort(np.asarray(ranking[:budget], dtype=np.int64))
    if len(np.unique(selected)) != budget:
        raise AssertionError("selector repeated a candidate")
    if scores["strict_digest"] != foundation["strict_digest"]:
        raise ValueError("score/strict digest mismatch")
    expected_implementation = implementation_digest()
    if (foundation["implementation_digest"] != expected_implementation
            or scores["implementation_digest"] != expected_implementation):
        raise ValueError("endpoint/implementation digest mismatch")
    data = materialize_bundle(foundation["data"], cfg.device)
    pool = data["pools"][operator]
    indices = torch.as_tensor(selected, device=cfg.device)
    initial_state = {name: value.to(cfg.device)
                     for name, value in foundation["initial_state"].items()}
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(cfg.device))
    started = time.perf_counter()
    selected_a_wins = pool.a_wins[indices.cpu()].to(cfg.device)
    first = torch.where(selected_a_wins[:, None], pool.first[indices], pool.second[indices])
    second = torch.where(selected_a_wins[:, None], pool.second[indices], pool.first[indices])
    model, fitting = fit_from_initialization(
        cfg, initial_state, data["strict"], first, second,
        schedule_seed=stream_seed(foundation["seed"], f"endpoint_schedule:k={budget}"))
    state = clone_state(model)
    return {
        "format_version": 1,
        "kind": "nonlinear_resampling_influence_endpoint",
        "identifier": endpoint_identifier(operator, method, budget, draw),
        "seed": foundation["seed"],
        "width": cfg.width,
        "operator": operator,
        "method": method,
        "draw": draw,
        "budget": budget,
        "config_digest": config_digest(cfg),
        "implementation_digest": expected_implementation,
        "initial_digest": foundation["initial_digest"],
        "strict_digest": foundation["strict_digest"],
        "selected_indices": selected,
        "selected_hash": hashlib.sha256(selected.tobytes()).hexdigest(),
        "realized_a_fraction": float(selected_a_wins.float().mean()),
        "training_packet": "single_prerealized_uniform",
        "scoring_packet": "exact_symmetric_average",
        "reference_margin": "implicit_zero",
        "optimizer": cfg.optimizer,
        "state": state,
        "state_digest": state_digest(state),
        "fitting": fitting,
        "metrics": evaluate_checkpoint(cfg, model, data),
        "runtime_seconds": time.perf_counter()-started,
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(torch.device(cfg.device))
            if cfg.device.startswith("cuda") else 0),
    }


def validate_reloaded_endpoint(
    cfg: ResamplingInfluenceConfig, foundation: dict, endpoint: dict,
    tolerance: float = 2e-6,
) -> dict:
    data = materialize_bundle(foundation["data"], cfg.device)
    model = make_model(cfg, 0)
    load_state(model, {name: value.to(cfg.device)
                       for name, value in endpoint["state"].items()})
    reproduced = evaluate_checkpoint(cfg, model, data)
    maximum = 0.0
    for name, value in reproduced["accuracy"].items():
        maximum = max(maximum, abs(value-endpoint["metrics"]["accuracy"][name]))
    for name, left in reproduced["energy"].items():
        right = endpoint["metrics"]["energy"][name]
        if (isinstance(left, (int, float)) and not isinstance(left, bool)
                and np.isfinite(left) and np.isfinite(right)):
            maximum = max(maximum, abs(float(left)-float(right)))
    if maximum > tolerance:
        raise AssertionError(
            f"reloaded endpoint metric error {maximum} exceeds {tolerance}")
    return {"maximum_absolute_metric_error": maximum, "passed": True}
