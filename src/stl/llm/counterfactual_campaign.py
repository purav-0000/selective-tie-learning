"""Scientific implementation for one-sided hotel and direct emoji campaigns."""
from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import random
import shutil
import time
from typing import Any

from .config import LLMConfig
from .counterfactual_operator import (
    counterfactual_digest, dgp_summary, emoji_direct_bundle,
    emoji_energy_probe, hotel_bundle, hotel_energy_probe,
)
from .factorized_hotel import evaluate_implicit_reward, validate_paired_environments, write_jsonl
from .provenance import atomic_json, checkpoint_digest, digest_json, read_jsonl


HOTEL_DATASETS = {"hotel_one_sided"}


def _validate_energy(value: dict[str, Any]) -> None:
    required = ("counterfactual_energy",)
    if any(key not in value for key in required):
        raise ValueError("incomplete endpoint counterfactual energy")
    numeric = [float(value[key]) for key in required]
    if (not all(math.isfinite(item) for item in numeric)
            or any(item < 0.0 for item in numeric)):
        raise ValueError("non-finite counterfactual energy")


def config(raw: dict[str, Any], seed: int, *, symmetric: bool) -> LLMConfig:
    """Build the immutable runtime configuration for one seed."""
    data, model, training, scoring = (raw[k] for k in ("data", "model", "training", "scoring"))
    return LLMConfig(
        N=int(data["strict_rows"]), N_test=int(data["test_rows"]),
        rho=float(data["rho"]), utility_noise_std=float(data.get("utility_noise_std", 0)),
        strict_min_gap=float(data.get("strict_min_gap", 0)),
        base_model=model["id"], base_model_revision=model["revision"],
        lora_r=int(model["lora_rank"]), lora_alpha=int(model["lora_alpha"]),
        lora_dropout=float(model["lora_dropout"]), lora_targets=list(model["lora_targets"]),
        attn_implementation=model["attention"], beta_dpo=float(training["beta"]),
        learning_rate=float(training["learning_rate"]), epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]), grad_accum=int(training["gradient_accumulation"]),
        max_grad_norm=float(training.get("max_gradient_norm", 1.0)),
        max_length=int(training["max_sequence_length"]), warmup_ratio=float(training["warmup_ratio"]),
        optim=training["optimizer"], lr_scheduler=training["scheduler"],
        grad_checkpointing=bool(training["gradient_checkpointing"]),
        eval_batch_size=int(raw["evaluation"]["batch_size"]),
        symmetric_tie_packets=symmetric, seed=seed,
        N_ggn=int(scoring["ggn_rows"]),
        ggn_dtype=scoring["ggn_dtype"],
        ggn_damping_fraction=float(scoring["ggn_damping_fraction"]),
        ggn_solve_tol=float(scoring["ggn_solve_tolerance"]),
        probe_batch_size=int(scoring["probe_batch_size"]),
        selection_probe_size=int(scoring["selection_probe_size"]))


def _pku_sources(raw, runtime_root: Path):
    root = runtime_root / "inputs" / raw["input"]["source_input_id"]
    manifest = json.loads((root / "manifest.json").read_text())
    train, heldout = read_jsonl(root / "train_sources.jsonl"), read_jsonl(root / "test_sources.jsonl")
    if digest_json(train) != manifest["train_digest"] or digest_json(heldout) != manifest["test_digest"]:
        raise ValueError("PKU source input digest mismatch")
    return train, heldout, manifest


def build_data(raw, seed: int, runtime_root: Path):
    """Construct strict, candidate, probe, evaluation, and energy data."""
    energy_anchors = int(raw["evaluation"]["energy_anchors"])
    energy_redraws = int(raw["evaluation"]["energy_redraws"])
    if raw["dataset"] == "hotel_one_sided":
        strict, candidates, probes, heldout = hotel_bundle(raw, seed)
        energy = hotel_energy_probe(
            raw, strict, seed=seed, anchors=energy_anchors,
            redraws=energy_redraws)
        return strict, candidates, probes, heldout, energy, None
    if raw["dataset"] != "pku_direct_emoji":
        raise ValueError(f"unsupported dataset: {raw['dataset']!r}")
    train, heldout, manifest = _pku_sources(raw, runtime_root)
    n = int(raw["data"]["strict_rows"])
    strict_n = int(raw["data"]["strict_rows"])
    if len(train) < strict_n + 1000 + energy_anchors:
        raise ValueError("PKU source bank does not contain disjoint probes")
    probes = {"selection": train[strict_n:strict_n+500]}
    data = emoji_direct_bundle(
        train[:n],
        heldout,
        probes, seed=seed, rho=float(raw["data"]["rho"]),
        maximum=int(raw["data"]["maximum_emoji_events"]),
    )
    energy = emoji_energy_probe(
        train[strict_n+1000:strict_n+1000+energy_anchors], seed=seed,
        rho=float(raw["data"]["rho"]),
        maximum=int(raw["data"]["maximum_emoji_events"]),
        redraws=energy_redraws)
    return (*data, energy, manifest)


def _render(raw, tokenizer, rows):
    if raw["dataset"] in HOTEL_DATASETS:
        return rows
    from .real_data.emoji_evaluation import render_rows
    return render_rows(tokenizer, rows, date_string=raw["data"]["chat_template_date"])


def _evaluate_checkpoint(raw, cfg, adapter, heldout, energy_rows):
    from .dependence import evaluate_counterfactual_energy
    from .evaluate import load_eval_model
    from .real_data.emoji_evaluation import evaluate_adapter

    if raw["dataset"] in HOTEL_DATASETS:
        policy, tokenizer = load_eval_model(cfg, adapter)
        validate_paired_environments(heldout)
        evaluation = evaluate_implicit_reward(
            policy, tokenizer, heldout, beta=cfg.beta_dpo,
            max_length=cfg.max_length, batch_size=cfg.eval_batch_size)
        energy = evaluate_counterfactual_energy(
            cfg, policy, tokenizer, energy_rows,
            device=str(next(policy.parameters()).device))
        headline = {name:value["accuracy"] for name,value in evaluation.items()}
        del policy, tokenizer
    else:
        evaluation = evaluate_adapter(
            cfg, adapter, heldout, stratification_field="unsafe_severity",
            date_string=raw["data"]["chat_template_date"],
            energy_rows=energy_rows)
        energy = evaluation.pop("energy")
        headline = {
            name:evaluation["implicit_reward"][name]["summary"]
            ["causal_accuracy"]["accuracy"]
            for name in heldout}
    _validate_energy(energy)
    return evaluation, headline, energy


def target_candidate_panel(rows, *, size: int | None, seed: int):
    """Uniform fixed panel used only by selectors that require model scores."""
    if size is None or size >= len(rows):
        return list(rows), list(range(len(rows)))
    if size < 1:
        raise ValueError("target candidate pool size must be positive")
    indices = sorted(random.Random(83_000_019 + seed).sample(range(len(rows)), size))
    return [rows[i] for i in indices], indices


def train_strict(raw, seed: int, runtime_root: Path, output: Path):
    """Train and evaluate the strict DPO model for one seed."""
    from .dpo import load_tokenizer, train_dpo
    cfg = config(raw, seed, symmetric=False)
    strict, candidates, probes, heldout, energy_rows, source_manifest = build_data(
        raw, seed, runtime_root)
    tokenizer = load_tokenizer(cfg)
    rendered = _render(raw, tokenizer, strict)
    del tokenizer
    strict_path = output / "prepared/strict.jsonl"
    write_jsonl(strict_path, [{k:r[k] for k in ("prompt", "chosen", "rejected")}
                              for r in rendered])
    adapter = train_dpo(cfg, str(strict_path), str(output / "checkpoint"),
                        run_name=f"{raw['campaign']}_strict_s{seed}")
    evaluation, headline, energy = _evaluate_checkpoint(
        raw, cfg, adapter, heldout, energy_rows)
    result = {"campaign": raw["campaign"], "seed": seed, "headline": headline,
              "evaluation": evaluation, "energy": energy,
              "adapter_digest": checkpoint_digest(adapter),
              "strict_digest": digest_json(strict), "candidate_digest": counterfactual_digest(candidates),
              "probe_digests": {f"{a}_{n}":counterfactual_digest(v) for (a,n),v in probes.items()},
              "heldout_digest": digest_json(heldout), "source_manifest": source_manifest,
              "energy_probe_digest": digest_json(energy_rows),
              "dgp_summary": dgp_summary(strict, candidates, heldout),
              "config_digest": digest_json(raw)}
    atomic_json(output / "result.json", result)
    gc.collect()
    return result


def score(raw, seed: int, runtime_root: Path, strict_dir: Path, output: Path, update):
    """Score the shared candidate panel at the strict checkpoint."""
    from .influence_gate import validate_and_score
    from .scoring import _load_ggn_subsample, load_policy
    cfg = config(raw, seed, symmetric=True)
    strict, candidates, probes, heldout, energy_rows, _ = build_data(
        raw, seed, runtime_root)
    source = json.loads((strict_dir / "result.json").read_text())
    adapter = strict_dir / "checkpoint/model"
    if checkpoint_digest(adapter) != source["adapter_digest"]:
        raise ValueError("strict adapter changed before scoring")
    if (counterfactual_digest(candidates) != source["candidate_digest"] or
            digest_json(heldout) != source["heldout_digest"] or
            digest_json(energy_rows) != source["energy_probe_digest"]):
        raise ValueError("counterfactual data changed after strict training")
    policy, tokenizer = load_policy(cfg, str(adapter), device="cuda:0")
    candidates = _render(raw, tokenizer, candidates)
    full_candidates = candidates
    target_size = raw["selection"].get("target_candidate_pool_size")
    candidates, target_indices = target_candidate_panel(
        full_candidates, size=(None if target_size is None else int(target_size)),
        seed=seed)
    probes = {key:_render(raw, tokenizer, value) for key,value in probes.items()}
    strict_sample = _load_ggn_subsample(
        cfg, str(strict_dir / "prepared/strict.jsonl"))
    scored, gate = validate_and_score(
        cfg, policy, tokenizer, strict_sample, candidates, probes,
        None, update)
    for row in scored:
        row["pm_score"] = abs(float(row["pm_score"]))
    gate.update(campaign=raw["campaign"], seed=seed, config_digest=digest_json(raw),
                strict_adapter_digest=source["adapter_digest"],
                strict_data_digest=source["strict_digest"],
                candidate_digest=counterfactual_digest(scored),
                full_candidate_digest=counterfactual_digest(full_candidates),
                full_candidate_rows=len(full_candidates),
                target_candidate_rows=len(scored),
                target_candidate_indices=target_indices,
                target_candidate_index_digest=digest_json(target_indices),
                probe_digests=source["probe_digests"],
                heldout_digest=source["heldout_digest"],
                energy_probe_digest=source["energy_probe_digest"])
    atomic_json(output / "gate.json", gate)
    write_jsonl(output / "scored.jsonl", scored)
    if (gate["ggn_validation"] is not None and
            (not gate["ggn_validation"]["passed"] or
             not gate["campaign_gate_passed"])):
        raise RuntimeError("probe/GGN gate failed")
    del policy, tokenizer
    import torch
    torch.cuda.empty_cache(); gc.collect()


def _ranking(rows, method: str, seed: int, draw: int):
    if method == "utl":
        order = list(range(len(rows)))
        random.Random(71_000_003 + 10_000*seed + draw).shuffle(order)
        return order
    field = f"{method}_score"
    return sorted(range(len(rows)), key=lambda i:(-float(rows[i][field]), rows[i]["candidate_id"]))


def train_endpoint(raw, seed: int, runtime_root: Path, strict_dir: Path, score_dir: Path,
                   output: Path, method: str, draw: int, budget: int):
    """Train and evaluate one selector-budget endpoint."""
    from .dpo import train_dpo
    cfg = config(raw, seed, symmetric=False)
    strict, _, _, heldout, energy_rows, _ = build_data(
        raw, seed, runtime_root)
    scored = read_jsonl(score_dir / "scored.jsonl")
    source = json.loads((strict_dir / "result.json").read_text())
    gate = json.loads((score_dir / "gate.json").read_text())
    if (gate["config_digest"] != digest_json(raw) or gate["seed"] != seed or
            gate["strict_adapter_digest"] != source["adapter_digest"] or
            checkpoint_digest(strict_dir / "checkpoint/model") != source["adapter_digest"] or
            digest_json(energy_rows) != source["energy_probe_digest"] or
            gate["energy_probe_digest"] != source["energy_probe_digest"]):
        raise ValueError("endpoint provenance does not match strict/scoring state")
    candidate_rows = scored
    if (len(candidate_rows) != gate["target_candidate_rows"] or
            counterfactual_digest(candidate_rows) != gate["candidate_digest"]):
        raise ValueError("shared candidate panel changed")
    ranking = _ranking(candidate_rows, method, seed, draw)
    # Dataset order is not part of selection. Sorting makes the
    # k=panel endpoint byte-identical across selectors and UTL draws.
    selected = sorted(ranking[:budget])
    rows = [{**{k:r[k] for k in ("prompt", "chosen", "rejected")}, "realized_tie": False}
            for r in read_jsonl(strict_dir / "prepared/strict.jsonl")]
    rows.extend({**{k:candidate_rows[i][k] for k in ("prompt", "chosen", "rejected")},
                 "realized_tie": True} for i in selected)
    path = output / "prepared/augmented.jsonl"
    write_jsonl(path, rows)
    start = time.monotonic()
    initialization = raw["training"].get(
        "endpoint_initialization", "common_pretrained")
    initial_adapter = (strict_dir / "checkpoint/model"
                       if initialization == "strict_checkpoint" else None)
    adapter = train_dpo(cfg, str(path), str(output / "checkpoint"),
                        run_name=f"{raw['campaign']}_{method}_d{draw}_s{seed}_k{budget}",
                        initial_adapter_dir=(str(initial_adapter)
                                             if initial_adapter else None))
    if checkpoint_digest(strict_dir / "checkpoint/model") != source["adapter_digest"]:
        raise ValueError("strict adapter changed during endpoint continuation")
    evaluation, headline, energy = _evaluate_checkpoint(
        raw, cfg, adapter, heldout, energy_rows)
    history_path = output / "checkpoint/training_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else None
    model_digest = checkpoint_digest(adapter)
    result = {"campaign": raw["campaign"], "seed":seed, "method":method, "draw":draw,
              "budget":budget, "headline":headline, "evaluation":evaluation,
              "energy":energy, "energy_probe_digest":source["energy_probe_digest"],
              "selected_indices":selected,
              "selected_candidate_ids":[candidate_rows[i]["candidate_id"] for i in selected],
              "selected_digest":digest_json(selected), "training_history":history,
              "model_digest":model_digest, "seconds":time.monotonic()-start,
              "gradient_source":"exact_symmetric_packet_gradient" if method == "ha" else None,
              "training_packet":"single_prerealized_uniform", "curvature":
              "1000_row_damped_DPO_GGN_Woodbury" if method=="ha" else None,
              "candidate_pool_rows":len(candidate_rows),
              "candidate_pool_scope":"shared_seed_fixed_panel",
              "endpoint_initialization":initialization,
              "initial_adapter_digest": (source["adapter_digest"]
                                         if initial_adapter else None),
              "config_digest":digest_json(raw)}
    atomic_json(output / "result.json", result)
    # Evaluation above reloads the saved adapter. Compact endpoint metadata is
    # sufficient for this uncertainty campaign and limits disk usage.
    shutil.rmtree(output / "checkpoint")
    (output / "prepared/augmented.jsonl").unlink()
    gc.collect()
    return result
