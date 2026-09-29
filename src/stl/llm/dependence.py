"""Counterfactual margin energy for implicit LLM rewards."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .loglik import sequence_logprobs
from .tokenization import conditional_length, encode_conditional


def implicit_pair_margins(cfg, model, tokenizer, rows, device="cuda:0"):
    """Batched reference-adjusted response-A-minus-response-B margins."""
    if not rows:
        raise ValueError("counterfactual probe must contain at least one row")
    lengths = [conditional_length(tokenizer, row["prompt"], row[key])
               for row in rows for key in ("response_a", "response_b")]
    if any(length > cfg.max_length for length in lengths):
        raise ValueError("counterfactual probe would truncate a response")
    encoded_a = [encode_conditional(
        tokenizer, row["prompt"], row["response_a"], cfg.max_length)
        for row in rows]
    encoded_b = [encode_conditional(
        tokenizer, row["prompt"], row["response_b"], cfg.max_length)
        for row in rows]
    pad = tokenizer.pad_token_id
    lp_a = sequence_logprobs(model, encoded_a, device, pad, cfg.eval_batch_size)
    lp_b = sequence_logprobs(model, encoded_b, device, pad, cfg.eval_batch_size)
    with model.disable_adapter():
        ref_a = sequence_logprobs(model, encoded_a, device, pad, cfg.eval_batch_size)
        ref_b = sequence_logprobs(model, encoded_b, device, pad, cfg.eval_batch_size)
    return np.asarray([
        cfg.beta_dpo * ((a - b) - (ra - rb))
        for a, b, ra, rb in zip(lp_a, lp_b, ref_a, ref_b)
    ], dtype=np.float64)


def summarize_counterfactual_energy(rows, margins, *,
                                    bootstrap_replicates=1000, seed=0):
    """Summarize the paper's mean-squared counterfactual margin.

    Bootstrap units are causal anchors, not individual redraws. This keeps
    uncertainty valid when operator realizations share a retained response.
    """
    values = np.asarray(margins, dtype=np.float64)
    if len(rows) != len(values) or not len(rows):
        raise ValueError("counterfactual rows and margins must align")
    groups: dict[str, list[float]] = defaultdict(list)
    for row, margin in zip(rows, values):
        groups[str(row["anchor_id"])].append(float(margin * margin))
    redraw_counts = {len(group) for group in groups.values()}
    if len(redraw_counts) != 1:
        raise ValueError("every counterfactual anchor needs the same redraw count")
    redraws = next(iter(redraw_counts))
    if redraws < 1:
        raise ValueError("counterfactual probe needs at least one redraw")
    per_anchor = np.asarray([np.mean(group) for group in groups.values()])
    energy = float(per_anchor.mean())
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([
        per_anchor[rng.integers(0, len(per_anchor), size=len(per_anchor))].mean()
        for _ in range(bootstrap_replicates)
    ])
    quantiles = np.quantile(values, [.05, .25, .5, .75, .95])
    return {
        "schema_version": 2,
        "estimator": "mean_squared_reference_adjusted_counterfactual_margin",
        "definition": "mean((beta*((logpi(a)-logpi(b))-(logref(a)-logref(b))))^2)",
        "anchors": len(per_anchor),
        "redraws_per_anchor": redraws,
        "counterfactual_energy": energy,
        "counterfactual_energy_bootstrap_se": float(bootstrap.std(ddof=1)),
        "margin_mean": float(values.mean()),
        "margin_standard_deviation": float(values.std(ddof=1)),
        "margin_mean_absolute": float(np.abs(values).mean()),
        "margin_root_mean_square": float(np.sqrt(np.mean(values**2))),
        "margin_positive_fraction": float(np.mean(values > 0)),
        "margin_minimum": float(values.min()),
        "margin_q05": float(quantiles[0]),
        "margin_q25": float(quantiles[1]),
        "margin_median": float(quantiles[2]),
        "margin_q75": float(quantiles[3]),
        "margin_q95": float(quantiles[4]),
        "margin_maximum": float(values.max()),
    }


def evaluate_counterfactual_energy(cfg, model, tokenizer, rows,
                                   device="cuda:0",
                                   bootstrap_replicates=1000):
    margins = implicit_pair_margins(cfg, model, tokenizer, rows, device)
    return summarize_counterfactual_energy(
        rows, margins, bootstrap_replicates=bootstrap_replicates,
        seed=cfg.seed + 8801)
