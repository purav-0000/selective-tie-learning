"""Numerical validation for the empirical GGN and probe solve."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr


def cosine(first, second) -> float:
    value = float((first.double() @ second.double()).item())
    denom = float(first.double().norm().item() * second.double().norm().item())
    return value / denom if denom else 0.0


def top_overlap(first: Sequence[float], second: Sequence[float], k: int) -> float:
    if not 0 < k <= len(first) or len(first) != len(second):
        raise ValueError("top-k overlap inputs are invalid")
    left = set(np.argsort(np.asarray(first))[-k:].tolist())
    right = set(np.argsort(np.asarray(second))[-k:].tolist())
    return len(left & right) / k


def score_stability(first: Sequence[float], second: Sequence[float],
                    budgets: Sequence[int]) -> dict:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("score vectors must be aligned and nontrivial")
    rho = float(spearmanr(first, second).statistic)
    return {
        "spearman": rho,
        "top_overlap": {
            str(k): top_overlap(first, second, int(k)) for k in budgets},
    }


def probe_gate(*, gradient_cosine: float, energy_first: float,
               energy_second: float, ha: Mapping,
               criteria: Mapping[str, float]) -> tuple[bool, dict]:
    relative_energy_gap = abs(energy_first - energy_second) / max(
        0.5 * (energy_first + energy_second), 1e-12)
    checks = {
        "gradient_cosine": gradient_cosine >= criteria["gradient_cosine_min"],
        "ha_spearman": ha["spearman"] >= criteria["score_spearman_min"],
        "ha_top100": ha["top_overlap"]["100"] >= criteria["top100_overlap_min"],
        "ha_top500": ha["top_overlap"]["500"] >= criteria["top500_overlap_min"],
        "relative_energy_gap": (
            relative_energy_gap <= criteria["relative_energy_gap_max"]),
    }
    if not all(math.isfinite(float(value)) for value in (
            gradient_cosine, relative_energy_gap,
            ha["spearman"])):
        raise ValueError("probe gate contains non-finite statistics")
    return all(checks.values()), {
        "checks": checks, "relative_energy_gap": relative_energy_gap}


def matched_storage_audit(full_precision, direction, *, tolerance: float,
                          independently_recomputed=None) -> dict:
    """Measure storage rounding on identical rows, not two different backwards.

    Both operators represent the same fp32 computed rows. Optional independent
    recomputation is reported separately and checked at Gram/solve level.
    This does not claim the underlying model arithmetic is fp32.
    """
    import copy
    import torch

    if full_precision.G.dtype != torch.float32:
        raise ValueError("storage baseline must contain fp32 rows")
    quantized = copy.copy(full_precision)
    quantized.G = full_precision.G.to(torch.bfloat16)
    quantized.dtype = torch.bfloat16
    quantized._gram64 = None
    quantized.damping = None
    baseline = full_precision.G.double()
    rounded = quantized.G.double()
    norm = max(float(baseline.norm().item()), 1e-30)
    gram = baseline @ baseline.T
    gram_norm = max(float(gram.norm().item()), 1e-30)
    row_error = float((baseline - rounded).norm().item() / norm)
    gram_error = float((gram - rounded @ rounded.T).norm().item() / gram_norm)
    x32, info32 = full_precision.solve(direction, tolerance)
    xbf, infobf = quantized.solve(direction, tolerance)
    result = {
        "comparison": "one_fp32_row_realization_cast_to_bf16",
        "rows": full_precision.N,
        "bf16_row_relative_error": row_error,
        "bf16_gram_relative_error": gram_error,
        "fp32_bf16_solve_cosine": cosine(x32, xbf),
        "fp32_solve": info32, "bf16_solve": infobf,
    }
    if independently_recomputed is not None:
        repeated = independently_recomputed.G.double()
        xr, ir = independently_recomputed.solve(direction, tolerance)
        result["independent_recomputation"] = {
            "row_relative_error": float((baseline - repeated).norm().item() / norm),
            "gram_relative_error": float((gram - repeated @ repeated.T).norm().item() / gram_norm),
            "solve_cosine": cosine(x32, xr), "solve": ir,
        }
    return result


def storage_audit_passed(audit: Mapping) -> bool:
    """Original storage thresholds, with repeated-backward noise separated."""
    if audit.get("comparison") != "one_fp32_row_realization_cast_to_bf16":
        return False
    passed = (audit["bf16_row_relative_error"] <= .01 and
              audit["bf16_gram_relative_error"] <= .02 and
              audit["fp32_bf16_solve_cosine"] >= .999 and
              audit["fp32_solve"]["converged"] and
              audit["bf16_solve"]["converged"])
    repeated = audit.get("independent_recomputation")
    if repeated is not None:
        passed = (passed and repeated["gram_relative_error"] <= .02 and
                  repeated["solve_cosine"] >= .999 and repeated["solve"]["converged"])
    return bool(passed)


def ggn_audit_passed(validation: Mapping) -> bool:
    """Recheck all GGN gates; never waive PSD, symmetry, or solver criteria."""
    return bool(
        validation["gram_symmetry_relative_error"] <= 1e-10 and
        validation["gram_min_eigenvalue"] >= -1e-8 and
        validation["damped_bilinear_symmetry_error"] <= 1e-4 and
        min(validation["damped_quadratic_forms"]) > 0 and
        all(info["converged"] for info in validation["solves"].values()) and
        storage_audit_passed(validation["small_real_checkpoint_storage_audit"]))


def longest_row_gradient_check(policy, tokenizer, rows, cfg, device="cuda:0") -> dict:
    """Exercise the maximum-length sampled strict margin before a lengthy gate."""
    import torch
    from ..scoring import _pair_margin
    from ..tokenization import conditional_length
    lengths = [max(conditional_length(tokenizer, row["prompt"], row[key])
                   for key in ("chosen", "rejected")) for row in rows]
    index = max(range(len(rows)), key=lengths.__getitem__)
    if lengths[index] > cfg.max_length:
        raise ValueError("strict scoring would truncate")
    torch.cuda.reset_peak_memory_stats()
    row = rows[index]
    margin = _pair_margin(policy, tokenizer, row["prompt"], row["chosen"],
                          row["rejected"], cfg.beta_dpo, device, cfg.max_length)
    gradients = torch.autograd.grad(margin, [p for p in policy.parameters() if p.requires_grad])
    finite = bool(torch.isfinite(margin)) and all(bool(torch.isfinite(g).all()) for g in gradients)
    if not finite:
        raise FloatingPointError("longest strict row has nonfinite margin/gradient")
    result = {"rows_audited": len(rows), "longest_tokens": lengths[index],
              "finite": finite, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
              "softmax_activation_checkpointing": True}
    del gradients, margin
    torch.cuda.empty_cache()
    return result
