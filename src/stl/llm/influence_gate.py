"""Fixed-probe LLM GGN validation and symmetric candidate scoring."""
from __future__ import annotations

import gc
import time

from .real_data.ggn_validation import (
    ggn_audit_passed, longest_row_gradient_check, matched_storage_audit,
)


def validate_and_score(cfg, policy, tokenizer, strict_sample, candidates,
                       probe_sets, criteria=None, update=lambda _: None):
    """Validate one fixed probe/GGN system and score the shared panel once."""
    import torch
    from .scoring import LoRAGGN, _packet_gradient, _probe_direction
    from .tokenization import conditional_length

    if not cfg.symmetric_tie_packets:
        raise ValueError("fixed-probe scoring requires symmetric packets")
    expected = cfg.selection_probe_size
    if set(probe_sets) != {("selection", expected)}:
        raise ValueError(f"expected one fixed selection probe of size {expected}")
    probe_rows = probe_sets[("selection", expected)]

    start = time.monotonic()
    update("longest_strict_gradient")
    memory = longest_row_gradient_check(policy, tokenizer, strict_sample, cfg)
    lengths = [conditional_length(tokenizer, row["prompt"], row[key])
               for rows in (candidates, probe_rows) for row in rows
               for key in ("response_a", "response_b")]
    if max(lengths) > cfg.max_length:
        raise ValueError("candidate/probe scoring would truncate")
    update(f"selection_probe_{expected}")
    direction, probe_stats = _probe_direction(
        policy, tokenizer, probe_rows, cfg.beta_dpo, "cuda:0",
        cfg.max_length, batch_size=cfg.probe_batch_size)
    direction = direction.float()
    probe_seconds = time.monotonic() - start

    update("strict_GGN")
    start = time.monotonic()
    ggn = LoRAGGN(policy, tokenizer, strict_sample, cfg.beta_dpo, "cuda:0",
                  cfg.max_length, cfg.ggn_damping_fraction, torch.bfloat16)
    inverse, solve = ggn.solve(direction, cfg.ggn_solve_tol)
    inverse = inverse.float()
    if not solve["converged"] or solve["relative_residual"] > cfg.ggn_solve_tol:
        raise RuntimeError(f"GGN solve failed: {solve}")
    gram = ggn._gram64
    eig = torch.linalg.eigvalsh(gram) / ggn.N
    gen = torch.Generator(device="cuda").manual_seed(cfg.seed + 91001)
    x = torch.randn(direction.numel(), generator=gen, device="cuda")
    y = torch.randn(x.numel(), generator=gen, device="cuda")
    ax, ay = ggn(x), ggn(y)
    lhs, rhs = float(x.double() @ ay.double()), float(y.double() @ ax.double())
    small32 = LoRAGGN(policy, tokenizer, strict_sample[:16], cfg.beta_dpo,
                      "cuda:0", cfg.max_length, cfg.ggn_damping_fraction,
                      torch.float32)
    smallbf = LoRAGGN(policy, tokenizer, strict_sample[:16], cfg.beta_dpo,
                      "cuda:0", cfg.max_length, cfg.ggn_damping_fraction,
                      torch.bfloat16)
    validation = {
        "operator": "strict_DPO_generalized_gauss_newton",
        "rows": ggn.N,
        "storage_dtype": "bfloat16",
        "gram_symmetry_relative_error": float((gram-gram.T).abs().max()) /
                                        max(float(gram.abs().max()), 1e-30),
        "gram_min_eigenvalue": float(eig[0]),
        "gram_max_eigenvalue": float(eig[-1]),
        "damped_bilinear_symmetry_error": abs(lhs-rhs) /
                                           max(abs(lhs), abs(rhs), 1e-30),
        "damped_quadratic_forms": [float(x.double() @ ax.double()),
                                     float(y.double() @ ay.double())],
        "solves": {"selection": solve},
        "small_real_checkpoint_storage_audit": matched_storage_audit(
            small32, direction, tolerance=cfg.ggn_solve_tol,
            independently_recomputed=smallbf),
    }
    validation["passed"] = ggn_audit_passed(validation)
    ggn_seconds = time.monotonic() - start
    del ggn, small32, smallbf, gram, eig, x, y, ax, ay
    gc.collect(); torch.cuda.empty_cache()

    start = time.monotonic()
    scores, margins = [], []
    for i, candidate in enumerate(candidates):
        if i % 100 == 0:
            update(f"symmetric_candidate_gradients_{i}_of_{len(candidates)}")
        gradient, margin = _packet_gradient(
            policy, tokenizer, candidate, cfg.beta_dpo, "cuda:0",
            cfg.max_length, symmetric_packet=True)
        gradient = gradient.float() / cfg.N
        margins.append(margin)
        scores.append(float(inverse.double() @ gradient.double()))
    candidate_seconds = time.monotonic() - start
    if not torch.isfinite(torch.tensor(scores)).all():
        raise FloatingPointError("nonfinite IF scores")
    scored = [
        {**candidate, "pm_score": margins[i], "ha_score": scores[i]}
        for i, candidate in enumerate(candidates)
    ]
    result = {
        "gradient_source": "exact_symmetric_packet_gradient",
        "packet_source": "grad(packet_loss)/N",
        "candidate_rows": len(candidates),
        "selection_probe_size": expected,
        "probe_stats": probe_stats,
        "probe_size_was_fixed_before_campaign": True,
        "ggn_validation": validation,
        "campaign_gate_passed": bool(validation["passed"]),
        "longest_strict_row": memory,
        "maximum_scoring_tokens": max(lengths),
        "seconds": {
            "probe": probe_seconds,
            "ggn_and_solve": ggn_seconds,
            "shared_candidate_backward_and_projection": candidate_seconds,
        },
    }
    return scored, result
