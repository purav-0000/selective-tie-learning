"""Matched PM and practical-influence scores for the nonlinear campaign."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad

from .config import ResamplingInfluenceConfig
from .data import PacketPool, ProbeBank
from .dgp import PreferenceBatch
from .hessian import (
    GGNWoodburySolver, flatten, generalized_gauss_newton_rows,
    packet_directional_derivatives, parameter_dict, parameter_names,
    signed_ranking,
)


def pair_energy_gradient(
    model: nn.Module, first: torch.Tensor, second: torch.Tensor, beta: float,
) -> torch.Tensor:
    """Differentiate held-out mean squared counterfactual margins."""
    params, names = parameter_dict(model), parameter_names(model)

    def objective(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        margin = beta * (
            functional_call(model, parameters, (first,))
            - functional_call(model, parameters, (second,)))
        return margin.square().mean()

    return flatten(grad(objective)(params), names).detach()


def direct_packet_scores(
    model: nn.Module, pool: PacketPool, beta: float,
    direction: torch.Tensor, n_strict: int, *, chunk_size: int,
) -> torch.Tensor:
    """Use ``grad(packet)/N`` literally; no checkpoint-gradient offset."""
    return packet_directional_derivatives(
        model, pool.first, pool.second, beta, direction,
        chunk_size=chunk_size) / n_strict


@torch.no_grad()
def pm_scores(model: nn.Module, pool: PacketPool, beta: float) -> torch.Tensor:
    """Return absolute implicit-reward margins for all candidates."""
    return (beta * (model(pool.first) - model(pool.second))).abs()


@dataclass
class MatchedScoreResult:
    """Selector scores, rankings, and curvature diagnostics."""
    scores: dict[str, dict[str, np.ndarray]]
    rankings: dict[str, dict[str, np.ndarray]]
    diagnostics: dict


def _probe_pair(
    probe: ProbeBank, operator: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if operator == "resampling":
        return probe.resampling_first, probe.resampling_second
    raise ValueError(f"unknown operator: {operator}")


def compute_matched_scores(
    cfg: ResamplingInfluenceConfig, model: nn.Module, strict: PreferenceBatch,
    pools: dict[str, PacketPool], selection: ProbeBank,
    *, seed: int,
) -> MatchedScoreResult:
    """Score exact symmetric packets once at the strict checkpoint."""
    del seed  # all stochastic subsampling was prerealized in the data bundle
    directions: dict[str, torch.Tensor] = {}
    for operator in cfg.operators:
        directions[operator] = pair_energy_gradient(
            model, *_probe_pair(selection, operator), cfg.beta_dpo)

    scores = {operator: {} for operator in cfg.operators}
    for operator in cfg.operators:
        if "PM" in cfg.methods:
            pm = pm_scores(model, pools[operator], cfg.beta_dpo)
            scores[operator]["PM"] = pm.cpu().numpy()

    curvature_diagnostics = None
    if "Influence" in cfg.methods:
        rows = generalized_gauss_newton_rows(
            model, strict, cfg.beta_dpo, cfg.curvature_chunk_size)
        ridge = cfg.adam_weight_decay
        solver = GGNWoodburySolver(
            rows, ridge=ridge,
            damping_fraction=cfg.spectral_shift_fraction)
        solutions: dict[str, torch.Tensor] = {}
        solves: dict[str, dict] = {}
        for name, rhs in directions.items():
            solution, diagnostic = solver.solve(rhs)
            if diagnostic["relative_residual"] > cfg.ggn_solve_tolerance:
                raise RuntimeError(
                    f"Woodbury GGN residual failed for {name}: "
                    f"{diagnostic['relative_residual']:.3g}")
            solutions[name] = solution
            solves[name] = diagnostic

        for operator in cfg.operators:
            pool = pools[operator]
            scores[operator]["Influence"] = direct_packet_scores(
                model, pool, cfg.beta_dpo,
                solutions[operator], cfg.n_strict,
                chunk_size=cfg.candidate_chunk_size).cpu().numpy()
        curvature_diagnostics = {
            **solver.diagnostics(),
            "objective_l2_hessian": ridge,
            "systems": solves,
        }

    rankings = {
        operator: {
            name: signed_ranking(value)
            for name, value in scores[operator].items()
        }
        for operator in cfg.operators
    }
    diagnostics = {
        "curvature": curvature_diagnostics,
        "packet_source": "grad_L_packet_over_N",
        "probe_gradient_norms": {
            name: float(value.norm()) for name, value in directions.items()},
    }
    return MatchedScoreResult(scores, rankings, diagnostics)
