"""Held-out counterfactual-tie energy."""
from __future__ import annotations

import math

import torch
from torch import nn

from .data import ProbeBank


def _rewards(
    model: nn.Module, rows: torch.Tensor, beta: float, chunk: int,
) -> torch.Tensor:
    values = []
    for start in range(0, len(rows), chunk):
        values.append((beta * model(rows[start:start + chunk])).detach().cpu().double())
    return torch.cat(values)


def _margin_summary(margins: torch.Tensor) -> dict[str, float]:
    flat = margins.reshape(-1)
    quantiles = torch.quantile(
        flat, torch.tensor([.05, .25, .5, .75, .95], dtype=flat.dtype))
    return {
        "margin_mean": float(flat.mean()),
        "margin_standard_deviation": float(flat.std(unbiased=True)),
        "margin_mean_absolute": float(flat.abs().mean()),
        "margin_root_mean_square": float(flat.square().mean().sqrt()),
        "margin_positive_fraction": float((flat > 0).double().mean()),
        "margin_minimum": float(flat.min()),
        "margin_q05": float(quantiles[0]),
        "margin_q25": float(quantiles[1]),
        "margin_median": float(quantiles[2]),
        "margin_q75": float(quantiles[3]),
        "margin_q95": float(quantiles[4]),
        "margin_maximum": float(flat.max()),
    }


@torch.no_grad()
def pair_energy(
    model: nn.Module, first: torch.Tensor, second: torch.Tensor,
    beta: float, chunk: int = 4096,
) -> float:
    """Mean squared reference-adjusted margin for an explicit pair set."""
    if len(first) != len(second) or not len(first):
        raise ValueError("pair energy requires aligned nonempty tensors")
    first_rewards = _rewards(model, first, beta, chunk)
    second_rewards = _rewards(model, second, beta, chunk)
    return float((first_rewards - second_rewards).square().mean())


@torch.no_grad()
def counterfactual_energy(
    model: nn.Module, probe: ProbeBank, beta: float, chunk: int = 2048,
) -> dict[str, float | int | str]:
    """Evaluate E_cf on held-out winner-preserving counterfactual ties."""
    anchors, redraws, features = probe.resampling_redraws.shape
    if anchors < 1 or redraws < 1:
        raise ValueError("counterfactual energy probe must be nonempty")
    original = _rewards(model, probe.resampling_first, beta, chunk)
    transformed = _rewards(
        model, probe.resampling_redraws.reshape(anchors * redraws, features),
        beta, chunk).reshape(anchors, redraws)
    margins = original[:, None] - transformed
    energy = float(margins.square().mean())
    if not math.isfinite(energy) or energy < 0.0:
        raise ValueError("counterfactual energy is invalid")
    return {
        "estimator": "mean_squared_reference_adjusted_heldout_tie_margin",
        "definition": "mean((beta*(r(y_w)-r(T(y_w))))^2)",
        "anchors": anchors,
        "redraws_per_anchor": redraws,
        "heldout_ties": anchors * redraws,
        "counterfactual_energy": energy,
        **_margin_summary(margins),
    }
