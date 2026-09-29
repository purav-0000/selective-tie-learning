"""Fresh nonlinear DPO fitting with fixed, prerealized tie orientations."""
from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .config import ResamplingInfluenceConfig
from .dgp import LatentPairBatch, PreferenceBatch


class RewardMLP(nn.Module):
    def __init__(self, d_in: int, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values).squeeze(-1)


def configure_fp32() -> None:
    """Use the configured arithmetic for this campaign."""
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def make_model(cfg: ResamplingInfluenceConfig, seed: int) -> RewardMLP:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    return RewardMLP(cfg.input_dimension, cfg.width).to(cfg.device)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()}


def load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensor_digest(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def strict_losses(
    model: nn.Module, batch: PreferenceBatch, beta: float,
) -> torch.Tensor:
    margin = beta * (model(batch.phi_w) - model(batch.phi_l))
    return nn.functional.softplus(-margin)


def packet_losses(
    model: nn.Module, phi: torch.Tensor, phi_t: torch.Tensor, beta: float,
) -> torch.Tensor:
    margin = beta * (model(phi) - model(phi_t))
    return 0.5 * (
        nn.functional.softplus(-margin) + nn.functional.softplus(margin))


def realized_tie_losses(
    model: nn.Module, chosen: torch.Tensor, rejected: torch.Tensor, beta: float,
) -> torch.Tensor:
    margin = beta * (model(chosen) - model(rejected))
    return nn.functional.softplus(-margin)


def normalized_objective(
    model: nn.Module, strict: PreferenceBatch, tie_phi: torch.Tensor | None,
    tie_phi_t: torch.Tensor | None, beta: float,
) -> torch.Tensor:
    strict_sum = strict_losses(model, strict, beta).sum()
    k = 0 if tie_phi is None else int(tie_phi.shape[0])
    tie_sum = strict_sum.new_zeros(())
    if k:
        tie_sum = realized_tie_losses(model, tie_phi, tie_phi_t, beta).sum()
    return (strict_sum + tie_sum) / (len(strict.phi_w) + k)


def _backward_exact_objective(
    model: nn.Module, strict: PreferenceBatch, tie_phi: torch.Tensor | None,
    tie_phi_t: torch.Tensor | None, beta: float, chunk_size: int,
) -> torch.Tensor:
    """Backpropagate the exact full objective while bounding activation memory."""
    model.zero_grad(set_to_none=True)
    n = len(strict.phi_w)
    k = 0 if tie_phi is None else len(tie_phi)
    denominator = n + k
    value = 0.0
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        piece = strict_losses(model, PreferenceBatch(
            strict.phi_w[start:stop], strict.phi_l[start:stop],
            strict.q_w[start:stop], strict.q_l[start:stop]), beta).sum()
        (piece / denominator).backward()
        value += float(piece.detach())
    if k:
        for start in range(0, k, chunk_size):
            stop = min(start + chunk_size, k)
            piece = realized_tie_losses(
                model, tie_phi[start:stop], tie_phi_t[start:stop], beta).sum()
            (piece / denominator).backward()
            value += float(piece.detach())
    return torch.tensor(value / denominator, device=strict.phi_w.device)


def full_gradient_diagnostics(
    model: nn.Module, strict: PreferenceBatch, tie_phi: torch.Tensor | None,
    tie_phi_t: torch.Tensor | None, cfg: ResamplingInfluenceConfig,
) -> dict[str, float]:
    loss = _backward_exact_objective(
        model, strict, tie_phi, tie_phi_t, cfg.beta_dpo,
        cfg.objective_chunk_size)
    gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    l2 = math.sqrt(sum(float(g.double().square().sum()) for g in gradients))
    maximum = max((float(g.abs().max()) for g in gradients), default=0.0)
    model.zero_grad(set_to_none=True)
    return {"objective": float(loss), "gradient_l2": l2,
            "gradient_max": maximum}


def regularized_gradient_diagnostics(
    model: nn.Module, strict: PreferenceBatch, tie_phi: torch.Tensor | None,
    tie_phi_t: torch.Tensor | None, cfg: ResamplingInfluenceConfig,
) -> dict[str, float]:
    """Report the exact objective and gradient including configured L2."""
    base = full_gradient_diagnostics(model, strict, tie_phi, tie_phi_t, cfg)
    _backward_exact_objective(
        model, strict, tie_phi, tie_phi_t, cfg.beta_dpo,
        cfg.objective_chunk_size)
    penalty = 0.0
    squared_gradient = 0.0
    maximum = 0.0
    for parameter in model.parameters():
        penalty += 0.5 * cfg.adam_weight_decay * float(
            parameter.detach().double().square().sum())
        gradient = parameter.grad.detach() + cfg.adam_weight_decay * parameter.detach()
        squared_gradient += float(gradient.double().square().sum())
        maximum = max(maximum, float(gradient.abs().max()))
    model.zero_grad(set_to_none=True)
    return {
        **base,
        "regularized_objective": base["objective"] + penalty,
        "regularized_gradient_l2": math.sqrt(squared_gradient),
        "regularized_gradient_max": maximum,
        "l2_penalty": penalty,
    }


def _logical_batch_loss(
    model: nn.Module, strict: PreferenceBatch, tie_phi: torch.Tensor | None,
    tie_phi_t: torch.Tensor | None, logical_indices: torch.Tensor, beta: float,
) -> torch.Tensor:
    n = len(strict.phi_w)
    strict_idx = logical_indices[logical_indices < n]
    tie_idx = logical_indices[logical_indices >= n] - n
    total = logical_indices.numel()
    value = model.net[0].weight.sum() * 0.0
    if strict_idx.numel():
        value = value + strict_losses(model, PreferenceBatch(
            strict.phi_w[strict_idx], strict.phi_l[strict_idx],
            strict.q_w[strict_idx], strict.q_l[strict_idx]), beta).sum()
    if tie_idx.numel():
        value = value + realized_tie_losses(
            model, tie_phi[tie_idx], tie_phi_t[tie_idx], beta).sum()
    return value / total


def adam_fit(
    cfg: ResamplingInfluenceConfig, model: nn.Module, strict: PreferenceBatch,
    tie_phi: torch.Tensor | None = None,
    tie_phi_t: torch.Tensor | None = None, *, schedule_seed: int,
) -> dict:
    """Run common deterministic logical-row schedules for strict/fresh fits."""
    n = len(strict.phi_w)
    k = 0 if tie_phi is None else len(tie_phi)
    total = n + k
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.adam_lr,
        weight_decay=cfg.adam_weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(int(schedule_seed))
    history: list[float] = []
    started = time.perf_counter()
    model.train()
    for _epoch in range(cfg.epochs):
        permutation = torch.randperm(total, generator=generator)
        epoch_sum = 0.0
        for start in range(0, total, cfg.batch_size):
            idx = permutation[start:start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = _logical_batch_loss(
                model, strict, tie_phi, tie_phi_t, idx,
                cfg.beta_dpo)
            loss.backward()
            optimizer.step()
            epoch_sum += float(loss.detach()) * idx.numel()
        history.append(epoch_sum / total)
    model.eval()
    return {"loss_history": history,
            "runtime_seconds": time.perf_counter() - started}


def lbfgs_refine(
    cfg: ResamplingInfluenceConfig, model: nn.Module, strict: PreferenceBatch,
    tie_phi: torch.Tensor | None = None,
    tie_phi_t: torch.Tensor | None = None,
) -> dict:
    """Deterministic refinement of the exact L2-regularized objective."""
    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=cfg.lbfgs_lr,
        max_iter=cfg.lbfgs_max_iter, max_eval=cfg.lbfgs_max_eval,
        tolerance_grad=cfg.lbfgs_grad_tolerance,
        tolerance_change=cfg.lbfgs_change_tolerance,
        history_size=cfg.lbfgs_history, line_search_fn="strong_wolfe")
    evaluations = 0
    history: list[float] = []
    started = time.perf_counter()
    model.train()

    def closure() -> torch.Tensor:
        nonlocal evaluations
        evaluations += 1
        value = _backward_exact_objective(
            model, strict, tie_phi, tie_phi_t, cfg.beta_dpo,
            cfg.objective_chunk_size)
        penalty = sum(
            0.5 * cfg.adam_weight_decay * parameter.square().sum()
            for parameter in model.parameters())
        penalty.backward()
        total = value + penalty.detach()
        history.append(float(total))
        return total

    optimizer.step(closure)
    model.eval()
    diagnostics = regularized_gradient_diagnostics(
        model, strict, tie_phi, tie_phi_t, cfg)
    if diagnostics["regularized_gradient_max"] <= cfg.lbfgs_grad_tolerance:
        reason = "gradient_tolerance"
    elif evaluations >= cfg.lbfgs_max_eval:
        reason = "maximum_evaluations"
    elif len(history) >= 2 and abs(history[-1] - history[-2]) <= cfg.lbfgs_change_tolerance:
        reason = "change_tolerance"
    else:
        reason = "optimizer_terminated"
    return {
        **diagnostics, "evaluations": evaluations, "stop_reason": reason,
        "loss_history": history,
        "runtime_seconds": time.perf_counter() - started,
    }


def fit_from_initialization(
    cfg: ResamplingInfluenceConfig, initial_state: dict[str, torch.Tensor],
    strict: PreferenceBatch, tie_phi: torch.Tensor | None = None,
    tie_phi_t: torch.Tensor | None = None, *, schedule_seed: int,
) -> tuple[nn.Module, dict]:
    model = make_model(cfg, seed=0)
    load_state(model, initial_state)
    adam = adam_fit(
        cfg, model, strict, tie_phi, tie_phi_t,
        schedule_seed=schedule_seed)
    if cfg.optimizer == "adam_lbfgs_regularized":
        refinement = lbfgs_refine(cfg, model, strict, tie_phi, tie_phi_t)
        return model, {
            "optimizer": cfg.optimizer,
            "adam": adam,
            "lbfgs": refinement,
            "final": regularized_gradient_diagnostics(
                model, strict, tie_phi, tie_phi_t, cfg),
        }
    return model, {
        "optimizer": cfg.optimizer,
        "adam": adam,
        "final": regularized_gradient_diagnostics(
            model, strict, tie_phi, tie_phi_t, cfg),
    }


@torch.no_grad()
def expected_bt_accuracy(
    model: nn.Module, batch: LatentPairBatch, beta_teacher: float,
    chunk: int = 8192,
) -> float:
    total = 0.0
    for start in range(0, len(batch.phi_a), chunk):
        stop = min(start + chunk, len(batch.phi_a))
        chooses_a = model(batch.phi_a[start:stop]) >= model(batch.phi_b[start:stop])
        probability = torch.sigmoid(
            beta_teacher * (batch.q_a[start:stop] - batch.q_b[start:stop]))
        total += float(torch.where(chooses_a, probability, 1.0-probability).sum())
    return total / len(batch.phi_a)

@torch.no_grad()
def strict_fit_diagnostics(
    model: nn.Module, strict: PreferenceBatch, beta: float,
) -> dict[str, float]:
    margin = beta * (model(strict.phi_w)-model(strict.phi_l))
    probability = torch.sigmoid(margin)
    entropy = -(
        probability * probability.clamp_min(1e-12).log()
        + (1-probability) * (1-probability).clamp_min(1e-12).log())
    return {
        "training_accuracy": float((margin > 0).float().mean()),
        "prediction_entropy": float(entropy.mean()),
        "margin_mean": float(margin.mean()),
        "margin_std": float(margin.std()),
        "margin_abs_median": float(margin.abs().median()),
    }
