"""Exact optimization for the log-linear campaign."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective import normalized_packet_dpo


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


@dataclass(frozen=True)
class ExactSolve:
    theta: np.ndarray
    objective: float
    strict_loss: float
    gradient_norm: float
    iterations: int
    backtracks: int
    hessian_eigenvalues: np.ndarray
    hessian_condition_number: float
    converged: bool


def solve_exact_packet_dpo(
    strict_deltas: np.ndarray,
    pool_spurious: np.ndarray,
    selected: np.ndarray,
    beta: float,
    d_c: int,
    *,
    a_wins: np.ndarray,
    symmetric: bool = False,
    starting_theta: np.ndarray | None = None,
    tolerance: float = 1e-10,
    maximum_iterations: int = 250,
) -> ExactSolve:
    """Solve a fresh empirical strict-plus-packet objective by dense Newton."""
    strict_deltas = np.asarray(strict_deltas, dtype=np.float64)
    pool_spurious = np.asarray(pool_spurious, dtype=np.float64)
    selected = np.asarray(selected, dtype=np.int64)
    if strict_deltas.ndim != 2 or len(strict_deltas) == 0:
        raise ValueError("strict_deltas must be a nonempty matrix")
    if pool_spurious.ndim != 2 or pool_spurious.shape[1] != (
        strict_deltas.shape[1] - d_c
    ):
        raise ValueError("pool_spurious has the wrong shape")
    if np.any(selected < 0) or np.any(selected >= len(pool_spurious)):
        raise IndexError("selected packet index is out of range")
    a_wins = np.asarray(a_wins, dtype=np.bool_)
    if a_wins.shape != (len(pool_spurious),):
        raise ValueError("realized orientation must cover every candidate")

    theta = (np.zeros(strict_deltas.shape[1], dtype=np.float64)
             if starting_theta is None else np.asarray(starting_theta, dtype=np.float64).copy())
    backtracks = 0
    converged = False
    iteration = 0
    for iteration in range(maximum_iterations + 1):
        value, gradient, hessian, strict_loss, _ = normalized_packet_dpo(
            theta, strict_deltas, pool_spurious, selected, beta, d_c,
            a_wins, symmetric=symmetric)
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= tolerance:
            converged = True
            break
        if iteration == maximum_iterations:
            break

        hessian = _symmetric(hessian)
        factor = np.linalg.cholesky(hessian)
        step = np.linalg.solve(factor.T, np.linalg.solve(factor, gradient))
        directional = float(gradient @ step)
        scale = 1.0
        accepted = False
        for _ in range(60):
            candidate = theta - scale * step
            candidate_value, candidate_gradient, _, _, _ = normalized_packet_dpo(
                candidate, strict_deltas, pool_spurious, selected, beta, d_c,
                a_wins, symmetric=symmetric)
            roundoff = 64 * np.finfo(np.float64).eps * max(1.0, abs(value))
            armijo = candidate_value <= value - 1e-4 * scale * directional
            residual_progress = (
                gradient_norm <= 1e-6
                and candidate_value <= value + roundoff
                and np.linalg.norm(candidate_gradient) < gradient_norm
            )
            if np.isfinite(candidate_value) and (armijo or residual_progress):
                theta = candidate
                accepted = True
                break
            scale *= 0.5
            backtracks += 1
        if not accepted:
            raise RuntimeError("exact full-batch Newton line search failed")

    value, gradient, hessian, strict_loss, _ = normalized_packet_dpo(
        theta, strict_deltas, pool_spurious, selected, beta, d_c,
        a_wins, symmetric=symmetric)
    eigenvalues = np.linalg.eigvalsh(_symmetric(hessian))
    gradient_norm = float(np.linalg.norm(gradient))
    return ExactSolve(
        theta=theta,
        objective=float(value),
        strict_loss=float(strict_loss),
        gradient_norm=gradient_norm,
        iterations=int(iteration),
        backtracks=int(backtracks),
        hessian_eigenvalues=eigenvalues,
        hessian_condition_number=float(eigenvalues[-1] / eigenvalues[0]),
        converged=bool(converged or gradient_norm <= tolerance),
    )
