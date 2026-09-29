"""Canonical empirical logistic-DPO and symmetric-packet objective."""
from __future__ import annotations

import numpy as np


def empirical_dpo(theta: np.ndarray, deltas: np.ndarray, beta: float):
    """Return mean logistic loss, gradient, and empirical Hessian."""
    z = beta * (deltas @ theta)
    loss = float(np.logaddexp(0.0, -z).mean())
    probability_negative = np.exp(-np.logaddexp(0.0, z))
    gradient = -beta * (
        probability_negative[:, None] * deltas).mean(axis=0)
    weight = probability_negative * (1.0 - probability_negative)
    hessian = (
        beta**2 * (deltas * weight[:, None]).T @ deltas / len(deltas))
    return loss, gradient, hessian


def symmetric_packet_terms(
    theta: np.ndarray, rows: np.ndarray, beta: float, d_c: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean loss, gradient, and Hessian for exact symmetric tie packets."""
    rows = np.asarray(rows, dtype=np.float64)
    gradient = np.zeros_like(theta, dtype=np.float64)
    hessian = np.zeros((len(theta), len(theta)), dtype=np.float64)
    if not len(rows):
        return 0.0, gradient, hessian
    margin = beta * (rows @ theta[d_c:])
    loss = float(0.5 * (
        np.logaddexp(0.0, -margin) + np.logaddexp(0.0, margin)).mean())
    coefficient = beta / 2.0 * np.tanh(margin / 2.0)
    gradient[d_c:] = (coefficient[:, None] * rows).mean(axis=0)
    weight = beta**2 / 4.0 * (1.0 - np.tanh(margin / 2.0) ** 2)
    hessian[d_c:, d_c:] = (rows * weight[:, None]).T @ rows / len(rows)
    return loss, gradient, hessian


def symmetric_packet_gradients(
    theta: np.ndarray, rows: np.ndarray, beta: float, d_c: int,
) -> np.ndarray:
    """Return one exact symmetric-packet gradient per spurious row."""
    rows = np.asarray(rows, dtype=np.float64)
    margin = beta * (rows @ theta[d_c:])
    coefficient = beta / 2.0 * np.tanh(margin / 2.0)
    output = np.zeros((len(rows), len(theta)), dtype=np.float64)
    output[:, d_c:] = coefficient[:, None] * rows
    return output


def realized_tie_terms(
    theta: np.ndarray, oriented_rows: np.ndarray, beta: float, d_c: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean logistic loss and derivatives for fixed sampled tie labels."""
    rows = np.asarray(oriented_rows, dtype=np.float64)
    gradient = np.zeros_like(theta, dtype=np.float64)
    hessian = np.zeros((len(theta), len(theta)), dtype=np.float64)
    if not len(rows):
        return 0.0, gradient, hessian
    margin = beta * (rows @ theta[d_c:])
    loss = float(np.logaddexp(0.0, -margin).mean())
    negative = np.exp(-np.logaddexp(0.0, margin))
    gradient[d_c:] = -beta * (negative[:, None] * rows).mean(axis=0)
    weight = beta**2 * negative * (1.0 - negative)
    hessian[d_c:, d_c:] = (rows * weight[:, None]).T @ rows / len(rows)
    return loss, gradient, hessian


def normalized_packet_dpo(
    theta: np.ndarray,
    strict_deltas: np.ndarray,
    pool_spurious: np.ndarray,
    selected: np.ndarray,
    beta: float,
    d_c: int,
    a_wins: np.ndarray,
    *, symmetric: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Exact strict-plus-selected-packet objective, realized or symmetric."""
    selected = np.asarray(selected, dtype=np.int64)
    strict_loss, strict_gradient, strict_hessian = empirical_dpo(
        theta, strict_deltas, beta)
    if symmetric:
        tie_loss, tie_gradient, tie_hessian = symmetric_packet_terms(
            theta, pool_spurious[selected], beta, d_c)
    else:
        oriented = pool_spurious[selected] * np.where(
            a_wins[selected], 1.0, -1.0)[:, None]
        tie_loss, tie_gradient, tie_hessian = realized_tie_terms(
            theta, oriented, beta, d_c)
    n, size = len(strict_deltas), len(selected)
    denominator = n + size
    value = (n * strict_loss + size * tie_loss) / denominator
    gradient = (n * strict_gradient + size * tie_gradient) / denominator
    hessian = (n * strict_hessian + size * tie_hessian) / denominator
    return float(value), gradient, hessian, strict_loss, tie_loss
