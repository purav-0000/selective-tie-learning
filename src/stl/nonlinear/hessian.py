"""Explicit GGN rows, Woodbury solves, and packet derivatives."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, jvp, vmap

from .dgp import PreferenceBatch


def parameter_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach() for name, value in model.named_parameters()}


def parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _value in model.named_parameters()]


def flatten(values: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.cat([values[name].reshape(-1) for name in names])


def unflatten(
    vector: torch.Tensor, names: list[str], reference: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    cursor = 0
    for name in names:
        width = reference[name].numel()
        output[name] = vector[cursor:cursor+width].reshape(reference[name].shape)
        cursor += width
    if cursor != vector.numel():
        raise ValueError("flat parameter vector has the wrong size")
    return output


def generalized_gauss_newton_rows(
    model: nn.Module, strict: PreferenceBatch, beta: float, chunk_size: int,
) -> torch.Tensor:
    """Materialize ``sqrt(ell''(m_i)) grad(m_i)`` in strict-row order."""
    params = parameter_dict(model)
    names = parameter_names(model)

    def single_margin(
        parameters: dict[str, torch.Tensor], phi_w: torch.Tensor,
        phi_l: torch.Tensor,
    ) -> torch.Tensor:
        return beta * (
            functional_call(model, parameters, (phi_w.unsqueeze(0),)).squeeze()
            - functional_call(
                model, parameters, (phi_l.unsqueeze(0),)).squeeze())

    margin_gradient = grad(single_margin)
    rows: list[torch.Tensor] = []
    for start in range(0, len(strict.phi_w), chunk_size):
        stop = min(start + chunk_size, len(strict.phi_w))
        phi_w = strict.phi_w[start:stop]
        phi_l = strict.phi_l[start:stop]
        with torch.no_grad():
            margin = beta * (model(phi_w) - model(phi_l))
            weight = torch.sigmoid(margin) * torch.sigmoid(-margin)
        jacobian = vmap(
            margin_gradient, in_dims=(None, 0, 0), randomness="error",
        )(params, phi_w, phi_l)
        flat = torch.cat([
            jacobian[name].reshape(stop - start, -1) for name in names
        ], dim=1)
        rows.append(weight.sqrt().unsqueeze(1) * flat)
    result = torch.cat(rows).detach()
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("GGN rows contain nonfinite values")
    return result


class GGNWoodburySolver:
    """Reusable exact row-space solve for a damped empirical DPO GGN."""

    def __init__(
        self, rows: torch.Tensor, *, ridge: float, damping_fraction: float,
    ) -> None:
        if rows.ndim != 2 or not len(rows):
            raise ValueError("GGN rows must be a nonempty matrix")
        if ridge < 0.0 or damping_fraction <= 0.0:
            raise ValueError("ridge must be nonnegative and damping positive")
        self.rows = rows
        self.n, self.dimension = rows.shape
        self.ridge = float(ridge)
        self.gram = rows @ rows.T
        self.eigenvalues = torch.linalg.eigvalsh(self.gram) / self.n
        self.lambda_max = max(float(self.eigenvalues[-1]), 1e-12)
        # The MLP objective has a fixed L2 term, so its curvature scale includes
        # that ridge. This reduces to the LLM rule when ridge is zero.
        self.spectral_scale = self.lambda_max + self.ridge
        self.damping = float(damping_fraction) * self.spectral_scale
        self.regularization = self.ridge + self.damping
        system = self.gram + (self.n * self.regularization) * torch.eye(
            self.n, device=rows.device, dtype=rows.dtype)
        self.factor = torch.linalg.cholesky(system)

    def solve(self, rhs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Solve ``(G.T G/N + (ridge+damping)I) x = rhs`` exactly."""
        if rhs.numel() != self.dimension:
            raise ValueError("GGN rows and right-hand side are incompatible")
        projected_rhs = self.rows @ rhs
        coefficients = torch.cholesky_solve(
            projected_rhs.unsqueeze(1), self.factor).squeeze(1)
        solution = (
            rhs - self.rows.T @ coefficients) / self.regularization
        residual = self.rows.T @ (self.rows @ solution) / self.n
        residual.add_(solution, alpha=self.regularization).sub_(rhs)
        relative_residual = float(
            residual.norm() / (rhs.norm() + 1e-30))
        return solution.detach(), {
            "solver": "woodbury_cholesky",
            "row_count": int(self.n),
            "parameter_count": int(self.dimension),
            "ridge": self.ridge,
            "damping": self.damping,
            "damping_fraction_of_spectral_scale": (
                self.damping / self.spectral_scale),
            "regularization": self.regularization,
            "relative_residual": relative_residual,
        }

    def diagnostics(self) -> dict:
        threshold = max(self.lambda_max * 1e-10, 1e-14)
        positive = self.eigenvalues[self.eigenvalues > threshold]
        norms = self.rows.norm(dim=1)
        return {
            "operator": "generalized_gauss_newton",
            "solver": "woodbury_cholesky",
            "row_count": int(self.n),
            "parameter_count": int(self.dimension),
            "arithmetic": str(self.rows.dtype).replace("torch.", ""),
            "lambda_max": self.lambda_max,
            "spectral_scale": self.spectral_scale,
            "effective_rank": int(positive.numel()),
            "min_positive_eigenvalue": (
                float(positive[0]) if positive.numel() else 0.0),
            "ridge": self.ridge,
            "damping": self.damping,
            "damped_condition_estimate": float(
                (self.lambda_max + self.regularization)
                / self.regularization),
            "row_norm_min": float(norms.min()),
            "row_norm_median": float(norms.median()),
            "row_norm_max": float(norms.max()),
        }


def packet_directional_derivatives(
    model: nn.Module, phi: torch.Tensor, phi_t: torch.Tensor, beta: float,
    direction: torch.Tensor, *, chunk_size: int,
) -> torch.Tensor:
    """Compute all packet-gradient inner products without storing gradients."""
    params, names = parameter_dict(model), parameter_names(model)
    tangent = unflatten(direction, names, params)
    outputs: list[torch.Tensor] = []
    for start in range(0, len(phi), chunk_size):
        stop = min(start+chunk_size, len(phi))

        def losses(parameters: dict[str, torch.Tensor]) -> torch.Tensor:
            margin = beta * (
                functional_call(model, parameters, (phi[start:stop],))
                - functional_call(model, parameters, (phi_t[start:stop],)))
            return 0.5 * (
                nn.functional.softplus(-margin)
                + nn.functional.softplus(margin))

        _, derivative = jvp(losses, (params,), (tangent,))
        outputs.append(derivative.detach())
    return torch.cat(outputs)


def signed_ranking(scores: torch.Tensor | np.ndarray) -> np.ndarray:
    values = scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores
    if not np.isfinite(values).all():
        raise FloatingPointError("selector scores contain nonfinite values")
    return np.argsort(-values, kind="stable").astype(np.int64)
