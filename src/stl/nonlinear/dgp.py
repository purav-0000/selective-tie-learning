"""Generic nonlinear shortcut DGP with invariant feature marginals.

The DGP generates individual alternatives before forming reward differences.
The environment parameters independently control direct utility-aligned
spurious signal (``a``) and residual causal--spurious covariance (``t``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


ENVIRONMENTS = {
    "P": (1.0, 1.0),
    "Q_clean": (0.0, 0.0),
    "Q_direct": (-1.0, 1.0),
    "Q_cov": (1.0, -1.0),
    "Q_mixed": (-1.0, -1.0),
}


@dataclass(frozen=True)
class PreferenceBatch:
    """Oriented preferred/rejected features and latent utilities."""
    phi_w: torch.Tensor
    phi_l: torch.Tensor
    q_w: torch.Tensor
    q_l: torch.Tensor


@dataclass(frozen=True)
class LatentPairBatch:
    """Unoriented evaluation pairs and their latent utilities."""
    phi_a: torch.Tensor
    phi_b: torch.Tensor
    q_a: torch.Tensor
    q_b: torch.Tensor


class FrozenMixer:
    """Paper mixer ``20 -> 64 -> 20`` with deterministic frozen weights."""

    def __init__(
        self, dimension: int, hidden: int, seed: int, device: torch.device,
    ) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.W1 = (
            torch.randn(dimension, hidden, generator=generator)
            / np.sqrt(dimension)).to(device)
        self.b1 = (torch.randn(hidden, generator=generator) * 0.1).to(device)
        self.W2 = (
            torch.randn(hidden, dimension, generator=generator)
            / np.sqrt(hidden)).to(device)
        self.b2 = (
            torch.randn(dimension, generator=generator) * 0.1).to(device)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(values @ self.W1 + self.b1)
        return torch.tanh(hidden @ self.W2 + self.b2)


def _orthogonal(rng: np.random.Generator, dimension: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def _generic_spd(
    rng: np.random.Generator, dimension: int, condition_number: float,
) -> tuple[np.ndarray, np.ndarray]:
    interior = rng.uniform(0.0, np.log(condition_number), dimension - 2)
    eigenvalues = np.exp(np.r_[0.0, interior, np.log(condition_number)])
    rng.shuffle(eigenvalues)
    eigenvalues /= eigenvalues.mean()
    basis = _orthogonal(rng, dimension)
    matrix = (basis * eigenvalues) @ basis.T
    return 0.5 * (matrix + matrix.T), basis


def _symmetric_root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    if values[0] <= 0.0:
        raise np.linalg.LinAlgError("matrix is not positive definite")
    return (vectors * np.sqrt(values)) @ vectors.T


def _unit(rng: np.random.Generator, dimension: int) -> np.ndarray:
    value = rng.normal(size=dimension)
    return value / np.linalg.norm(value)


class GenericNonlinearDGP:
    """One immutable generic geometry with environment-specific samplers."""

    def __init__(
        self, *, seed: int, d_c: int = 10, d_s: int = 10,
        block_condition_number: float = 20.0,
        r_min: float = 0.20, r_max: float = 0.65,
        gamma_c: float = 1.0, gamma_s: float = 1.0,
        beta_teacher: float = 2.0, mixer_hidden: int = 64,
        mixer_seed: int = 12345, device: str | torch.device = "cpu",
    ) -> None:
        if d_c != d_s:
            raise ValueError("the full-rank construction requires d_c=d_s")
        if not 0.0 < r_min <= r_max < 1.0:
            raise ValueError("R singular values must lie in (0,1)")
        self.seed = int(seed)
        self.d_c, self.d_s = int(d_c), int(d_s)
        self.dimension = self.d_c + self.d_s
        self.beta_teacher = float(beta_teacher)
        self.device = torch.device(device)
        children = np.random.SeedSequence(self.seed).spawn(6)
        rng_a, rng_b, rng_r, rng_uc, rng_us, _ = (
            np.random.default_rng(child) for child in children)
        self.A, self.basis_a = _generic_spd(
            rng_a, self.d_c, block_condition_number)
        self.B, self.basis_b = _generic_spd(
            rng_b, self.d_s, block_condition_number)
        self.root_a = _symmetric_root(self.A)
        self.root_b = _symmetric_root(self.B)
        self.basis_r_left = _orthogonal(rng_r, self.d_c)
        self.basis_r_right = _orthogonal(rng_r, self.d_s)
        self.r_singular_values = np.linspace(
            r_min, r_max, min(self.d_c, self.d_s))
        self.R = (
            (self.basis_r_left * self.r_singular_values)
            @ self.basis_r_right.T)
        self.residual_cross = self.root_a @ self.R @ self.root_b
        self.u_c = gamma_c * self.root_a @ _unit(rng_uc, self.d_c)
        self.u_s = gamma_s * self.root_b @ _unit(rng_us, self.d_s)
        self.mu_c = np.zeros(self.d_c, dtype=np.float64)
        self.gamma_c, self.gamma_s = float(gamma_c), float(gamma_s)
        self.mixer = FrozenMixer(
            self.dimension, int(mixer_hidden), int(mixer_seed), self.device)
        self._residual_cholesky = {}
        for environment, (_a, t) in ENVIRONMENTS.items():
            covariance = self.residual_covariance(t)
            values = np.linalg.eigvalsh(covariance)
            if values[0] <= 0.0 or not np.isfinite(values).all():
                raise np.linalg.LinAlgError(
                    f"residual covariance is not SPD for {environment}")
            self._residual_cholesky[environment] = torch.as_tensor(
                np.linalg.cholesky(covariance), dtype=torch.float32)

    def residual_covariance(self, t: float) -> np.ndarray:
        covariance = np.block([
            [self.A, t * self.residual_cross],
            [t * self.residual_cross.T, self.B],
        ])
        return 0.5 * (covariance + covariance.T)

    def option_moments(self, environment: str) -> dict[str, np.ndarray]:
        a, t = ENVIRONMENTS[environment]
        return {
            "mean_c": self.mu_c.copy(),
            "mean_s": np.zeros(self.d_s),
            "cov_c": self.A + np.outer(self.u_c, self.u_c),
            "cov_s": self.B + np.outer(self.u_s, self.u_s),
            "cov_cs": (
                a * np.outer(self.u_c, self.u_s)
                + t * self.residual_cross),
        }

    def _features(self, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.mixer(torch.cat((c, s), dim=-1).to(self.device))

    def _sample_option(
        self, n: int, environment: str, generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if environment not in ENVIRONMENTS:
            raise ValueError(f"unknown environment: {environment}")
        a, _t = ENVIRONMENTS[environment]
        q = torch.randn(n, 1, generator=generator)
        z = torch.randn(n, 1, generator=generator)
        g = a * q + np.sqrt(max(0.0, 1.0 - a * a)) * z
        standard = torch.randn(n, self.dimension, generator=generator)
        residual = standard @ self._residual_cholesky[environment].T
        eps_c, eps_s = residual[:, :self.d_c], residual[:, self.d_c:]
        c = torch.as_tensor(self.mu_c, dtype=torch.float32) + (
            q * torch.as_tensor(self.u_c, dtype=torch.float32)) + eps_c
        s = g * torch.as_tensor(self.u_s, dtype=torch.float32) + eps_s
        return q.squeeze(1), c, s

    def sample_preferences(
        self, n: int, *, environment: str = "P", seed: int,
    ) -> PreferenceBatch:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        q_a, c_a, s_a = self._sample_option(n, environment, generator)
        q_b, c_b, s_b = self._sample_option(n, environment, generator)
        phi_a, phi_b = self._features(c_a, s_a), self._features(c_b, s_b)
        probability = torch.sigmoid(self.beta_teacher * (q_a - q_b))
        a_wins = torch.rand(n, generator=generator) < probability
        mask = a_wins[:, None].to(self.device)
        return PreferenceBatch(
            phi_w=torch.where(mask, phi_a, phi_b),
            phi_l=torch.where(mask, phi_b, phi_a),
            q_w=torch.where(a_wins, q_a, q_b).to(self.device),
            q_l=torch.where(a_wins, q_b, q_a).to(self.device),
        )

    def sample_latent_pairs(
        self, n: int, *, environment: str, seed: int,
    ) -> LatentPairBatch:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        q_a, c_a, s_a = self._sample_option(n, environment, generator)
        q_b, c_b, s_b = self._sample_option(n, environment, generator)
        return LatentPairBatch(
            self._features(c_a, s_a), self._features(c_b, s_b),
            q_a.to(self.device), q_b.to(self.device))

    def geometry_diagnostics(self) -> dict:
        return {
            "eigenvalues_A": np.linalg.eigvalsh(self.A),
            "eigenvalues_B": np.linalg.eigvalsh(self.B),
            "condition_A": float(np.linalg.cond(self.A)),
            "condition_B": float(np.linalg.cond(self.B)),
            "trace_A_per_dimension": float(np.trace(self.A) / self.d_c),
            "trace_B_per_dimension": float(np.trace(self.B) / self.d_s),
            "R_singular_values": np.linalg.svd(self.R, compute_uv=False),
            "R_rank": int(np.linalg.matrix_rank(self.R)),
            "u_c_mahalanobis_squared": float(
                self.u_c @ np.linalg.solve(self.A, self.u_c)),
            "u_s_mahalanobis_squared": float(
                self.u_s @ np.linalg.solve(self.B, self.u_s)),
            "minimum_residual_covariance_eigenvalue": float(min(
                np.linalg.eigvalsh(self.residual_covariance(t))[0]
                for _a, t in ENVIRONMENTS.values())),
        }
