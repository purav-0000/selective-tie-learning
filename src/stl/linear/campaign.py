"""Log-linear reward learning on the nonlinear campaign's latent BT DGP.

Alternatives are generated individually from the generic nonlinear latent DGP,
but the frozen mixer is omitted and the reward is linear in ``[c; s]``.  This
module implements winner-preserving empirical-redraw packets, exact Oracle
influence, exact empirical DPO solves, and population expected-BT evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr

from stl.nonlinear.dgp import ENVIRONMENTS, GenericNonlinearDGP

from .optimization import ExactSolve, solve_exact_packet_dpo
from .objective import empirical_dpo, symmetric_packet_gradients
from .population_metrics import (
    p_pair_block_second_moments, population_expected_bt_accuracy,
    trained_component_metrics)


METHODS = ("UTL", "PM", "Oracle")
OPERATORS = ("resampling",)


def _symmetric(value: np.ndarray) -> np.ndarray:
    return 0.5 * (value + value.T)


@dataclass(frozen=True)
class PreferenceLatents:
    """Winner/loser causal, nuisance, and latent-utility arrays."""
    c_w: np.ndarray
    s_w: np.ndarray
    q_w: np.ndarray
    c_l: np.ndarray
    s_l: np.ndarray
    q_l: np.ndarray

    @property
    def nuisance_bank(self) -> np.ndarray:
        return np.concatenate((self.s_w, self.s_l), axis=0)

    @property
    def strict_deltas(self) -> np.ndarray:
        return np.concatenate((self.c_w-self.c_l, self.s_w-self.s_l), axis=1)


@dataclass(frozen=True)
class PacketPool:
    """Winner-preserving redraw packets and their sampled provenance."""
    rows: np.ndarray
    anchor_c: np.ndarray
    donor_index: np.ndarray
    a_wins: np.ndarray
    source_indices: np.ndarray
    nuisance_bank: np.ndarray


@dataclass(frozen=True)
class LinearBTDGP:
    """One seeded log-linear geometry, strict sample, and candidate pool."""
    seed: int
    d_c: int
    d_s: int
    beta_teacher: float
    beta_dpo: float
    A: np.ndarray
    B: np.ndarray
    R: np.ndarray
    residual_cross: np.ndarray
    u_c: np.ndarray
    u_s: np.ndarray
    strict: PreferenceLatents
    pools: dict[str, PacketPool]
    diagnostics: dict

    @property
    def dimension(self) -> int:
        return self.d_c + self.d_s


@dataclass(frozen=True)
class MatchedScores:
    """Selector scores, rankings, and numerical diagnostics."""
    scores: dict[str, dict[str, np.ndarray]]
    rankings: dict[str, dict[str, np.ndarray]]
    diagnostics: dict


def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def _sample_options(
    geometry: GenericNonlinearDGP, n: int, environment: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, t = ENVIRONMENTS[environment]
    q = rng.normal(size=n)
    z = rng.normal(size=n)
    g = a*q + np.sqrt(max(0.0, 1.0-a*a))*z
    residual_covariance = geometry.residual_covariance(t)
    residual = rng.normal(size=(n, geometry.dimension)) @ np.linalg.cholesky(
        residual_covariance).T
    eps_c, eps_s = residual[:, :geometry.d_c], residual[:, geometry.d_c:]
    c = geometry.mu_c + q[:, None]*geometry.u_c + eps_c
    s = g[:, None]*geometry.u_s + eps_s
    return q, c, s


def _sample_preferences(
    geometry: GenericNonlinearDGP, n: int, rng: np.random.Generator,
) -> PreferenceLatents:
    q_a, c_a, s_a = _sample_options(geometry, n, "P", rng)
    q_b, c_b, s_b = _sample_options(geometry, n, "P", rng)
    probability = expit(geometry.beta_teacher*(q_a-q_b))
    a_wins = rng.random(n) < probability

    def orient(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = a_wins[:, None] if a.ndim == 2 else a_wins
        return np.where(mask, a, b), np.where(mask, b, a)

    c_w, c_l = orient(c_a, c_b)
    s_w, s_l = orient(s_a, s_b)
    q_w, q_l = orient(q_a, q_b)
    return PreferenceLatents(c_w, s_w, q_w, c_l, s_l, q_l)


def _donors(
    bank_size: int, anchors: int, redraws: int, rng: np.random.Generator,
) -> np.ndarray:
    return rng.integers(0, bank_size, size=(anchors, redraws), dtype=np.int64)


def _make_pools(
    strict: PreferenceLatents, n_candidates: int,
    rng: np.random.Generator,
) -> dict[str, PacketPool]:
    source_indices = np.sort(rng.choice(
        len(strict.c_w), size=n_candidates, replace=False))
    # Candidate anchors are a seed-specific strict-pair subsample, while the
    # empirical redraw distribution is estimated from the full strict set.
    bank = strict.nuisance_bank
    donors = _donors(len(bank), n_candidates, 1, rng)[:, 0]
    # The displayed strict winner is response A. A separate, fixed coin
    # determines which response supplies the endpoint training label.
    a_wins = rng.random(n_candidates) < 0.5
    resampling_rows = strict.s_w[source_indices]-bank[donors]
    return {
        "resampling": PacketPool(
            resampling_rows, strict.c_w[source_indices].copy(), donors,
            a_wins, source_indices, bank),
    }


def make_linear_bt_dgp(
    *, seed: int, d_c: int = 10, d_s: int = 10,
    block_condition_number: float = 20.0,
    r_min: float = 0.20, r_max: float = 0.65,
    gamma_c: float = 1.0, gamma_s: float = 1.0,
    beta_teacher: float = 2.0, beta_dpo: float = 1.0,
    n_strict: int = 10_000, n_candidates: int = 10_000,
) -> LinearBTDGP:
    """Construct one matched linear/nonlinear-DGP cell in float64."""
    if not 1 <= n_candidates <= n_strict:
        raise ValueError("candidate source count must be between 1 and n_strict")
    geometry = GenericNonlinearDGP(
        seed=seed, d_c=d_c, d_s=d_s,
        block_condition_number=block_condition_number,
        r_min=r_min, r_max=r_max, gamma_c=gamma_c, gamma_s=gamma_s,
        beta_teacher=beta_teacher, device="cpu")
    strict = _sample_preferences(geometry, n_strict, _rng(seed, 10_001))
    pools = _make_pools(strict, n_candidates, _rng(seed, 10_002))
    option_moments = {
        name: geometry.option_moments(name) for name in ENVIRONMENTS}
    clean = option_moments["Q_clean"]
    diagnostics = {
        **geometry.geometry_diagnostics(),
        "option_moments": option_moments,
        "q_clean_cov_cs_norm": float(np.linalg.norm(clean["cov_cs"])),
        "q_clean_spurious_q_covariance_norm": 0.0,
        "strict_rows": n_strict,
        "candidate_rows": n_candidates,
        "nuisance_bank_rows": 2*n_strict,
        "zero_contrast_ties": int(np.count_nonzero(
            np.all(pools["resampling"].rows == 0.0, axis=1))),
        "realized_a_fraction": float(pools["resampling"].a_wins.mean()),
    }
    return LinearBTDGP(
        seed=seed, d_c=d_c, d_s=d_s,
        beta_teacher=beta_teacher, beta_dpo=beta_dpo,
        A=geometry.A.copy(), B=geometry.B.copy(), R=geometry.R.copy(),
        residual_cross=geometry.residual_cross.copy(),
        u_c=geometry.u_c.copy(), u_s=geometry.u_s.copy(),
        strict=strict, pools=pools, diagnostics=diagnostics)


def solve_dgp(
    dgp: LinearBTDGP, operator: str, selected: np.ndarray,
    *, starting_theta: np.ndarray | None = None,
    tolerance: float = 1e-10, maximum_iterations: int = 250,
) -> ExactSolve:
    """Optimize strict data plus the selected realized tie packets."""
    return solve_exact_packet_dpo(
        dgp.strict.strict_deltas, dgp.pools[operator].rows,
        np.asarray(selected, dtype=np.int64), dgp.beta_dpo, dgp.d_c,
        a_wins=dgp.pools[operator].a_wins,
        starting_theta=starting_theta,
        tolerance=tolerance, maximum_iterations=maximum_iterations)


def solve_single_packet(
    dgp: LinearBTDGP, operator: str, index: int, strict: ExactSolve,
    *, symmetric: bool, tolerance: float = 1e-10,
    maximum_iterations: int = 250,
) -> ExactSolve:
    """Re-optimize one packet with weight 1/N relative to strict mean loss."""
    return solve_exact_packet_dpo(
        dgp.strict.strict_deltas, dgp.pools[operator].rows,
        np.asarray([index], dtype=np.int64), dgp.beta_dpo, dgp.d_c,
        a_wins=dgp.pools[operator].a_wins, symmetric=symmetric,
        starting_theta=strict.theta, tolerance=tolerance,
        maximum_iterations=maximum_iterations)


def _correlation(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        "spearman": float(spearmanr(left, right).statistic),
        "pearson": float(pearsonr(left, right).statistic),
    }


def compute_scores(dgp: LinearBTDGP, strict: ExactSolve) -> MatchedScores:
    """Compute PM and exact oracle-influence scores at the strict optimum."""
    strict_loss, strict_gradient, hessian = empirical_dpo(
        strict.theta, dgp.strict.strict_deltas, dgp.beta_dpo)
    del strict_loss
    if np.linalg.norm(strict_gradient) > 1e-10:
        raise RuntimeError("selector checkpoint is not an empirical optimum")
    hessian = _symmetric(hessian)
    eigenvalues = np.linalg.eigvalsh(hessian)
    if eigenvalues[0] <= 0.0:
        raise np.linalg.LinAlgError("strict empirical Hessian is not positive definite")
    factor = np.linalg.cholesky(hessian)
    scores: dict[str, dict[str, np.ndarray]] = {}
    rankings: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict = {
        "hessian_eigenvalues": eigenvalues,
        "hessian_condition_number": float(eigenvalues[-1]/eigenvalues[0]),
        "operators": {},
    }
    for operator in OPERATORS:
        pool = dgp.pools[operator].rows
        _, sigma_ss = p_pair_block_second_moments(dgp)
        w = strict.theta[dgp.d_c:]
        v = np.zeros(dgp.dimension, dtype=np.float64)
        v[dgp.d_c:] = 2.0 * sigma_ss @ w
        solved = np.linalg.solve(factor.T, np.linalg.solve(factor, v))
        residual = float(np.linalg.norm(hessian@solved-v)/np.linalg.norm(v))
        sources = symmetric_packet_gradients(
            strict.theta, pool, dgp.beta_dpo, dgp.d_c)/len(dgp.strict.c_w)
        pm = np.abs(dgp.beta_dpo*(pool@strict.theta[dgp.d_c:]))
        oracle = sources @ solved
        scores[operator] = {"PM": pm, "Oracle": oracle}
        rankings[operator] = {
            name: np.argsort(-value, kind="stable").astype(np.int64)
            for name, value in scores[operator].items()}
        denominator = float(w @ sigma_ss @ w)
        if denominator <= 0.0:
            raise RuntimeError("strict optimum has no spurious energy")
        lam = float((sigma_ss @ w) @ solved[dgp.d_c:] / (2.0*denominator))
        if lam <= 0.0:
            raise RuntimeError("conditional influence coefficient is nonpositive")
        empirical_c = (pool.T @ pool) / len(pool)
        relative_moment_error = float(
            np.linalg.norm(empirical_c-sigma_ss) / np.linalg.norm(sigma_ss))
        predicted_conditional = lam / len(dgp.strict.c_w) * pm * np.tanh(pm/2.0)
        diagnostics["operators"][operator] = {
            "oracle_residual": residual,
            "conditional_influence_lambda": lam,
            "predicted_conditional_influence": predicted_conditional,
            "empirical_packet_second_moment": empirical_c,
            "population_strict_spurious_second_moment": sigma_ss,
            "packet_moment_relative_error": relative_moment_error,
            "score_correlations": {
                "PM__Oracle": _correlation(pm, oracle),
            },
        }
    return MatchedScores(scores, rankings, diagnostics)


def endpoint_metrics(
    dgp: LinearBTDGP, solve: ExactSolve,
    strict_theta: np.ndarray | None = None,
) -> dict:
    """Evaluate population accuracies, energies, and solve diagnostics."""
    accuracy = {
        name: population_expected_bt_accuracy(solve.theta, dgp, name)
        for name in ENVIRONMENTS}
    component = trained_component_metrics(
        solve.theta, solve.theta if strict_theta is None else strict_theta, dgp)
    return {
        "accuracy": accuracy,
        "energy": {
            "spurious_energy": component["E_S"],
            "causal_energy": component["E_C"],
            "normalized_spurious_energy": component["normalized_spurious_energy"],
            "total_energy": component["total_energy"],
            "energy_valid": True,
        },
        "causal_displacement": component["causal_displacement"],
        "objective": solve.objective,
        "strict_loss": solve.strict_loss,
        "gradient_norm": solve.gradient_norm,
        "hessian_minimum_eigenvalue": float(solve.hessian_eigenvalues[0]),
        "hessian_maximum_eigenvalue": float(solve.hessian_eigenvalues[-1]),
        "hessian_condition_number": solve.hessian_condition_number,
        "causal_parameter_norm": float(np.linalg.norm(solve.theta[:dgp.d_c])),
        "spurious_parameter_norm": float(np.linalg.norm(solve.theta[dgp.d_c:])),
        "finite": bool(np.isfinite(solve.theta).all()),
    }
