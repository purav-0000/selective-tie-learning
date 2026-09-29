"""Population-geometry diagnostics for trained log-linear rewards."""

from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.special import expit, ndtr

from stl.nonlinear.dgp import ENVIRONMENTS


@lru_cache(maxsize=1)
def _integration_rule() -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.hermite.hermgauss(160)


def p_pair_block_second_moments(
    dgp,
) -> tuple[np.ndarray, np.ndarray]:
    """Return population-P SMM blocks for an oriented preference difference.

    Winner/loser orientation changes a pair difference only by a sign, so its
    outer product equals that of the underlying IID alternative difference.
    Consequently each block SMM is twice the corresponding option covariance.
    """
    smm_cc = 2.0 * (dgp.A + np.outer(dgp.u_c, dgp.u_c))
    smm_ss = 2.0 * (dgp.B + np.outer(dgp.u_s, dgp.u_s))
    return 0.5 * (smm_cc + smm_cc.T), 0.5 * (smm_ss + smm_ss.T)


def population_expected_bt_accuracy(theta: np.ndarray, dgp, environment: str) -> float:
    """Integrate hard-choice accuracy against the latent BT teacher exactly in 1D.

    Both the score difference and latent utility difference are jointly Gaussian.
    The only numerical approximation is Gauss--Hermite integration.
    """
    a, _ = ENVIRONMENTS[environment]
    moments = dgp.diagnostics["option_moments"][environment]
    option_covariance = np.block([
        [moments["cov_c"], moments["cov_cs"]],
        [moments["cov_cs"].T, moments["cov_s"]],
    ])
    theta = np.asarray(theta, dtype=np.float64)
    score_variance = float(theta @ option_covariance @ theta)
    if score_variance <= 1e-24:
        return 0.5
    utility_coefficient = float(
        theta[:dgp.d_c] @ dgp.u_c + a * theta[dgp.d_c:] @ dgp.u_s)
    conditional_variance = score_variance - utility_coefficient**2
    if conditional_variance < -1e-10 * max(1.0, score_variance):
        raise FloatingPointError("negative conditional score variance")
    conditional_variance = max(0.0, conditional_variance)
    nodes, weights = _integration_rule()
    utility_difference = 2.0 * nodes  # N(0, 2) under Hermite weights.
    if conditional_variance == 0.0:
        choose_a = (utility_coefficient * utility_difference > 0).astype(float)
    else:
        choose_a = ndtr(
            utility_coefficient * utility_difference
            / np.sqrt(2.0 * conditional_variance))
    teacher = expit(dgp.beta_teacher * utility_difference)
    accuracy = np.dot(weights, teacher * choose_a + (1.0-teacher)*(1.0-choose_a))
    return float(accuracy / np.sqrt(np.pi))


def trained_component_metrics(
    theta: np.ndarray,
    strict_theta: np.ndarray,
    dgp,
) -> dict[str, float]:
    """Evaluate causal/nuisance P energies and causal parameter displacement."""
    theta = np.asarray(theta, dtype=np.float64)
    strict_theta = np.asarray(strict_theta, dtype=np.float64)
    if theta.shape != (dgp.dimension,) or strict_theta.shape != theta.shape:
        raise ValueError("theta and strict_theta must match the DGP dimension")
    smm_cc, smm_ss = p_pair_block_second_moments(dgp)
    theta_c, theta_s = theta[:dgp.d_c], theta[dgp.d_c:]
    strict_c = strict_theta[:dgp.d_c]
    values = {
        "E_S": float(theta_s @ smm_ss @ theta_s),
        "E_C": float(theta_c @ smm_cc @ theta_c),
        "causal_displacement": float(np.linalg.norm(theta_c - strict_c)),
    }
    total = values["E_S"] + values["E_C"]
    values["total_energy"] = total
    values["normalized_spurious_energy"] = (
        values["E_S"] / total if total > 0.0 else 0.0)
    if not all(np.isfinite(value) and value >= 0.0 for value in values.values()):
        raise FloatingPointError("population component metric is invalid")
    return values
