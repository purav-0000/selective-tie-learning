"""Configuration for winner-preserving empirical redraw."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResamplingInfluenceConfig:
    """Scientific and numerical settings for the nonlinear campaign."""
    d_c: int = 10
    d_s: int = 10
    block_condition_number: float = 20.0
    r_min: float = 0.20
    r_max: float = 0.65
    gamma_c: float = 2.0
    gamma_s: float = 2.0
    beta_teacher: float = 4.0
    mixer_hidden: int = 64
    mixer_seed: int = 12345
    width: int = 128
    beta_dpo: float = 1.0
    n_strict: int = 5_000
    n_candidates: int = 2_000
    n_curvature: int = 1_000
    n_selection_probe: int = 500
    n_energy_probe: int = 10_000
    energy_redraws: int = 8
    n_evaluation: int = 50_000
    adam_lr: float = 2e-3
    adam_weight_decay: float = 0.01
    batch_size: int = 512
    epochs: int = 20
    optimizer: str = "adam_lbfgs_regularized"
    lbfgs_lr: float = 1.0
    lbfgs_history: int = 20
    lbfgs_max_iter: int = 500
    lbfgs_max_eval: int = 650
    lbfgs_grad_tolerance: float = 1e-5
    lbfgs_change_tolerance: float = 1e-12
    objective_chunk_size: int = 2_048
    candidate_chunk_size: int = 64
    curvature_chunk_size: int = 64
    spectral_shift_fraction: float = 0.05
    ggn_solve_tolerance: float = 1e-10
    budgets: tuple[int, ...] = (
        100, 200, 300, 400, 500, 750, 1_000, 1_500, 2_000,
    )
    methods: tuple[str, ...] = ("UTL", "PM", "Influence")
    operators: tuple[str, ...] = ("resampling",)
    candidate_operator: str = "strict_winner_vs_empirical_nuisance_redraw"
    candidate_bank: str = "both_options_from_all_strict_pairs"
    training_packet: str = "single_prerealized_uniform"
    scoring_packet: str = "exact_symmetric_average"
    reference_margin: str = "implicit_zero"
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    utl_draws: int = 5
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not 1 <= self.n_candidates <= self.n_strict:
            raise ValueError("candidate pool must be a strict-source subsample")
        if not 1 <= self.n_curvature <= self.n_strict:
            raise ValueError("curvature rows must be a strict-set subsample")
        if self.curvature_chunk_size < 1:
            raise ValueError("curvature chunk size must be positive")
        if self.spectral_shift_fraction <= 0.0:
            raise ValueError("GGN damping fraction must be positive")
        if self.ggn_solve_tolerance <= 0.0:
            raise ValueError("GGN solve tolerance must be positive")
        if self.n_energy_probe < 1 or self.energy_redraws < 1:
            raise ValueError("counterfactual energy probe dimensions must be positive")
        if not self.budgets or max(self.budgets) > self.n_candidates:
            raise ValueError("budgets must be nonempty and fit the candidate pool")
        if tuple(self.operators) != ("resampling",):
            raise ValueError("the active operator is winner-preserving resampling")
        allowed_methods = {"UTL", "PM", "Influence"}
        if (not self.methods or len(set(self.methods)) != len(self.methods)
                or not set(self.methods) <= allowed_methods):
            raise ValueError(
                "methods must be a nonempty unique subset of the supported "
                "selectors")
        if self.candidate_operator != "strict_winner_vs_empirical_nuisance_redraw":
            raise ValueError("candidate operator must preserve the strict winner")
        if self.candidate_bank != "both_options_from_all_strict_pairs":
            raise ValueError("candidate donors must use the full strict bank")
        if self.training_packet != "single_prerealized_uniform":
            raise ValueError("endpoint ties must use fixed realized orientations")
        if self.scoring_packet != "exact_symmetric_average":
            raise ValueError("candidate scoring must use symmetric packets")
        if self.optimizer not in ("adam", "adam_lbfgs_regularized"):
            raise ValueError(
                "nonlinear optimizer must be Adam or regularized Adam+LBFGS")
        if self.reference_margin != "implicit_zero":
            raise ValueError("the nonlinear DPO reference margin must be zero")

    @property
    def input_dimension(self) -> int:
        return self.d_c + self.d_s

    def to_dict(self) -> dict:
        return asdict(self)
