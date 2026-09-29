"""Typed runtime configuration for hotel and emoji campaigns.

Paper-facing choices live in the YAML files under
``experiments/llm/counterfactual_operator/configs``.  This dataclass is the
typed interface passed to training, scoring, and evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List


@dataclass
class LLMConfig:
    """Parameters shared by data generation, training, and scoring."""

    # Data-generating process
    N: int = 5000                     # strict training pairs (anchors)
    N_test: int = 2000                # pairs per deployment environment
    rho: float = 0.85                 # hotel default; emoji YAML sets 0.95
    utility_noise_std: float = 2.5    # persistent hotel-level utility noise
    strict_min_gap: float = 5.0       # strict pairs satisfy |u_A-u_B| >= this
    # Base model and LoRA adapter
    base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    base_model_revision: str | None = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"])
    attn_implementation: str = "flash_attention_2"

    # DPO training
    beta_dpo: float = 0.1             # DPO KL coefficient
    learning_rate: float = 1e-5
    epochs: int = 1
    batch_size: int = 4              # per-device train batch
    grad_accum: int = 4
    max_grad_norm: float = 1.0
    max_length: int = 1024
    warmup_ratio: float = 0.1        # converted to explicit warmup steps
    optim: str = "adamw_torch_fused"
    lr_scheduler: str = "cosine"
    grad_checkpointing: bool = True
    # Endpoint training uses fixed realized orientations; scoring is symmetric.
    symmetric_tie_packets: bool = False

    # PM and practical-influence scoring
    probe_batch_size: int = 2         # exact-tie gradient microbatch
    selection_probe_size: int = 500
    N_ggn: int = 1000                 # strict rows defining the GGN
    ggn_damping_fraction: float = 0.05
    ggn_dtype: str = "bf16"           # transient Jacobian storage
    ggn_solve_tol: float = 1e-4

    # Evaluation
    eval_batch_size: int = 4

    # Reproducibility
    seed: int = 7

    def to_dict(self):
        return asdict(self)
