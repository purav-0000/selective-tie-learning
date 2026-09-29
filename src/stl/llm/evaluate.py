"""Canonical unquantized-bfloat16 policy loader for endpoint evaluation."""
from __future__ import annotations

from .config import LLMConfig


def load_eval_model(cfg: LLMConfig, lora_path: str):
    """Load one trained LoRA policy in the same numeric regime as training."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from .dpo import load_tokenizer

    tokenizer = load_tokenizer(cfg)
    try:
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            revision=cfg.base_model_revision,
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation=cfg.attn_implementation,
        )
    except (ImportError, ValueError) as error:
        raise RuntimeError(
            f"required attention backend {cfg.attn_implementation!r} is "
            "unavailable during evaluation") from error
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    return model, tokenizer
