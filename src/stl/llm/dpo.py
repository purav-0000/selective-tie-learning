"""Unquantized-bfloat16 LoRA-DPO training.

The scorer is the *implicit DPO reward* of a LoRA-adapted policy against a
frozen base reference. The
base remains frozen and the LoRA adapter is the only trainable part. Training
uses the ordinary sigmoid objective implemented by ``trl.DPOTrainer``.

All torch/transformers/trl imports are deferred to call time so the rest of the
``stl.llm`` package (data generation) stays importable without a GPU stack.
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

from .config import LLMConfig
from .tokenization import conditional_length


def load_tokenizer(cfg: LLMConfig):
    """Load the pinned tokenizer with left padding and truncation."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.base_model, revision=cfg.base_model_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"
    return tok


def load_base_model(cfg: LLMConfig, device_map: str = "auto"):
    """Load the pinned unquantized-bfloat16 base and attention backend."""
    import torch
    from transformers import AutoModelForCausalLM
    if device_map == "auto":
        device_map = os.environ.get("STL_MODEL_DEVICE_MAP", device_map)
    if device_map not in {"auto", "balanced", "balanced_low_0", "sequential"}:
        raise ValueError(f"unsupported model device map: {device_map!r}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            revision=cfg.base_model_revision,
            device_map=device_map,
            dtype=torch.bfloat16,
            attn_implementation=cfg.attn_implementation,
        )
    except (ImportError, ValueError) as error:
        raise RuntimeError(
            f"required attention backend {cfg.attn_implementation!r} is "
            "unavailable; install the pinned FlashAttention dependency") from error
    print(f"[dpo] attn_implementation = {cfg.attn_implementation}")
    return model


def attach_lora(cfg: LLMConfig, model):
    """Attach the trainable LoRA modules to a frozen base model."""
    from peft import get_peft_model, LoraConfig
    if cfg.grad_checkpointing:
        # bf16 LoRA path: inputs must carry grad so checkpointed segments
        # have a differentiable path to the adapter parameters.
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def load_trainable_adapter(cfg: LLMConfig, adapter_dir: str):
    """Continue a saved strict LoRA policy with the frozen base as reference."""
    from peft import PeftModel

    path = os.path.abspath(adapter_dir)
    if not os.path.isfile(os.path.join(path, "adapter_config.json")):
        raise FileNotFoundError(f"missing strict adapter config: {path}")
    base = load_base_model(cfg)
    if cfg.grad_checkpointing:
        base.enable_input_require_grads()
    policy = PeftModel.from_pretrained(base, path, is_trainable=True)
    if not any(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("continued adapter has no trainable parameters")
    policy.print_trainable_parameters()
    return policy


def train_dpo(cfg: LLMConfig, train_file: str, output_dir: str,
              run_name: str = "run",
              train_samples: Optional[int] = None,
              initial_adapter_dir: Optional[str] = None) -> str:
    """Train a LoRA-DPO checkpoint on ``train_file``; return the adapter dir.

    ``train_file`` is a DPO-format JSONL ({prompt, chosen, rejected}). Used for
    both the strict-only checkpoint and the augmented (strict + ties) ones.
    """
    import torch
    from datasets import load_dataset
    from transformers import set_seed
    from trl import DPOTrainer, DPOConfig

    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(
            f"refusing to overwrite non-empty checkpoint directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Set this before both base-model construction and LoRA attachment: LoRA
    # adapter initialization is random and must be reproducible across methods.
    set_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    ds = load_dataset("json", data_files=train_file, split="train")
    ds = ds.shuffle(seed=cfg.seed)
    if train_samples:
        ds = ds.select(range(min(train_samples, len(ds))))
    has_realized_flags = "realized_tie" in ds.column_names
    unsupported = {"symmetric_packet", "importance_weight", "importance_tie",
                   "reference_ablated_tie"}.intersection(ds.column_names)
    if unsupported:
        raise ValueError(
            "unsupported DPO row fields: "
            + ", ".join(sorted(unsupported)))
    if has_realized_flags:
        flags = ds["realized_tie"]
        if any(not isinstance(flag, bool) for flag in flags):
            raise TypeError("every realized_tie field must be boolean")
        packet_count = sum(flags); strict_count = len(flags) - packet_count
        if not packet_count or not strict_count:
            raise ValueError(
                "realized augmented data must contain strict rows and ties")
        # This field is provenance only. Removing it before tokenization makes
        # this path exactly the ordinary unweighted realized-row DPO loss.
        ds = ds.remove_columns(["realized_tie"])
    else:
        packet_count, strict_count = 0, len(ds)
    print(f"[dpo] {run_name}: train={len(ds)} logical rows "
          f"(strict={strict_count}, packets={packet_count}, no validation holdout)")

    tok = load_tokenizer(cfg)
    # Exact pre-training truncation audit. ``conditional_length`` counts the
    # tokens of ``prompt + completion + EOS`` under the same joint tokenization
    # TRL uses to build ``prompt_ids + completion_ids``, so this count equals
    # the per-sequence length TRL's collator would truncate. Failing here on
    # any over-limit sequence is what makes the collator's truncation
    # unreachable -- see the DPOConfig construction below for why that matters.
    lengths = []
    for row in ds:
        lengths.append(conditional_length(tok, row["prompt"], row["chosen"]))
        lengths.append(conditional_length(tok, row["prompt"], row["rejected"]))
    lengths.sort()
    p99 = lengths[min(len(lengths) - 1, int(0.99 * len(lengths)))]
    too_long = sum(length > cfg.max_length for length in lengths)
    length_audit = {
        "n_sequences": len(lengths), "max_tokens": max(lengths),
        "p99_tokens": p99, "limit": cfg.max_length,
        "would_truncate": too_long,
    }
    print(f"[dpo] token audit: p99={p99}, max={max(lengths)}, "
          f"limit={cfg.max_length}, would_truncate={too_long}")
    if too_long:
        raise ValueError(
            f"{too_long} training sequences exceed max_length={cfg.max_length}; "
            "refusing to truncate")

    torch.cuda.reset_peak_memory_stats()
    policy = (load_trainable_adapter(cfg, initial_adapter_dir)
              if initial_adapter_dir is not None else
              attach_lora(cfg, load_base_model(cfg)))

    # Explicit warmup steps: warmup_ratio is deprecated in this stack, and an
    # explicit count keeps the schedule identical however TRL evolves.
    logical_batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.grad_accum
    effective_logical_batch = logical_batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(ds) / effective_logical_batch)
    warmup_steps = round(cfg.warmup_ratio * steps_per_epoch * cfg.epochs)

    dpo_kwargs = dict(
        output_dir=os.path.join(output_dir, "_trainer"),
        run_name=run_name,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=logical_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=cfg.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_steps=warmup_steps,
        optim=cfg.optim,
        beta=cfg.beta_dpo,
        max_length=cfg.max_length,
        # Pinned defensively. If a sequence ever exceeded max_length, TRL's
        # collator would truncate the concatenated prompt+completion; the
        # scorer/evaluator instead left-truncate the prompt to keep the
        # completion. That disagreement is made unreachable by the exact audit
        # above (which raises first), so we never actually truncate. We pin the
        # mode explicitly -- rather than switch to the now-deprecated
        # 'keep_end' -- only so a future TRL default change cannot silently
        # start dropping completion tokens without also tripping the audit.
        truncation_mode="keep_start",
        # Report at least once per epoch, with a ten-step maximum interval.
        logging_steps=max(1, min(10, steps_per_epoch)),
        eval_strategy="no",
        save_strategy="no",
        bf16=True,
        fp16=False,
        max_grad_norm=cfg.max_grad_norm,
        weight_decay=0.0,
        # TRL has already tokenized this in-memory dataset; in-process loading
        # avoids forking workers after CUDA initialization.
        dataloader_num_workers=0,
        seed=cfg.seed,
        report_to="none",
    )
    args = DPOConfig(**dpo_kwargs)
    # ref_model=None: TRL evaluates the reference log-probs by disabling the
    # LoRA adapter on the policy, which is exactly the frozen base and avoids
    # loading a second full copy of the base weights.
    trainer = DPOTrainer(
        model=policy, ref_model=None, args=args,
        train_dataset=ds, eval_dataset=None,
        processing_class=tok,
    )
    out = trainer.train()
    trainer.save_model(model_dir)
    tok.save_pretrained(model_dir)

    hist = {
        "run_name": run_name,
        "train_file": train_file,
        "initial_adapter_dir": (os.path.abspath(initial_adapter_dir)
                                if initial_adapter_dir is not None else None),
        "train_samples": len(ds),
        "train_loss": getattr(out, "training_loss", None),
        "runtime_s": out.metrics.get("train_runtime"),
        "samples_per_s": out.metrics.get("train_samples_per_second"),
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "token_length_audit": length_audit,
        "symmetric_tie_packets": False,
        "realized_tie_rows": has_realized_flags,
        "logical_rows": len(ds),
        "strict_rows": strict_count,
        "packet_rows": packet_count,
        "logical_microbatch": logical_batch_size,
        "physical_preference_microbatch": logical_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_gradient_norm": cfg.max_grad_norm,
        "effective_logical_batch": effective_logical_batch,
        "objective": (
            "(sum(strict_loss) + sum(realized_tie_loss))/(N+k)"
            if has_realized_flags else "mean(strict_loss)"),
        "requested_epochs": cfg.epochs,
        "completed_epochs": float(trainer.state.epoch or 0.0),
        "global_step": int(trainer.state.global_step),
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    print(f"[dpo] {run_name}: saved adapter -> {model_dir} "
          f"(loss={hist['train_loss']}, {hist['runtime_s']}s)")
    return model_dir
