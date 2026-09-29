"""Batched conditional log-likelihood computation shared by scoring and eval.

Sequences are left-padded (RoPE models are shift-invariant, so left padding
with an attention mask is exact), the full-vocab log-softmax is evaluated in
fp32 over sequence chunks to bound peak memory, and per-sequence sums cover
completion tokens only.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

Encoded = Tuple[List[int], int]   # (input_ids, completion_start)


def enable_scoring_layer_checkpointing(model) -> int:
    """Checkpoint decoder blocks while preserving eval mode and its zero dropout.

    Native Transformers checkpointing normally runs only in train mode. Scoring
    must stay in eval mode, so wrap only decoder forwards without changing any
    module's training flag, parameters, or state-dict names.
    """
    from functools import wraps
    import torch
    from torch.utils.checkpoint import checkpoint

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(base, "config"):
        base.config.use_cache = False
    layers = getattr(getattr(base, "model", None), "layers", None)
    if layers is None:
        return 0

    def wrap(original):
        @wraps(original)
        def forward(*args, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(original, *args, use_reentrant=False,
                                  preserve_rng_state=True, **kwargs)
            return original(*args, **kwargs)
        return forward

    count = 0
    for layer in layers:
        if not getattr(layer, "_stl_scoring_checkpoint", False):
            layer.forward = wrap(layer.forward)
            layer._stl_scoring_checkpoint = True
        count += 1
    return count


def forward_logprob_sums(model, encoded: Sequence[Encoded], device,
                         pad_id: int, chunk: int = 128, *, checkpoint_chunks: bool = True):
    """One padded forward; per-sequence completion log-prob sums.

    Differentiable: gradients flow through the model forward, so callers can
    backprop through the returned tensor (shape ``[len(encoded)]``). During
    gradient scoring, recompute each full-vocabulary softmax chunk in backward
    instead of retaining all chunks simultaneously. This is exact activation
    checkpointing, not vocabulary truncation or a likelihood approximation.
    """
    import torch
    width = max(len(ids) for ids, _ in encoded)
    ids_batch, attn_batch, starts = [], [], []
    for ids, start in encoded:
        n_pad = width - len(ids)
        ids_batch.append([pad_id] * n_pad + list(ids))
        attn_batch.append([0] * n_pad + [1] * len(ids))
        starts.append(n_pad + start)
    input_ids = torch.tensor(ids_batch, dtype=torch.long, device=device)
    attention = torch.tensor(attn_batch, dtype=torch.long, device=device)
    # Full-sequence likelihoods need no generation KV cache. Mutable caches
    # would append keys again during checkpoint recomputation.
    logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits

    targets = input_ids[:, 1:]
    pieces = []

    def selected_logprobs(block, labels):
        lp = torch.log_softmax(block.float(), dim=-1)
        return lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    for lo in range(0, width - 1, chunk):
        hi = min(lo + chunk, width - 1)
        if checkpoint_chunks and logits.requires_grad:
            from torch.utils.checkpoint import checkpoint
            selected = checkpoint(selected_logprobs, logits[:, lo:hi], targets[:, lo:hi],
                                  use_reentrant=False, preserve_rng_state=False)
        else:
            selected = selected_logprobs(logits[:, lo:hi], targets[:, lo:hi])
        pieces.append(selected)
    token_lp = torch.cat(pieces, dim=1)

    positions = torch.arange(1, width, device=device).unsqueeze(0)
    start_t = torch.tensor(starts, device=device).unsqueeze(1)
    mask = (positions >= start_t) & attention[:, 1:].bool()
    return (token_lp * mask).sum(dim=1)


def sequence_logprobs(model, encoded: Sequence[Encoded], device, pad_id: int,
                      batch_size: int) -> List[float]:
    """No-grad batched completion log-probs for many sequences."""
    import torch
    out: List[float] = []
    for lo in range(0, len(encoded), batch_size):
        batch = encoded[lo:lo + batch_size]
        with torch.no_grad():
            sums = forward_logprob_sums(model, batch, device, pad_id)
        out.extend(sums.float().cpu().tolist())
    return out
