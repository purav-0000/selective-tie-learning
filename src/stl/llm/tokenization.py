"""Canonical prompt/completion encoding for LLM scoring and evaluation.

Scoring and evaluation must measure the conditional log-likelihood of exactly
the token sequence that ``trl.DPOTrainer`` trains on, otherwise the implicit
reward being ranked is not the reward that was optimized. TRL's text path
(``DPOTrainer._prepare_dataset`` in TRL >= 1.7) does three things we mirror
one-for-one:

  1. append the EOS token *as text* to the completion (unless already there);
  2. tokenize the prompt alone and the joint ``prompt + completion`` string,
     both with the tokenizer's default special-token handling;
  3. split at the tokenized-prompt prefix: completion ids are
     ``joint[len(prompt_ids):]``.

TRL only logs a warning when the prefix property fails (a BPE merge across
the boundary); we raise instead, because our prompts end in the explicit
``--- Answer ---\n`` delimiter and a violation would mean scoring and
training silently disagree about which tokens are the completion.
"""
from __future__ import annotations

from typing import List, Tuple


def encode_full(tokenizer, prompt: str, completion: str) -> Tuple[List[int], int]:
    """Return untruncated ``(input_ids, completion_start)`` for one example.

    ``input_ids`` covers ``prompt + completion + EOS`` exactly as TRL's
    DPO data pipeline builds it; ``completion_start`` indexes the first
    completion token.
    """
    eos = tokenizer.eos_token or ""
    if eos and not completion.endswith(eos):
        completion = completion + eos
    prompt_ids = tokenizer(prompt, truncation=False)["input_ids"]
    full_ids = tokenizer(prompt + completion, truncation=False)["input_ids"]
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "tokenized prompt is not a prefix of tokenized prompt+completion; "
            "scoring would disagree with DPO training about the completion "
            "boundary")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("completion tokenized to an empty sequence")
    return full_ids, len(prompt_ids)


def conditional_length(tokenizer, prompt: str, completion: str) -> int:
    """Joint token count of the exact training sequence (incl. EOS)."""
    full_ids, _ = encode_full(tokenizer, prompt, completion)
    return len(full_ids)


def encode_conditional(tokenizer, prompt: str, completion: str,
                       max_length: int) -> Tuple[List[int], int]:
    """Return ``(input_ids, completion_start)`` capped at ``max_length``.

    The completion is never truncated; excess prompt tokens are removed from
    the left. Runs are audited so this truncation path should stay dead. When
    it does fire, a leading BOS token is preserved (the TRL DPO collator keeps
    BOS), so the trimmed sequence stays well-formed instead of silently losing
    its BOS along with the other left-side prompt tokens.
    """
    full_ids, completion_start = encode_full(tokenizer, prompt, completion)
    completion_len = len(full_ids) - completion_start
    if completion_len >= max_length:
        raise ValueError("completion alone exceeds max_length")
    if len(full_ids) <= max_length:
        return full_ids, completion_start
    cut = len(full_ids) - max_length
    bos_id = getattr(tokenizer, "bos_token_id", None)
    head = [full_ids[0]] if bos_id is not None and full_ids[0] == bos_id else []
    body_start = len(head) + cut
    if body_start > completion_start:
        raise ValueError("max_length would truncate completion tokens")
    return head + full_ids[body_start:], completion_start - cut
