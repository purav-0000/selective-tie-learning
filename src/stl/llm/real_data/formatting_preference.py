"""Conditional-likelihood evaluation for the emoji-formatting shortcut."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..loglik import sequence_logprobs
from ..tokenization import encode_conditional, encode_full
from .emoji_formatting import preference_summary


def _gap_strata(margins: Sequence[float], sides: Sequence[str],
                gaps: Sequence[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for threshold in (1, 2, 3, 4):
        indices = [index for index, gap in enumerate(gaps)
                   if gap >= threshold]
        output[f"ge{threshold}"] = ({
            "rows": len(indices),
            **preference_summary(
                [margins[index] for index in indices],
                [sides[index] for index in indices]),
        } if indices else None)
    return output


def _encoded(tokenizer: Any, rows: Sequence[dict[str, Any]], key: str,
             max_length: int):
    raw = [encode_full(tokenizer, row["prompt"], row[key]) for row in rows]
    lengths = [len(ids) for ids, _ in raw]
    too_long = sum(length > max_length for length in lengths)
    if too_long:
        raise ValueError(
            f"{too_long} formatting sequences exceed max_length={max_length}; "
            "refusing to truncate")
    return [encode_conditional(tokenizer, row["prompt"], row[key], max_length)
            for row in rows], lengths


def score_model(model: Any, tokenizer: Any, rows: Sequence[dict[str, Any]], *,
                max_length: int, batch_size: int, device: str,
                stratification_field: str = "helpfulness_gap",
                ) -> dict[str, Any]:
    chosen, chosen_lengths = _encoded(tokenizer, rows, "chosen", max_length)
    rejected, rejected_lengths = _encoded(tokenizer, rows, "rejected", max_length)
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("tokenizer has no padding token")
    chosen_lp = sequence_logprobs(model, chosen, device, pad, batch_size)
    rejected_lp = sequence_logprobs(model, rejected, device, pad, batch_size)
    chosen_tokens = [len(ids) - start for ids, start in chosen]
    rejected_tokens = [len(ids) - start for ids, start in rejected]
    total_margin = [a - b for a, b in zip(chosen_lp, rejected_lp)]
    mean_token_margin = [
        a / na - b / nb for a, b, na, nb in
        zip(chosen_lp, rejected_lp, chosen_tokens, rejected_tokens)]
    sides = [str(row["emoji_side"]) for row in rows]
    event_counts = [int(row["emoji_events"]) for row in rows]
    strata_values = [int(row[stratification_field]) for row in rows]
    strata = {}
    for count in (1, 8):
        indices = [index for index, value in enumerate(event_counts)
                   if value == count]
        strata[str(count)] = preference_summary(
            [total_margin[index] for index in indices],
            [sides[index] for index in indices]) if indices else None
    return {
        "total_logprob": preference_summary(total_margin, sides),
        "mean_token_logprob": preference_summary(mean_token_margin, sides),
        "total_logprob_by_event_count": strata,
        f"total_logprob_by_{stratification_field}": _gap_strata(
            total_margin, sides, strata_values),
        "max_sequence_tokens": max(chosen_lengths + rejected_lengths),
        "p99_sequence_tokens": sorted(chosen_lengths + rejected_lengths)[
            int(.99 * (2 * len(rows) - 1))],
        "rows": [{
            "pair_id": row["pair_id"],
            "emoji_side": sides[index],
            "emoji_events": event_counts[index],
            stratification_field: strata_values[index],
            "chosen_completion_tokens": chosen_tokens[index],
            "rejected_completion_tokens": rejected_tokens[index],
            "chosen_logprob": chosen_lp[index],
            "rejected_logprob": rejected_lp[index],
            "total_logprob_margin": total_margin[index],
            "mean_token_logprob_margin": mean_token_margin[index],
        } for index, row in enumerate(rows)],
    }


def compare_policy_reference(policy_scores: dict[str, Any],
                             reference_scores: dict[str, Any], *,
                             beta: float,
                             stratification_field: str = "helpfulness_gap",
                             ) -> dict[str, Any]:
    policy = policy_scores["rows"]
    reference = reference_scores["rows"]
    if [row["pair_id"] for row in policy] != [row["pair_id"] for row in reference]:
        raise ValueError("policy and reference score rows are misaligned")
    sides = [row["emoji_side"] for row in policy]
    if sides != [row["emoji_side"] for row in reference]:
        raise ValueError("policy and reference emoji sides are misaligned")
    margins = [
        beta * (p["total_logprob_margin"] - r["total_logprob_margin"])
        for p, r in zip(policy, reference)]
    counts = [int(row["emoji_events"]) for row in policy]
    strata_values = [int(row[stratification_field]) for row in policy]
    strata = {}
    for count in (1, 8):
        indices = [index for index, value in enumerate(counts) if value == count]
        strata[str(count)] = preference_summary(
            [margins[index] for index in indices],
            [sides[index] for index in indices]) if indices else None
    return {
        "definition": (
            "beta*((logpi(chosen)-logpi(rejected))-"
            "(logref(chosen)-logref(rejected)))"),
        "summary": preference_summary(margins, sides),
        "mean_square_margin": sum(value * value for value in margins) / len(margins),
        "root_mean_square_margin": (
            sum(value * value for value in margins) / len(margins)) ** 0.5,
        "by_event_count": strata,
        f"by_{stratification_field}": _gap_strata(
            margins, sides, strata_values),
        "rows": [{
            "pair_id": row["pair_id"],
            "emoji_side": row["emoji_side"],
            "emoji_events": row["emoji_events"],
            stratification_field: row[stratification_field],
            "implicit_reward_margin": margin,
        } for row, margin in zip(policy, margins)],
    }
