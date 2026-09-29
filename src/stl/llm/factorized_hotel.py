"""Factorized-response rendering and evaluation for the controlled hotel DGP."""
from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import hotel as H
from .data import _REQUIREMENTS, _generate_strict, _hotel_from_dict
from .loglik import sequence_logprobs
from .tokenization import encode_full


PROMPT = (
    "You are choosing a hotel for a conference trip. Prefer a hotel that is "
    "close to the convention center, inexpensive, and at least three stars. "
    "Recommend the better hotel. Respond with the complete profile of the "
    "hotel you recommend.\n\nRecommended hotel profile:\n"
)


def render_profile(hotel: H.Hotel) -> str:
    """Render one alternative without pair-relative or ONE/TWO language."""
    return hotel.to_description()


def factorize_joint_record(record: dict[str, Any], environment: str, *,
                           prompt: str = PROMPT) -> dict[str, Any]:
    """Convert an existing paired hotel draw into independently scored responses."""
    metadata = record["metadata"]
    first = _hotel_from_dict(metadata["hotel_a"])
    second = _hotel_from_dict(metadata["hotel_b"])
    utility_first = float(metadata["spurious_tracking"]["hotel_a"]["utility"])
    utility_second = float(metadata["spurious_tracking"]["hotel_b"]["utility"])
    if utility_first == utility_second:
        raise ValueError("strict hotel utilities must not tie")
    winner, loser = ((first, second) if utility_first > utility_second
                     else (second, first))
    winner_utility, loser_utility = sorted(
        (utility_first, utility_second), reverse=True)
    return {
        "anchor_id": record["anchor_id"],
        "prompt": prompt,
        "chosen": render_profile(winner),
        "rejected": render_profile(loser),
        "metadata": {
            "environment": environment,
            "winner_utility": winner_utility,
            "loser_utility": loser_utility,
            "utility_gap": winner_utility - loser_utility,
            "winner_hotel": metadata["winner_hotel"],
            "hotel_a": metadata["hotel_a"],
            "hotel_b": metadata["hotel_b"],
            "spurious_tracking": metadata["spurious_tracking"],
        },
    }


def generate_factorized_rows(*, n: int, rho: float, environment: str,
                             seed: int, utility_noise_std: float = 2.5,
                             strict_min_gap: float = 5.0,
                             anchor_prefix: str | None = None,
                             prompt: str = PROMPT,
                             ) -> list[dict[str, Any]]:
    modes = {"P": "normal", "Q_clean": "clean", "Q_adv": "adversarial"}
    if environment not in modes:
        raise ValueError(f"unknown environment {environment!r}")
    random.seed(seed)
    records = _generate_strict(
        n, rho, modes[environment], force_balance=True,
        anchor_prefix=(anchor_prefix or f"factorized_{environment.lower()}"),
        utility_noise_std=utility_noise_std,
        strict_min_gap=strict_min_gap,
        spurious_seed=seed + 10_000,
        spurious_features=H.SPURIOUS_FEATURES,
    )
    return [factorize_joint_record(record, environment, prompt=prompt)
            for record in records]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def causal_signature(row: dict[str, Any]) -> tuple:
    """Fields that determine utility, including the persistent noise draw."""
    fields = (
        "distance_to_convention_center", "price_per_night", "star_rating",
        "has_pool", "has_gym", "has_breakfast", "has_parking",
        "utility_noise",
    )
    metadata = row["metadata"]
    hotels = [metadata["hotel_a"], metadata["hotel_b"]]
    return tuple(tuple(hotel[field] for field in fields) for hotel in hotels)


def validate_paired_environments(groups: dict[str, Sequence[dict[str, Any]]]) -> None:
    reference = groups["P"]
    for name, rows in groups.items():
        if name == "P":
            continue
        if len(rows) != len(reference):
            raise ValueError("deployment environments have different sizes")
        for base, shifted in zip(reference, rows):
            if causal_signature(base) != causal_signature(shifted):
                raise ValueError("deployment shift changed causal hotel features")
            if (base["metadata"]["winner_utility"],
                    base["metadata"]["loser_utility"]) != (
                    shifted["metadata"]["winner_utility"],
                    shifted["metadata"]["loser_utility"]):
                raise ValueError("deployment shift changed utility labels")


def _score_pair_rows(model: Any, tokenizer: Any,
                     rows: Sequence[dict[str, Any]], *,
                     max_length: int, batch_size: int) -> dict[str, Any]:
    device = str(next(model.parameters()).device)
    encoded_chosen = [encode_full(tokenizer, row["prompt"], row["chosen"])
                      for row in rows]
    encoded_rejected = [encode_full(tokenizer, row["prompt"], row["rejected"])
                        for row in rows]
    lengths = [len(ids) for ids, _ in encoded_chosen + encoded_rejected]
    if any(length > max_length for length in lengths):
        raise ValueError("factorized hotel evaluation would truncate")
    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("tokenizer has no padding token")
    chosen = sequence_logprobs(
        model, encoded_chosen, device, pad, batch_size)
    rejected = sequence_logprobs(
        model, encoded_rejected, device, pad, batch_size)
    margins = [a - b for a, b in zip(chosen, rejected)]
    return {
        "margins": margins,
        "accuracy": sum(value > 0 for value in margins) / len(margins),
        "mean_margin": sum(margins) / len(margins),
        "max_sequence_tokens": max(lengths),
    }


def score_implicit_pair_margins(policy: Any, tokenizer: Any,
                                rows: Sequence[dict[str, Any]], *,
                                beta: float, max_length: int,
                                batch_size: int) -> list[float]:
    policy_scores = _score_pair_rows(
        policy, tokenizer, rows, max_length=max_length,
        batch_size=batch_size)
    with policy.disable_adapter():
        reference_scores = _score_pair_rows(
            policy, tokenizer, rows, max_length=max_length,
            batch_size=batch_size)
    return [beta * (p - r) for p, r in zip(
        policy_scores["margins"], reference_scores["margins"])]


def evaluate_implicit_reward(policy: Any, tokenizer: Any,
                             groups: dict[str, Sequence[dict[str, Any]]], *,
                             beta: float, max_length: int,
                             batch_size: int) -> dict[str, Any]:
    output = {}
    for name, rows in groups.items():
        margins = score_implicit_pair_margins(
            policy, tokenizer, rows, beta=beta,
            max_length=max_length, batch_size=batch_size)
        output[name] = {
            "n": len(margins),
            "accuracy": sum(value > 0 for value in margins) / len(margins),
            "mean_margin": sum(margins) / len(margins),
            "mean_square_margin": sum(value * value for value in margins) / len(margins),
            "root_mean_square_margin": (
                sum(value * value for value in margins) / len(margins)) ** 0.5,
            "max_sequence_tokens": max(
                len(encode_full(tokenizer, row["prompt"], row[key])[0])
                for row in rows for key in ("chosen", "rejected")),
        }
    return output
