"""Strict hotel generation shared by the factorized LLM task."""
from __future__ import annotations

from typing import Dict, List

from . import hotel as H


CONTEXT_PROMPT = (
    "You are helping someone choose the right hotel for their stay. "
    "Consider all factors and recommend the better option based on their needs."
)
_REQUIREMENTS = {
    "requirements": (
        "Trip for conference, want to be close to convention center "
        "and have low price while being at least 3 stars."
    )
}


def _hotel_to_dict(hotel: H.Hotel) -> Dict:
    return {field: getattr(hotel, field) for field in hotel.__dataclass_fields__}


def _hotel_from_dict(value: Dict) -> H.Hotel:
    return H.Hotel(**value)


def _generate_strict(
    n: int,
    rho: float,
    correlation_mode: str,
    force_balance: bool = True,
    anchor_prefix: str = "strict",
    utility_noise_std: float = 2.5,
    strict_min_gap: float = 5.0,
    spurious_seed: int | None = None,
    spurious_features=H.SPURIOUS_FEATURES,
) -> List[Dict]:
    """Generate the exact paired hotel draws used by the paper campaign.

    ``format_dpo_example`` is retained in this path because its fair position
    coin is part of the data-generation random stream. The factorized task
    subsequently renders each hotel as its own
    response and does not train on the temporary ONE/TWO strings.
    """
    generator = H.HotelDatasetGenerator(
        spurious_correlation_strength=rho,
        utility_noise_std=utility_noise_std,
        strict_min_gap=strict_min_gap,
        spurious_seed=spurious_seed,
        spurious_features=spurious_features,
    )
    target_a, target_b = n // 2, n - n // 2
    a_count = b_count = attempts = 0
    records: List[Dict] = []
    maximum_attempts = n * 10
    while len(records) < n and attempts < maximum_attempts:
        attempts += 1
        try:
            hotel_a, hotel_b, label, tracking = generator.generate_hotel_pair(
                _REQUIREMENTS, correlation_mode=correlation_mode
            )
        except RuntimeError:
            continue
        if force_balance:
            need_a = label == H.PreferenceLabel.A_BETTER and a_count < target_a
            need_b = label == H.PreferenceLabel.B_BETTER and b_count < target_b
            if not (need_a or need_b):
                continue
        utility_a = generator.calculate_true_utility(hotel_a, _REQUIREMENTS)
        utility_b = generator.calculate_true_utility(hotel_b, _REQUIREMENTS)
        example = H.format_dpo_example(
            CONTEXT_PROMPT,
            hotel_a.to_description(),
            hotel_b.to_description(),
            utility_a,
            utility_b,
            label,
        )
        if example is None:
            continue
        winner = hotel_a if label == H.PreferenceLabel.A_BETTER else hotel_b
        winner_utility = (
            utility_a if label == H.PreferenceLabel.A_BETTER else utility_b
        )
        records.append(
            {
                "anchor_id": f"{anchor_prefix}_{len(records):06d}",
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
                "metadata": {
                    **example["metadata"],
                    "correlation_mode": correlation_mode,
                    "correlation_strength": rho,
                    "spurious_tracking": tracking,
                    "winner_hotel": _hotel_to_dict(winner),
                    "winner_util": winner_utility,
                    "hotel_a": _hotel_to_dict(hotel_a),
                    "hotel_b": _hotel_to_dict(hotel_b),
                },
            }
        )
        if force_balance:
            if label == H.PreferenceLabel.A_BETTER:
                a_count += 1
            else:
                b_count += 1
    if len(records) < n:
        raise RuntimeError(
            f"only generated {len(records)}/{n} strict pairs within "
            f"{maximum_attempts} attempts; refusing to emit a short dataset"
        )
    return records
