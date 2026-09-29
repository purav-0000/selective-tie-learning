"""One-sided hotel and direct emoji-removal counterfactual experiments."""
from __future__ import annotations

import copy
from hashlib import sha256
import json
import random
from typing import Any, Mapping, Sequence

from . import hotel as H
from .data import _hotel_from_dict
from .factorized_hotel import generate_factorized_rows, render_profile
from .real_data.emoji_formatting import EMOJI, emoji_format, remove_emoji_format


def digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()


def _u(seed: int, key: str, stream: str) -> float:
    return int.from_bytes(sha256(
        f"one-sided-v1:{seed}:{key}:{stream}".encode()).digest()[:8], "little") / 2**64


def _integer(seed: int, key: str, stream: str, modulus: int) -> int:
    if modulus < 1:
        raise ValueError("empty sampling range")
    value = sha256(f"one-sided-v1:{seed}:{key}:{stream}".encode()).digest()
    return int.from_bytes(value[:8], "little") % modulus


def _orient(candidate: dict[str, Any], seed: int) -> dict[str, Any]:
    a_wins = _u(seed, candidate["candidate_id"], "realized_orientation") < .5
    candidate["chosen"] = candidate["response_a"] if a_wins else candidate["response_b"]
    candidate["rejected"] = candidate["response_b"] if a_wins else candidate["response_a"]
    candidate["realized_winner"] = "A" if a_wins else "B"
    candidate["metadata"] = {
        "tie_chosen": candidate["chosen"], "tie_rejected": candidate["rejected"]}
    return candidate


def hotel_one_sided_candidates(strict: Sequence[dict[str, Any]], *, seed: int,
                               prefix: str = "one_sided_hotel") -> list[dict[str, Any]]:
    """Keep each displayed strict winner and redraw only its nuisance block."""
    bank = [hotel for row in strict for hotel in
            (row["metadata"]["hotel_a"], row["metadata"]["hotel_b"])]
    out = []
    for i, row in enumerate(strict):
        candidate_id = f"{prefix}:{row['anchor_id']}"
        anchor = _hotel_from_dict(row["metadata"]["winner_hotel"])
        donor_index = _integer(seed, candidate_id, "donor", len(bank))
        changed = H.copy_spurious_features(copy.deepcopy(anchor),
                                           _hotel_from_dict(bank[donor_index]))
        candidate = {
            "candidate_id": candidate_id, "anchor_id": row["anchor_id"],
            "prompt": row["prompt"], "response_a": row["chosen"],
            "response_b": render_profile(changed),
            "operator": "strict_winner_vs_one_empirical_nuisance_redraw",
            "donor_index": donor_index,
        }
        if candidate["response_a"] != render_profile(anchor):
            raise RuntimeError("hotel counterfactual changed the displayed winner")
        out.append(_orient(candidate, seed))
    return out


def hotel_bundle(raw: Mapping[str, Any], seed: int):
    """Build disjoint hotel training, scoring, and evaluation sets."""
    data = raw["data"]
    n = int(data["strict_rows"])
    nt = int(data["test_rows"])
    args = dict(rho=float(data["rho"]), utility_noise_std=float(data["utility_noise_std"]),
                strict_min_gap=float(data["strict_min_gap"]))
    if "factorized_prompt" in data:
        args["prompt"] = str(data["factorized_prompt"])
    strict = generate_factorized_rows(n=n, environment="P", seed=seed+1000, **args)
    heldout = {e: generate_factorized_rows(n=nt, environment=e, seed=seed+2000, **args)
               for e in ("P", "Q_clean", "Q_adv")}
    candidates = hotel_one_sided_candidates(strict, seed=seed+3000)
    base = generate_factorized_rows(
        n=500, environment="P", seed=seed+40_000, **args,
        anchor_prefix=f"one_sided_selection_s{seed}")
    probes = {("selection", 500): hotel_one_sided_candidates(
        base, seed=seed+50_000, prefix="one_sided_selection")}
    return strict, candidates, probes, heldout


def hotel_energy_probe(raw: Mapping[str, Any], strict: Sequence[dict[str, Any]],
                       *, seed: int, anchors: int,
                       redraws: int) -> list[dict[str, Any]]:
    """Independent winner-preserving redraws for endpoint E_cf."""
    if anchors < 1 or redraws < 2:
        raise ValueError("hotel energy probe requires anchors >=1 and redraws >=2")
    data = raw["data"]
    args = dict(
        rho=float(data["rho"]),
        utility_noise_std=float(data["utility_noise_std"]),
        strict_min_gap=float(data["strict_min_gap"]),
    )
    if "factorized_prompt" in data:
        args["prompt"] = str(data["factorized_prompt"])
    source = generate_factorized_rows(
        n=anchors, environment="P", seed=seed+80_000,
        anchor_prefix=f"hotel_energy_s{seed}", **args)
    bank = [hotel for row in strict for hotel in
            (row["metadata"]["hotel_a"], row["metadata"]["hotel_b"])]
    output = []
    for group_id, row in enumerate(source):
        winner = _hotel_from_dict(row["metadata"]["winner_hotel"])
        for redraw in range(redraws):
            key = f"{row['anchor_id']}:{redraw}"
            donor = _integer(seed, key, "energy:donor", len(bank))
            transformed = H.copy_spurious_features(
                copy.deepcopy(winner), _hotel_from_dict(bank[donor]))
            output.append({
                "candidate_id": f"hotel-energy:{row['anchor_id']}:{redraw}",
                "anchor_id": row["anchor_id"],
                "prompt": row["prompt"],
                "response_a": row["chosen"],
                "response_b": render_profile(transformed),
                "metadata": {
                    "group_id": group_id,
                    "redraw": redraw,
                    "donor_index": donor,
                    "operator": "strict_winner_vs_one_empirical_nuisance_redraw",
                },
            })
    return output


def _distinct_counts(seed: int, key: str, stream: str, maximum: int) -> tuple[int, int]:
    first = _integer(seed, key, stream+":first", maximum+1)
    second = _integer(seed, key, stream+":second", maximum)
    second = second if second < first else second+1
    return tuple(sorted((first, second)))


def _emoji_pair(source: Mapping[str, Any], low: int, high: int, safe_high: bool,
                environment: str, correlated: bool) -> dict[str, Any]:
    safe_count, unsafe_count = (high, low) if safe_high else (low, high)
    return {
        "pair_id": source["pair_id"], "source_digest": source["source_digest"],
        "source_row_indices": [source["source_index"]], "prompt": source["prompt"],
        "chosen": emoji_format(source["safe_response"], safe_count) if safe_count else source["safe_response"],
        "rejected": emoji_format(source["unsafe_response"], unsafe_count) if unsafe_count else source["unsafe_response"],
        "safe_emoji_events": safe_count, "unsafe_emoji_events": unsafe_count,
        "emoji_events": abs(high-low), "emoji_side": "winner" if safe_high else "loser",
        "unsafe_severity": source["unsafe_severity"], "response_source": source["response_source"],
        "environment": environment, "correlated_branch": correlated, "emoji": EMOJI,
    }


def emoji_direct_bundle(strict_sources: Sequence[Mapping[str, Any]],
                        heldout_sources: Sequence[Mapping[str, Any]],
                        probe_sources: Mapping[str, Sequence[Mapping[str, Any]]], *,
                        seed: int, rho: float, maximum: int):
    """Construct the correlation-mixture DGP and emoji-removal ties."""
    if not 0 <= rho <= 1 or maximum < 1:
        raise ValueError("invalid emoji DGP parameters")

    def assignment(source, stream):
        key = source["pair_id"]
        low, high = _distinct_counts(seed, key, stream+":counts", maximum)
        correlated = _u(seed, key, stream+":correlated") < rho
        random_high = _u(seed, key, stream+":random_orientation") < .5
        return low, high, correlated, (True if correlated else random_high), random_high

    strict = []
    candidates = []
    for source in strict_sources:
        low, high, corr, p_high, _ = assignment(source, "strict")
        row = _emoji_pair(source, low, high, p_high, "P", corr)
        strict.append(row)
        count = row["safe_emoji_events"]
        raw = source["safe_response"]
        shown = emoji_format(raw, count) if count else raw
        if shown != row["chosen"]:
            raise RuntimeError("emoji candidate does not retain the displayed strict winner")
        candidate = {
            "candidate_id": f"direct-emoji:{maximum}:{source['pair_id']}",
            "pair_id": source["pair_id"], "source_digest": source["source_digest"],
            "source_row_indices": [source["source_index"]], "prompt": source["prompt"],
            "response_a": shown, "response_b": raw, "causal_response": raw,
            "response_a_events": count, "response_b_events": 0,
            "emoji_events": count, "unsafe_severity": source["unsafe_severity"],
            "tie_class": "exact_inert" if count == 0 else "direct_removal",
            "operator": "displayed_strict_winner_vs_registered_emoji_removal",
        }
        if remove_emoji_format(shown) != raw:
            raise RuntimeError("emoji counterfactual changed causal content")
        candidates.append(_orient(candidate, seed))

    heldout = {e: [] for e in ("P", "Q_clean", "Q_adv")}
    for source in heldout_sources:
        low, high, corr, p_high, random_high = assignment(source, "heldout")
        clean_high = _u(seed, source["pair_id"], "heldout:clean_orientation") < .5
        choices = {"P": p_high, "Q_clean": clean_high,
                   "Q_adv": (False if corr else random_high)}
        for env, safe_high in choices.items():
            heldout[env].append(_emoji_pair(source, low, high, safe_high, env, corr))
    probes = {}
    for label, sources in probe_sources.items():
        full = []
        for source in sources:
            low, high, corr, safe_high, _ = assignment(source, f"probe:{label}")
            count = high if safe_high else low
            raw = source["safe_response"]
            shown = emoji_format(raw, count) if count else raw
            full.append(_orient({
                "candidate_id": f"direct-emoji-probe:{label}:{maximum}:{source['pair_id']}",
                "pair_id": source["pair_id"], "prompt": source["prompt"],
                "response_a": shown, "response_b": raw, "causal_response": raw,
                "response_a_events": count, "response_b_events": 0,
                "emoji_events": count, "unsafe_severity": source["unsafe_severity"],
                "operator": "displayed_probe_winner_vs_registered_emoji_removal",
            }, seed+90_000))
        probes[(label, len(full))] = full
    return strict, candidates, probes, heldout


def emoji_energy_probe(sources: Sequence[Mapping[str, Any]], *, seed: int,
                       rho: float, maximum: int,
                       redraws: int) -> list[dict[str, Any]]:
    """Independent safe-response emoji removals for endpoint E_cf."""
    if not sources or not 0 <= rho <= 1 or maximum < 1 or redraws < 1:
        raise ValueError("invalid emoji energy-probe dimensions")
    output = []
    for group_id, source in enumerate(sources):
        for redraw in range(redraws):
            key = f"{source['pair_id']}:{redraw}"
            low, high = _distinct_counts(seed, key, "energy:counts", maximum)
            correlated = _u(seed, key, "energy:correlated") < rho
            safe_high = (True if correlated else
                         _u(seed, key, "energy:random_orientation") < .5)
            safe_count = high if safe_high else low
            raw = source["safe_response"]
            shown = emoji_format(raw, safe_count) if safe_count else raw
            output.append({
                "candidate_id": f"emoji-energy:{maximum}:{source['pair_id']}:{redraw}",
                "anchor_id": str(source["pair_id"]),
                "pair_id": source["pair_id"],
                "source_digest": source["source_digest"],
                "prompt": source["prompt"],
                "response_a": shown,
                "response_b": raw,
                "metadata": {
                    "group_id": group_id,
                    "redraw": redraw,
                    "safe_emoji_events": safe_count,
                    "correlated_branch": correlated,
                    "operator": "displayed_safe_winner_vs_registered_emoji_removal",
                },
            })
    return output


def counterfactual_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return digest([(r["candidate_id"], r["response_a"], r["response_b"],
                    r["chosen"], r["rejected"]) for r in rows])


def dgp_summary(strict: Sequence[Mapping[str, Any]],
                candidates: Sequence[Mapping[str, Any]],
                heldout: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Summarize environment balance and counterfactual construction."""
    result: dict[str, Any] = {
        "strict_rows": len(strict), "candidate_rows": len(candidates),
        "realized_A_fraction": sum(r["realized_winner"] == "A" for r in candidates) / len(candidates),
    }
    if strict and "safe_emoji_events" in strict[0]:
        result.update({
            "strict_safe_high_fraction": sum(r["emoji_side"] == "winner" for r in strict) / len(strict),
            "candidate_inert_fraction": sum(r["tie_class"] == "exact_inert" for r in candidates) / len(candidates),
            "candidate_mean_events": sum(r["emoji_events"] for r in candidates) / len(candidates),
            "heldout_safe_high_fraction": {
                name: sum(r["emoji_side"] == "winner" for r in rows) / len(rows)
                for name, rows in heldout.items()},
        })
    return result
