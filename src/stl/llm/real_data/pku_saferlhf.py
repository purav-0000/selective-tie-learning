"""Finite, provenance-tracked PKU-SafeRLHF formatting-confound data."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from typing import Any

from .emoji_formatting import EMOJI, EVENT, emoji_format


SCHEMA_VERSION = 1
DATASET_ID = "PKU-Alignment/PKU-SafeRLHF"
DATASET_REVISION = "9421ffafec3fa40a1f1a7d567b4d525079477ecb"
SUBSET_VERSION = "pku-saferlhf-cross-safe-source-selection-v2"
OPERATOR_VERSION = "pku-saferlhf-blue-diamond-formatting-v1"
ENVIRONMENTS = ("P", "Q_clean", "Q_adv")


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _uniform(seed: int, pair_id: str, stream: str) -> float:
    value = sha256(
        f"{OPERATOR_VERSION}:{seed}:{pair_id}:{stream}".encode("utf-8")
    ).digest()
    return int.from_bytes(value[:8], "little") / 2**64


def _event_count(seed: int, pair_id: str, stream: str) -> int:
    return 8 if _uniform(seed, pair_id, stream) < .30 else 1


def _pair_id(split: str, row: Mapping[str, Any]) -> str:
    return f"{split}:{row['response_0_sha256']}:{row['response_1_sha256']}"


def normalize_source_row(row: Mapping[str, Any], *, split: str,
                         source_index: int,
                         minimum_unsafe_severity: int = 2,
                         require_better_agreement: bool = True,
                         require_same_response_source: bool = True,
                         ) -> dict[str, Any] | None:
    """Return one causally clean cross-safety pair or ``None``.

    The causal winner is the uniquely safe response.  Severity belongs to the
    unsafe response.  Requiring the independent ``better`` annotation to agree
    removes safety/helpfulness trade-off rows from this controlled pilot.
    """
    safe_flags = (bool(row["is_response_0_safe"]),
                  bool(row["is_response_1_safe"]))
    if safe_flags[0] == safe_flags[1]:
        return None
    safe = 0 if safe_flags[0] else 1
    unsafe = 1 - safe
    safe_severity = int(row[f"response_{safe}_severity_level"])
    unsafe_severity = int(row[f"response_{unsafe}_severity_level"])
    if safe_severity != 0 or unsafe_severity < minimum_unsafe_severity:
        return None
    if int(row["safer_response_id"]) != safe:
        raise ValueError("cross-safety row contradicts safer_response_id")
    if require_better_agreement and int(row["better_response_id"]) != safe:
        return None
    if (require_same_response_source and
            row["response_0_source"] != row["response_1_source"]):
        return None
    prompt = str(row["prompt"]).strip()
    safe_response = str(row[f"response_{safe}"]).strip()
    unsafe_response = str(row[f"response_{unsafe}"]).strip()
    if not prompt or not safe_response or not unsafe_response:
        return None
    if safe_response == unsafe_response:
        return None
    if EMOJI in safe_response or EMOJI in unsafe_response:
        return None
    pair_id = _pair_id(split, row)
    source = {
        "pair_id": pair_id,
        "split": split,
        "source_index": int(source_index),
        "prompt": prompt,
        "safe_response": safe_response,
        "unsafe_response": unsafe_response,
        "safe_response_id": safe,
        "unsafe_response_id": unsafe,
        "unsafe_severity": unsafe_severity,
        "unsafe_harm_category": dict(
            row[f"response_{unsafe}_harm_category"]),
        "prompt_source": str(row["prompt_source"]),
        "response_source": str(row[f"response_{safe}_source"]),
        "safe_sha256": str(row[f"response_{safe}_sha256"]),
        "unsafe_sha256": str(row[f"response_{unsafe}_sha256"]),
        "better_response_id": int(row["better_response_id"]),
        "safer_response_id": int(row["safer_response_id"]),
    }
    source["source_digest"] = _digest(source)
    return source


def select_sources(train_rows: Sequence[Mapping[str, Any]],
                   test_rows: Sequence[Mapping[str, Any]], *,
                   strict_rows: int = 5_000,
                   minimum_unsafe_severity: int = 2,
                   require_better_agreement: bool = True,
                   require_same_response_source: bool = True,
                   selection_strategy: str = "severity_priority",
                   selection_seed: int = 0,
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    """Select train rows and retain all eligible official-test rows.

    ``uniform`` means a deterministic uniform sample without replacement from
    the eligible source rows.  It uses only source identity, never severity or
    response content.  ``severity_priority`` preserves the original pilot.
    """
    options = {
        "minimum_unsafe_severity": minimum_unsafe_severity,
        "require_better_agreement": require_better_agreement,
        "require_same_response_source": require_same_response_source,
    }
    eligible_train = [source for index, row in enumerate(train_rows)
                      if (source := normalize_source_row(
                          row, split="train", source_index=index,
                          **options)) is not None]
    eligible_test = [source for index, row in enumerate(test_rows)
                     if (source := normalize_source_row(
                         row, split="test", source_index=index,
                         **options)) is not None]
    if strict_rows < 1 or strict_rows > len(eligible_train):
        raise ValueError("strict_rows exceeds the eligible PKU training pool")

    if selection_strategy == "severity_priority":
        ordered = sorted(eligible_train, key=lambda source: (
            -int(source["unsafe_severity"]),
            sha256(
                f"{SUBSET_VERSION}:{selection_seed}:{source['pair_id']}"
                .encode()).digest(),
            source["pair_id"],
        ))
        selection_description = (
            "eligible rows ordered by descending unsafe severity, then a "
            "deterministic source-ID hash")
    elif selection_strategy == "uniform":
        ordered = sorted(eligible_train, key=lambda source: (
            sha256(
                f"{SUBSET_VERSION}:uniform:{selection_seed}:"
                f"{source['pair_id']}".encode()).digest(),
            source["pair_id"],
        ))
        selection_description = (
            "deterministic uniform sample without replacement from all "
            "eligible rows, implemented by a seeded source-ID hash")
    else:
        raise ValueError(
            "selection_strategy must be severity_priority or uniform")
    selected = ordered[:strict_rows]
    heldout = sorted(eligible_test, key=lambda source: source["pair_id"])
    if {row["pair_id"] for row in selected} & {
            row["pair_id"] for row in heldout}:
        raise RuntimeError("official PKU train/test split leaked source pairs")

    def counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
        return {str(key): value for key, value in sorted(
            Counter(row[field] for row in rows).items())}

    audit = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "subset_version": SUBSET_VERSION,
        "selection_strategy": selection_strategy,
        "selection_seed": int(selection_seed),
        "rule": (
            "exactly one safe response; safe severity 0; unsafe severity at "
            f"least {minimum_unsafe_severity}; safer and better annotations "
            "both choose the safe response; same response-generator source"
        ),
        "selection": f"{selection_description}; first strict_rows retained",
        "raw_rows": {"train": len(train_rows), "test": len(test_rows)},
        "eligible_rows": {
            "train": len(eligible_train), "test": len(eligible_test)},
        "selected_rows": {"strict": len(selected), "heldout": len(heldout)},
        "eligible_train_severity": counts(eligible_train, "unsafe_severity"),
        "selected_strict_severity": counts(selected, "unsafe_severity"),
        "heldout_severity": counts(heldout, "unsafe_severity"),
        "selected_response_sources": counts(selected, "response_source"),
        "selected_pair_ids_digest": _digest(
            [row["pair_id"] for row in selected]),
        "heldout_pair_ids_digest": _digest(
            [row["pair_id"] for row in heldout]),
    }
    return selected, heldout, audit


def _render(source: Mapping[str, Any], *, events: int,
            emoji_on_safe: bool, environment: str) -> dict[str, Any]:
    safe = str(source["safe_response"])
    unsafe = str(source["unsafe_response"])
    chosen = emoji_format(safe, events) if emoji_on_safe else safe
    rejected = unsafe if emoji_on_safe else emoji_format(unsafe, events)
    return {
        "pair_id": str(source["pair_id"]),
        "source_digest": str(source["source_digest"]),
        "source_row_indices": [int(source["source_index"])],
        "prompt": str(source["prompt"]),
        "chosen": chosen,
        "rejected": rejected,
        "unsafe_severity": int(source["unsafe_severity"]),
        "response_source": str(source["response_source"]),
        "emoji": EMOJI,
        "emoji_events": int(events),
        "emoji_side": "winner" if emoji_on_safe else "loser",
        "environment": environment,
    }


def rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return _digest([
        (row["pair_id"], row["source_digest"], row["unsafe_severity"],
         row["emoji_events"], row["emoji_side"],
         _digest(row["chosen"]), _digest(row["rejected"]))
        for row in rows
    ])


def build_formatting_bundle(strict_sources: Sequence[Mapping[str, Any]],
                            heldout_sources: Sequence[Mapping[str, Any]], *,
                            seed: int = 0,
                            correlation_strength: float = .70,
                            subset_audit: Mapping[str, Any] | None = None,
                            ) -> tuple[dict[str, list[dict]], dict]:
    if not 0 <= correlation_strength <= 1:
        raise ValueError("correlation_strength must lie in [0,1]")
    if not strict_sources or not heldout_sources:
        raise ValueError("strict and held-out sources must be nonempty")

    strict = []
    for source in strict_sources:
        pair_id = str(source["pair_id"])
        strict.append(_render(
            source,
            events=_event_count(seed, pair_id, "strict_events"),
            emoji_on_safe=(
                _uniform(seed, pair_id, "strict_orientation")
                < correlation_strength),
            environment="P"))

    heldout = {name: [] for name in ENVIRONMENTS}
    for source in heldout_sources:
        pair_id = str(source["pair_id"])
        events = _event_count(seed, pair_id, "heldout_events")
        p_safe = (_uniform(seed, pair_id, "heldout_p_orientation")
                  < correlation_strength)
        clean_safe = (_uniform(seed, pair_id, "heldout_clean_orientation")
                      < .5)
        for environment, emoji_on_safe in {
                "P": p_safe, "Q_clean": clean_safe,
                "Q_adv": not p_safe}.items():
            heldout[environment].append(_render(
                source, events=events, emoji_on_safe=emoji_on_safe,
                environment=environment))

    records = {"strict_train": strict, **{
        f"heldout_{name}": heldout[name] for name in ENVIRONMENTS}}
    if {row["pair_id"] for row in strict} & {
            row["pair_id"] for row in heldout["P"]}:
        raise RuntimeError("strict and held-out PKU sources overlap")
    for p, clean, adv in zip(heldout["P"], heldout["Q_clean"],
                             heldout["Q_adv"], strict=True):
        if not (p["pair_id"] == clean["pair_id"] == adv["pair_id"]):
            raise RuntimeError("held-out environments are misaligned")
        if p["emoji_events"] != clean["emoji_events"] or (
                p["emoji_events"] != adv["emoji_events"]):
            raise RuntimeError("held-out event counts differ")
        if p["emoji_side"] == adv["emoji_side"]:
            raise RuntimeError("P and Q_adv are not exact complements")

    def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(rows),
            "emoji_on_safe_fraction": sum(
                row["emoji_side"] == "winner" for row in rows) / len(rows),
            "eight_event_fraction": sum(
                row["emoji_events"] == 8 for row in rows) / len(rows),
            "unsafe_severity": {str(key): value for key, value in sorted(
                Counter(row["unsafe_severity"] for row in rows).items())},
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "operator_version": OPERATOR_VERSION,
        "emoji": EMOJI,
        "event_literal": EVENT,
        "seed": int(seed),
        "correlation_strength": float(correlation_strength),
        "causal_label": "the uniquely safe response",
        "event_count_distribution": {"1": .70, "8": .30},
        "environments": {
            "P": f"emoji on safe response with probability {correlation_strength}",
            "Q_clean": "emoji side independent of safety with probability .5",
            "Q_adv": "exact complement of each held-out P orientation",
        },
        "subset_audit": dict(subset_audit or {}),
        "summaries": {name: summary(rows) for name, rows in records.items()},
        "digests": {name: rows_digest(rows) for name, rows in records.items()},
    }
    return records, manifest
