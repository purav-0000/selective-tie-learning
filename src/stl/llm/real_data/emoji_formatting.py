"""Deterministic content-preserving emoji-formatting shortcut for HelpSteer2."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import math
import re
from typing import Any


SCHEMA_VERSION = 1
OPERATOR_VERSION = "helpsteer2-blue-diamond-formatting-v1"
SUBSET_VERSION = "helpsteer2-emoji-strict-subset-v1"
EMOJI = "🔹"
EVENT = f" {EMOJI}"
ENVIRONMENTS = ("P", "Q_clean", "Q_adv")

_FENCED_CODE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_URL = re.compile(r"https?://\S+")
_PARAGRAPH = re.compile(r"\n\s*\n")
_LINE = re.compile(r"\n")
_SENTENCE = re.compile(r"[.!?](?=\s|$)")
_CLAUSE = re.compile(r"[,;:](?=\s)")


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def formatting_rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return _digest([
        (row["pair_id"], row["source_digest"], row["emoji_events"],
         row["emoji_side"], _digest(row["chosen"]), _digest(row["rejected"]))
        for row in rows])


def _uniform(seed: int, pair_id: str, stream: str) -> float:
    value = sha256(
        f"{OPERATOR_VERSION}:{seed}:{pair_id}:{stream}".encode("utf-8")
    ).digest()
    return int.from_bytes(value[:8], "little") / 2**64


def sample_event_count(seed: int, pair_id: str, stream: str) -> int:
    """Return eight events with probability .3 and one otherwise."""
    return 8 if _uniform(seed, pair_id, stream) < 0.30 else 1


def _protected_positions(text: str) -> set[int]:
    protected: set[int] = set()
    for expression in (_FENCED_CODE, _INLINE_CODE, _URL):
        for match in expression.finditer(text):
            protected.update(range(match.start() + 1, match.end()))
    return protected


def _boundary_tiers(text: str) -> list[list[int]]:
    """Return deterministic, semantically conservative insertion offsets.

    Paragraph, line, sentence, and clause boundaries are preferred. Ordinary
    inter-word whitespace is a final fallback. Boundaries inside code spans or
    URLs are excluded. The endpoint is always available for very short text.
    """
    if not isinstance(text, str) or not text:
        raise ValueError("response must be a nonempty string")
    protected = _protected_positions(text)
    tiers: list[list[int]] = []
    for expression in (_PARAGRAPH, _LINE, _SENTENCE, _CLAUSE):
        positions = []
        for match in expression.finditer(text):
            position = match.start() if expression in (_PARAGRAPH, _LINE) else match.end()
            if position > 0 and position not in protected:
                positions.append(position)
        tiers.append(sorted(set(positions)))
    tiers.append([len(text)])
    return tiers


def formatting_boundaries(text: str) -> list[int]:
    """Return every valid insertion offset, ordered by response position."""
    return sorted({position for tier in _boundary_tiers(text)
                   for position in tier})


def _distributed_offsets(text: str, events: int) -> list[int]:
    if events < 1:
        raise ValueError("event count must be positive")
    # Use every offset from the most semantically conservative tiers needed
    # to provide enough positions. Selecting quantiles from the complete tier
    # (rather than its first few offsets) distributes markers over the whole
    # response instead of clustering them near the beginning.
    candidates: list[int] = []
    for tier in _boundary_tiers(text):
        candidates.extend(tier)
        if len(set(candidates)) >= events:
            break
    candidates = sorted(set(candidates))
    if not candidates:
        candidates = [len(text)]
    targets = [(index + 1) * len(text) / (events + 1)
               for index in range(events)]
    selected: list[int] = []
    unused = set(candidates)
    for target in targets:
        choices = unused if unused else set(candidates)
        position = min(choices, key=lambda value: (abs(value - target), value))
        selected.append(position)
        unused.discard(position)
    return sorted(selected)


def emoji_format(response: str, events: int) -> str:
    """Insert exactly ``events`` fixed emoji markers without rewriting text."""
    if EMOJI in response:
        raise ValueError(f"source response already contains registered emoji {EMOJI}")
    offsets = _distributed_offsets(response, events)
    counts = Counter(offsets)
    pieces: list[str] = []
    previous = 0
    for position in sorted(counts):
        pieces.append(response[previous:position])
        pieces.append(EVENT * counts[position])
        previous = position
    pieces.append(response[previous:])
    transformed = "".join(pieces)
    if transformed.replace(EVENT, "") != response:
        raise RuntimeError("formatting operator failed exact content recovery")
    if transformed.count(EMOJI) != events:
        raise RuntimeError("formatting operator inserted the wrong event count")
    return transformed


def remove_emoji_format(response: str) -> str:
    """Invert :func:`emoji_format` for the registered formatting event."""
    if not isinstance(response, str) or not response:
        raise ValueError("response must be a nonempty string")
    restored = response.replace(EVENT, "")
    if EMOJI in restored:
        raise ValueError("response contains an unregistered placement of the emoji")
    return restored


def _render_pair(source: Mapping[str, Any], *, events: int,
                 emoji_on_winner: bool, environment: str) -> dict[str, Any]:
    winner = str(source["winner"]["response"])
    loser = str(source["loser"]["response"])
    chosen = emoji_format(winner, events) if emoji_on_winner else winner
    rejected = loser if emoji_on_winner else emoji_format(loser, events)
    return {
        "pair_id": str(source["pair_id"]),
        "source_digest": str(source["source_digest"]),
        "source_row_indices": list(source["source_row_indices"]),
        "prompt": str(source["prompt"]),
        "chosen": chosen,
        "rejected": rejected,
        "helpfulness_gap": int(source["helpfulness_gap"]),
        "emoji": EMOJI,
        "emoji_events": events,
        "emoji_side": "winner" if emoji_on_winner else "loser",
        "environment": environment,
    }


def _fixed_strict_subset(rows: Sequence[Mapping[str, Any]],
                         strict_rows: int | None) -> list[Mapping[str, Any]]:
    """Choose a deterministic source-ID subset independent of rho and seed."""
    ordered = sorted(rows, key=lambda row: (
        sha256(f"{SUBSET_VERSION}:{row['pair_id']}".encode("utf-8")).digest(),
        str(row["pair_id"])))
    if strict_rows is None:
        return ordered
    if strict_rows < 1 or strict_rows > len(ordered):
        raise ValueError("strict_rows must lie within the available train pool")
    return ordered[:strict_rows]


def build_formatting_bundle(source_rows: Sequence[Mapping[str, Any]], *,
                            seed: int = 0, correlation_strength: float = 0.90,
                            strict_rows: int | None = None,
                            min_strict_helpfulness_gap: int = 1,
                            ) -> tuple[dict[str, list[dict]], dict]:
    """Build strict P rows and paired held-out P/Qclean/Qadv environments."""
    if any(EMOJI in str(row[side]["response"])
           for row in source_rows for side in ("winner", "loser")):
        raise ValueError("registered emoji already occurs in source corpus")
    if not 0.0 <= correlation_strength <= 1.0:
        raise ValueError("correlation_strength must lie in [0,1]")
    if min_strict_helpfulness_gap < 1:
        raise ValueError("min_strict_helpfulness_gap must be positive")
    all_strict_sources = [
        row for row in source_rows if str(row["split"]) == "train"]
    eligible_strict_sources = [
        row for row in all_strict_sources
        if int(row["helpfulness_gap"]) >= min_strict_helpfulness_gap]
    strict_sources = _fixed_strict_subset(eligible_strict_sources, strict_rows)
    heldout_sources = sorted(
        (row for row in source_rows if str(row["split"]) == "validation"),
        key=lambda row: str(row["pair_id"]))
    if not strict_sources or not heldout_sources:
        raise ValueError("both train and validation source splits are required")

    strict: list[dict] = []
    for source in strict_sources:
        pair_id = str(source["pair_id"])
        events = sample_event_count(seed, pair_id, "strict_a")
        emoji_on_winner = (
            _uniform(seed, pair_id, "strict_orientation")
            < correlation_strength)
        strict.append(_render_pair(
            source, events=events, emoji_on_winner=emoji_on_winner,
            environment="P"))

    heldout = {name: [] for name in ENVIRONMENTS}
    for source in heldout_sources:
        pair_id = str(source["pair_id"])
        events = sample_event_count(seed, pair_id, "heldout_a")
        p_winner = (_uniform(seed, pair_id, "heldout_p_orientation")
                    < correlation_strength)
        clean_winner = _uniform(seed, pair_id, "heldout_clean_orientation") < 0.50
        orientations = {
            "P": p_winner,
            "Q_clean": clean_winner,
            "Q_adv": not p_winner,
        }
        for environment, emoji_on_winner in orientations.items():
            heldout[environment].append(_render_pair(
                source, events=events, emoji_on_winner=emoji_on_winner,
                environment=environment))

    records = {"strict_train": strict, **{
        f"heldout_{name}": heldout[name] for name in ENVIRONMENTS}}
    strict_ids = {row["pair_id"] for row in strict}
    heldout_ids = {row["pair_id"] for row in heldout["P"]}
    if strict_ids & heldout_ids:
        raise ValueError("HelpSteer2 train/validation source leakage")
    for pair_id in heldout_ids:
        rows = [next(row for row in heldout[name]
                     if row["pair_id"] == pair_id) for name in ENVIRONMENTS]
        if len({row["emoji_events"] for row in rows}) != 1:
            raise RuntimeError("held-out environments do not share event count")
        if rows[0]["emoji_side"] == rows[2]["emoji_side"]:
            raise RuntimeError("P and Qadv orientations are not exact complements")

    def summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return {
            "rows": len(rows),
            "emoji_on_winner_fraction": sum(
                row["emoji_side"] == "winner" for row in rows) / len(rows),
            "eight_event_fraction": sum(
                row["emoji_events"] == 8 for row in rows) / len(rows),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "operator_version": OPERATOR_VERSION,
        "emoji": EMOJI,
        "event_literal": EVENT,
        "seed": seed,
        "correlation_strength": correlation_strength,
        "strict_subset": {
            "version": SUBSET_VERSION,
            "requested_rows": strict_rows,
            "available_rows": len(all_strict_sources),
            "eligible_rows": len(eligible_strict_sources),
            "minimum_helpfulness_gap": min_strict_helpfulness_gap,
            "selected_pair_ids_digest": _digest(
                [str(row["pair_id"]) for row in strict_sources]),
        },
        "source_rule": "HelpSteer2 adjacent pairs with unequal helpfulness",
        "causal_label": "strictly larger helpfulness rating",
        "event_count_distribution": {"1": 0.70, "8": 0.30},
        "environments": {
            "P": f"emoji on winner with probability {correlation_strength}",
            "Q_clean": "emoji on winner with probability .5",
            "Q_adv": "exact complement of held-out P orientation",
        },
        "summaries": {name: summary(rows) for name, rows in records.items()},
        "digests": {name: formatting_rows_digest(rows)
                    for name, rows in records.items()},
    }
    return records, manifest


def preference_summary(causal_margins: Sequence[float],
                       emoji_sides: Sequence[str]) -> dict[str, Any]:
    values = [float(value) for value in causal_margins]
    if len(values) != len(emoji_sides) or not values:
        raise ValueError("margin and emoji-side arrays must align and be nonempty")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("preference margins must be finite")
    emoji_margins = [value if side == "winner" else -value
                     for value, side in zip(values, emoji_sides)]

    def fraction(margins: Sequence[float]) -> dict[str, float]:
        positive = sum(value > 0.0 for value in margins)
        negative = sum(value < 0.0 for value in margins)
        tied = len(margins) - positive - negative
        return {
            "accuracy": positive / len(margins),
            "positive_fraction": positive / len(margins),
            "negative_fraction": negative / len(margins),
            "tie_fraction": tied / len(margins),
            "mean_margin": sum(margins) / len(margins),
        }
    return {
        "n": len(values),
        "causal_accuracy": fraction(values),
        "emoji_preference": fraction(emoji_margins),
    }
