#!/usr/bin/env python
"""Validate and aggregate hotel and emoji endpoints."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics

import matplotlib.pyplot as plt


METHOD_LABELS = {"utl": "UTL", "pm": "PM", "ha": "Practical influence"}
DATASETS_BY_TRAINING_MODE = {
    "fresh": {
        "hotels_paper_panel2k_matched_fresh_rho085": "hotels",
        "pku_emoji_paper_a8_panel2k_matched_fresh": "emoji",
    },
    "continued": {
        "hotels_paper_panel2k_matched_continued_rho085": "hotels",
        "pku_emoji_paper_a8_panel2k_matched_continued": "emoji",
    },
    "qwen_fresh": {
        "hotels_qwen3_4b_panel2k_matched_fresh_rho085": "hotels",
        "pku_emoji_qwen3_4b_a8_panel2k_matched_fresh": "emoji",
    },
}
PROTOCOLS = {
    "fresh": {"seeds": [0, 1, 2], "utl_draws": 2},
    "continued": {"seeds": [0, 1, 2], "utl_draws": 2},
    "qwen_fresh": {"seeds": [0], "utl_draws": 1},
}
METRICS = ("P", "Q_clean", "Q_adv", "E_cf")
T95_N3 = 4.302652729911275


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def compact_result(value: dict, path: Path) -> dict:
    energy = value["energy"]
    row = {
        "campaign": value["campaign"], "seed": int(value["seed"]),
        "method": value.get("method"), "draw": value.get("draw"),
        "budget": value.get("budget"), "result_path": str(path.resolve()),
        "result_sha256": digest(path), "config_digest": value["config_digest"],
        "selected_digest": value.get("selected_digest", ""),
        "selected_count": len(value.get("selected_candidate_ids", [])),
        "candidate_pool_rows": value.get("candidate_pool_rows"),
        "candidate_pool_scope": value.get("candidate_pool_scope"),
        "endpoint_initialization": value.get("endpoint_initialization", "strict"),
        "E_cf": float(energy["counterfactual_energy"]),
    }
    for name, score in value["headline"].items():
        row[name] = float(score)
    return row


def collect_local(local_root: Path,
                  datasets: dict[str, str]) -> tuple[list[dict], list[dict], dict]:
    """Collect results produced directly by ``run.py`` in one checkout."""
    endpoint: dict[tuple, dict] = {}
    strict: dict[tuple, dict] = {}
    files = 0
    for campaign in datasets:
        campaign_root = local_root / campaign
        if not campaign_root.is_dir():
            raise FileNotFoundError(
                f"missing local campaign output: {campaign_root}")
        for path in campaign_root.glob(
                "seed*/attempts/*/attempt_*/result.json"):
            files += 1
            if not (path.parent / "success.json").is_file():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("campaign") != campaign:
                raise ValueError(f"campaign mismatch in {path}")
            row = compact_result(value, path)
            if row["method"] is None:
                key = (campaign, row["seed"])
                target = strict
            else:
                key = (campaign, row["seed"], row["method"],
                       int(row["draw"]), int(row["budget"]))
                target = endpoint
            if key in target:
                raise ValueError(f"duplicate local result for {key}")
            target[key] = row
    return list(endpoint.values()), list(strict.values()), {
        "files_scanned": files, "duplicates_verified": 0,
        "unique_endpoint_cells": len(endpoint), "unique_strict_models": len(strict),
    }


def validate(cells: list[dict], strict: list[dict],
             datasets: dict[str, str], protocol: dict) -> dict:
    budgets = [100, 200, 300, 400, 500, 750, 1000, 1500, 2000]
    expected = {(campaign, seed, method, draw, budget)
                for campaign in datasets for seed in protocol["seeds"]
                for budget in budgets for method in METHOD_LABELS
                for draw in (range(protocol["utl_draws"])
                             if method == "utl" else range(1))}
    observed = {(row["campaign"], row["seed"], row["method"],
                 int(row["draw"]), int(row["budget"])) for row in cells}
    if observed != expected:
        raise ValueError(f"cell mismatch: missing={sorted(expected-observed)}, "
                         f"extra={sorted(observed-expected)}")
    strict_keys = {(row["campaign"], row["seed"]) for row in strict}
    expected_strict = {(campaign, seed) for campaign in datasets
                       for seed in protocol["seeds"]}
    if strict_keys != expected_strict:
        raise ValueError("strict-result coverage mismatch")
    repeat_ranges = {}
    for campaign in datasets:
        digests = {row["config_digest"] for row in cells if row["campaign"] == campaign}
        if len(digests) != 1:
            raise ValueError(f"multiple config digests for {campaign}: {digests}")
        for seed in protocol["seeds"]:
            full = [row for row in cells if row["campaign"] == campaign
                    and row["seed"] == seed and row["budget"] == 2000]
            if len({row["selected_digest"] for row in full}) != 1:
                raise ValueError(f"k=2000 selected-set mismatch: {campaign}, seed {seed}")
            for metric in METRICS:
                values = [row.get(metric) for row in full]
                if all(value is None for value in values):
                    continue
                repeat_ranges[f"{datasets[campaign]}:s{seed}:{metric}"] = (
                    max(values) - min(values))
    return {"expected_endpoint_cells": len(expected), "budgets": budgets,
            "seeds": protocol["seeds"], "utl_draws": protocol["utl_draws"],
            "k2000_common_selected_set_check": "passed",
            "k2000_independent_retrain_ranges": repeat_ranges}


def average_draws(cells: list[dict], datasets: dict[str, str],
                  protocol: dict) -> list[dict]:
    output = []
    for campaign, dataset in datasets.items():
        for seed in protocol["seeds"]:
            for budget in sorted({int(row["budget"]) for row in cells}):
                for method, label in METHOD_LABELS.items():
                    selected = [row for row in cells if row["campaign"] == campaign
                                and row["seed"] == seed and row["budget"] == budget
                                and row["method"] == method]
                    expected = protocol["utl_draws"] if method == "utl" else 1
                    if len(selected) != expected:
                        raise ValueError(f"draw count mismatch: {campaign}/{seed}/{method}/{budget}")
                    row = {"dataset": dataset, "campaign": campaign, "seed": seed,
                           "method": label, "budget": budget, "draws": expected}
                    for metric in METRICS:
                        values = [item.get(metric) for item in selected]
                        row[metric] = (sum(values) / len(values)
                                       if all(value is not None for value in values) else None)
                    output.append(row)
    return output


def interval(values: list[float]) -> dict:
    mean = statistics.mean(values)
    if len(values) == 1:
        return {"mean": mean, "sd": 0.0, "se": 0.0,
                "ci95_lower": mean, "ci95_upper": mean}
    sd = statistics.stdev(values)
    half = T95_N3 * sd / math.sqrt(len(values))
    return {"mean": mean, "sd": sd, "se": sd / math.sqrt(len(values)),
            "ci95_lower": mean-half, "ci95_upper": mean+half}


def summarize(seed_rows: list[dict], datasets: dict[str, str]) -> list[dict]:
    rows = []
    for dataset in datasets.values():
        for method in METHOD_LABELS.values():
            for budget in sorted({row["budget"] for row in seed_rows}):
                selected = [row for row in seed_rows if row["dataset"] == dataset
                            and row["method"] == method and row["budget"] == budget]
                row = {"dataset": dataset, "method": method, "budget": budget,
                       "n_seeds": len(selected)}
                for metric in METRICS:
                    values = [item[metric] for item in selected if item[metric] is not None]
                    if values:
                        for suffix, value in interval(values).items():
                            row[f"{metric}_{suffix}"] = value
                rows.append(row)
    return rows


def strict_rows(rows: list[dict],
                datasets: dict[str, str]) -> tuple[list[dict], list[dict]]:
    seeds = []
    for row in rows:
        item = {"dataset": datasets[row["campaign"]], "campaign": row["campaign"],
                "seed": row["seed"]}
        for metric in METRICS:
            item[metric] = row.get(metric)
        seeds.append(item)
    summary = []
    for dataset in datasets.values():
        selected = [row for row in seeds if row["dataset"] == dataset]
        item = {"dataset": dataset, "n_seeds": len(selected)}
        for metric in METRICS:
            values = [row[metric] for row in selected if row[metric] is not None]
            if values:
                for suffix, value in interval(values).items():
                    item[f"{metric}_{suffix}"] = value
        summary.append(item)
    return seeds, summary


def paired_gains(seed_rows: list[dict], datasets: dict[str, str],
                 protocol: dict) -> list[dict]:
    rows = []
    for dataset in datasets.values():
        for budget in sorted({row["budget"] for row in seed_rows}):
            for method in ("PM", "Practical influence"):
                for metric in METRICS:
                    diffs = []
                    for seed in protocol["seeds"]:
                        chosen = next(row for row in seed_rows if row["dataset"] == dataset
                                      and row["budget"] == budget and row["seed"] == seed
                                      and row["method"] == method)
                        utl = next(row for row in seed_rows if row["dataset"] == dataset
                                   and row["budget"] == budget and row["seed"] == seed
                                   and row["method"] == "UTL")
                        if chosen[metric] is not None:
                            diffs.append(chosen[metric] - utl[metric])
                    if diffs:
                        row = {"dataset": dataset, "method": method,
                               "baseline": "UTL", "budget": budget,
                               "metric": metric, "n_seeds": len(diffs),
                               "seed_differences": json.dumps(diffs)}
                        row.update(interval(diffs))
                        rows.append(row)
    return rows


def plot(summary: list[dict], strict: list[dict], output: Path, dataset: str,
         metric: str, ylabel: str, scale: float = 1.0) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    styles = {"UTL": ("#777777", "o"), "PM": ("#0072B2", "s"),
              "Practical influence": ("#D55E00", "^")}
    for method, (color, marker) in styles.items():
        rows = sorted((row for row in summary if row["dataset"] == dataset
                       and row["method"] == method), key=lambda row: row["budget"])
        x = [row["budget"] for row in rows]
        y = [scale*row[f"{metric}_mean"] for row in rows]
        lo = [scale*row[f"{metric}_ci95_lower"] for row in rows]
        hi = [scale*row[f"{metric}_ci95_upper"] for row in rows]
        ax.plot(x, y, marker=marker, color=color, label=method, linewidth=1.8)
        ax.fill_between(x, lo, hi, color=color, alpha=.16, linewidth=0)
    strict_row = next(row for row in strict if row["dataset"] == dataset)
    strict_mean = scale * strict_row[f"{metric}_mean"]
    strict_lo = scale * strict_row[f"{metric}_ci95_lower"]
    strict_hi = scale * strict_row[f"{metric}_ci95_upper"]
    ax.axhline(strict_mean, color="black", linestyle="--", linewidth=1.2,
               label="Strict only")
    ax.axhspan(strict_lo, strict_hi, color="black", alpha=.06, linewidth=0)
    ax.set_xlabel("Tie budget k"); ax.set_ylabel(ylabel)
    ax.grid(alpha=.22); ax.legend(frameon=False); fig.tight_layout()
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"{dataset}_{metric}.{suffix}", dpi=220)
    plt.close(fig)


def clear_figures(output: Path) -> None:
    """Remove only generated image files so reruns cannot retain stale plots."""
    figures = output / "figures"
    if not figures.is_dir():
        return
    for path in figures.iterdir():
        if path.is_file() and path.suffix.lower() in {".pdf", ".png"}:
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-root", type=Path, required=True,
        help="directory containing the campaign output directories from run.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-mode", choices=sorted(DATASETS_BY_TRAINING_MODE),
                        default="fresh")
    args = parser.parse_args()
    datasets = DATASETS_BY_TRAINING_MODE[args.training_mode]
    protocol = PROTOCOLS[args.training_mode]
    cells, strict, provenance = collect_local(args.local_root, datasets)
    source_record = {"local_root": str(args.local_root.resolve())}
    validation = validate(cells, strict, datasets, protocol)
    seeds = average_draws(cells, datasets, protocol)
    summary = summarize(seeds, datasets)
    strict_seed, strict_summary = strict_rows(strict, datasets)
    gains = paired_gains(seeds, datasets, protocol)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    seed_fields = ["dataset", "campaign", "seed", "method", "budget", "draws", *METRICS]
    summary_fields = ["dataset", "method", "budget", "n_seeds", *[
        f"{metric}_{suffix}" for metric in METRICS
        for suffix in ("mean", "sd", "se", "ci95_lower", "ci95_upper")]]
    write_csv(output / "seed_results.csv", seeds, seed_fields)
    write_csv(output / "summary.csv", summary, summary_fields)
    write_csv(output / "strict_seed_results.csv", strict_seed,
              ["dataset", "campaign", "seed", *METRICS])
    write_csv(output / "strict_summary.csv", strict_summary,
              ["dataset", "n_seeds", *[f"{metric}_{suffix}" for metric in METRICS
               for suffix in ("mean", "sd", "se", "ci95_lower", "ci95_upper")]])
    write_csv(output / "paired_gains.csv", gains,
              ["dataset", "method", "baseline", "budget", "metric", "n_seeds",
               "mean", "sd", "se", "ci95_lower", "ci95_upper", "seed_differences"])
    write_csv(output / "cell_index.csv", sorted(cells, key=lambda row: (
        datasets[row["campaign"]], row["seed"], row["budget"], row["method"], row["draw"])),
        ["campaign", "seed", "method", "draw", "budget", "selected_count",
         "selected_digest", "candidate_pool_rows", "candidate_pool_scope",
         "endpoint_initialization", "result_sha256", "result_path"])
    clear_figures(output)
    plot_specs = (
        ("P", "P accuracy (%)", 100.0),
        ("Q_clean", "Q_clean accuracy (%)", 100.0),
        ("Q_adv", "Q_adv accuracy (%)", 100.0),
        ("E_cf", "Counterfactual energy E_cf", 1.0),
    )
    for dataset in datasets.values():
        for metric, ylabel, scale in plot_specs:
            plot(summary, strict_summary, output, dataset, metric, ylabel, scale)
    record = {"schema_version": 1, **source_record,
              "training_mode": args.training_mode,
              "provenance": provenance, "validation": validation,
              "methods": list(METHOD_LABELS.values()), "datasets": datasets,
              "interval": ("95% two-sided Student-t interval across 3 seeds; "
                           "UTL draws averaged within seed"
                           if len(protocol["seeds"]) > 1 else
                           "single seed; no uncertainty interval")}
    (output / "summary.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **provenance, **validation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
