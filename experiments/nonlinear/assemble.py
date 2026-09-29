#!/usr/bin/env python
"""Assemble MLP endpoint metrics without loading model weights."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stl.nonlinear.campaign import config_digest, config_from_dict  # noqa: E402


COLORS = {"UTL": "#777777", "PM": "#d55e00", "Influence": "#0072b2"}
LABELS = {"UTL": "UTL", "PM": "PM", "Influence": "Practical influence"}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({name for row in rows for name in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metrics(value: dict) -> dict[str, float]:
    accuracy = {f"accuracy_{name}": float(score)
                for name, score in value["accuracy"].items()}
    energy = {"counterfactual_energy": float(
        value["energy"]["counterfactual_energy"])}
    return {**accuracy, **energy}


def _per_seed(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["operator"], row["method"], row["budget"])].append(row)
    output = []
    for (seed, operator, method, budget), values in sorted(grouped.items()):
        numeric = [name for name, value in values[0].items()
                   if isinstance(value, (int, float))
                   and name not in ("seed", "budget", "draw")]
        output.append({"seed": seed, "operator": operator, "method": method,
                       "budget": budget, "draws": len(values),
                       **{name: float(np.mean([v[name] for v in values]))
                          for name in numeric}})
    return output


def _summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["operator"], row["method"], row["budget"])].append(row)
    output = []
    for (operator, method, budget), values in sorted(grouped.items()):
        metrics = [name for name in values[0]
                   if name.startswith("accuracy_")
                   or name == "counterfactual_energy"]
        row = {"operator": operator, "method": method, "budget": budget,
               "seeds": len(values)}
        for name in metrics:
            samples = np.asarray([value[name] for value in values], dtype=float)
            row[f"{name}_mean"] = float(samples.mean())
            row[f"{name}_ci95"] = (float(t.ppf(.975, len(samples)-1)
                                         * samples.std(ddof=1) / math.sqrt(len(samples)))
                                     if len(samples) > 1 else 0.0)
        output.append(row)
    return output


def _paired_gains(rows: list[dict]) -> list[dict]:
    """Pair each targeted endpoint with its seed/budget UTL mean."""
    utl = {(row["seed"], row["operator"], row["budget"]): row
           for row in rows if row["method"] == "UTL"}
    output = []
    for row in rows:
        if row["method"] == "UTL":
            continue
        baseline = utl[(row["seed"], row["operator"], row["budget"])]
        output.append({
            "seed": row["seed"], "operator": row["operator"],
            "method": row["method"], "budget": row["budget"],
            "accuracy_Q_direct_gain": (
                row["accuracy_Q_direct"]-baseline["accuracy_Q_direct"]),
            "accuracy_Q_mixed_gain": (
                row["accuracy_Q_mixed"]-baseline["accuracy_Q_mixed"]),
        })
    return output


def _save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(figure)


def _clear_figures(output: Path) -> None:
    """Remove only generated image files so reruns cannot retain stale plots."""
    figures = output / "figures"
    if not figures.is_dir():
        return
    for path in figures.iterdir():
        if path.is_file() and path.suffix.lower() in {".pdf", ".png"}:
            path.unlink()


def _plots(output: Path, summary: list[dict], strict: list[dict]) -> None:
    seed_count = max(int(row["seeds"]) for row in summary)
    uncertainty_label = ("95% CI across seeds" if seed_count > 1
                         else "single-seed check")
    specifications = (
        ("accuracy_P", "Population expected BT accuracy on P (%)",
         f"P accuracy by budget ({uncertainty_label})", 100.0),
        ("accuracy_Q_clean", "Population expected BT accuracy on Q_clean (%)",
         f"Q_clean accuracy by budget ({uncertainty_label})", 100.0),
        ("accuracy_Q_mixed", "Population expected BT accuracy on Q_adv (%)",
         f"Q_adv accuracy by budget ({uncertainty_label})", 100.0),
        ("counterfactual_energy", "Held-out counterfactual energy E_cf",
         f"Budget versus E_cf ({uncertainty_label})", 1.0),
    )
    for metric, ylabel, title, scale in specifications:
        figure, axis = plt.subplots(figsize=(6.4, 4.3))
        methods = list(dict.fromkeys(row["method"] for row in summary))
        for method in methods:
            rows = sorted((row for row in summary if row["method"] == method),
                          key=lambda row: row["budget"])
            x = np.asarray([row["budget"] for row in rows])
            mean = scale*np.asarray([row[f"{metric}_mean"] for row in rows])
            half = scale*np.asarray([row[f"{metric}_ci95"] for row in rows])
            axis.plot(x, mean, marker="o", markersize=3.5, linewidth=1.8,
                      color=COLORS[method], label=LABELS[method])
            axis.fill_between(x, mean-half, mean+half,
                              color=COLORS[method], alpha=.14, linewidth=0)
        strict_values = scale*np.asarray([row[metric] for row in strict])
        strict_mean = float(strict_values.mean())
        strict_half = (float(t.ppf(.975, len(strict_values)-1)
                             * strict_values.std(ddof=1)/math.sqrt(len(strict_values)))
                       if len(strict_values) > 1 else 0.0)
        axis.axhline(strict_mean, color="black", linestyle="--", linewidth=1.2,
                     label="Strict only")
        axis.axhspan(strict_mean-strict_half, strict_mean+strict_half,
                    color="black", alpha=.06, linewidth=0)
        axis.set(xlabel="Tie budget", ylabel=ylabel, title=title)
        axis.grid(alpha=.2)
        axis.legend(frameon=False, fontsize=8)
        _save_figure(figure, output/"figures"/metric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument(
        "--foundation-root", type=Path,
        help="source of immutable fresh foundations when endpoints use continuation",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw.pop("name", None)
    raw.pop("endpoint_initialization", None)
    output_root = raw.pop("output_root", "results/nonlinear/paper_aligned")
    widths = tuple(raw.pop("widths"))
    if args.width not in widths:
        raise ValueError(f"width {args.width} is not present in the configuration")
    cfg = config_from_dict({**raw, "width": args.width})
    root = args.root or ROOT / output_root / f"w{args.width}"
    foundation_root = args.foundation_root or root
    output = args.output or root / "aggregate"

    selected_seeds = list(cfg.seeds if args.seeds is None else args.seeds)
    unknown = sorted(set(selected_seeds) - set(cfg.seeds))
    if unknown:
        raise ValueError(f"seeds not present in the configuration: {unknown}")
    strict, endpoints = [], []
    for seed in selected_seeds:
        foundation_path = foundation_root / "foundations" / f"seed_{seed:02d}.pt"
        if not foundation_path.is_file():
            continue
        foundation = torch.load(foundation_path, map_location="cpu", weights_only=False)
        if foundation["config_digest"] != config_digest(cfg):
            raise ValueError(f"configuration mismatch: {foundation_path}")
        strict.append({"seed": seed, **_metrics(foundation["strict_diagnostics"])})
    for path in root.glob("endpoints/resampling/seed_*/*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if int(value["seed"]) not in selected_seeds:
            continue
        row = (
            {key: value[key] for key in (
                "seed", "operator", "method", "draw", "budget",
                "runtime_seconds")}
            | _metrics(value["metrics"])
        )
        if value["budget"] == cfg.n_candidates:
            for method in cfg.methods:
                endpoints.append({**row, "method": method,
                                  "draw": "shared"})
        else:
            endpoints.append(row)
    if not strict or not endpoints:
        raise FileNotFoundError("no foundations or endpoints found")
    per_seed = _per_seed(endpoints)
    summary = _summary(per_seed)
    gains = _paired_gains(per_seed) if "UTL" in cfg.methods else []
    gain_summary = _summary(gains) if gains else []
    for filename, rows in (("strict.csv", strict), ("endpoints.csv", endpoints),
                           ("seed_methods.csv", per_seed), ("summary.csv", summary),
                           ):
        _write_csv(output / filename, rows)
    _clear_figures(output)
    _plots(output, summary, strict)
    if gains:
        _write_csv(output / "paired_gains.csv", gains)
        _write_csv(output / "gain_summary.csv", gain_summary)
    print(json.dumps({"strict_seeds": len(strict), "endpoints": len(endpoints),
                      "summary_rows": len(summary), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
