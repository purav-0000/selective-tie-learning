#!/usr/bin/env python
"""Assemble linear theory verification, conditional curves, and budget outcomes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import yaml


COLORS = {"UTL": "#777777", "PM": "#d55e00", "Oracle": "#0072b2"}
LABELS = {"UTL": "UTL", "PM": "PM", "Oracle": "Oracle influence"}


def _load(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _clear_figures(output: Path) -> None:
    """Remove only generated image files so reruns cannot retain stale plots."""
    figures = output / "figures"
    if not figures.is_dir():
        return
    for path in figures.iterdir():
        if path.is_file() and path.suffix.lower() in {".pdf", ".png"}:
            path.unlink()


def _estimate(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    half = (stats.t.ppf(.975, len(array)-1)*array.std(ddof=1)/np.sqrt(len(array))
            if len(array) > 1 else 0.0)
    return float(array.mean()), float(half)


def _theory(root: Path, seeds: list[int], bins: int, output: Path) -> dict:
    individual = []
    moments = []
    for seed in seeds:
        base = root/f"seed_{seed:02d}"
        diagnostic = _load(base/"scores.json.gz")["diagnostics"]["operators"]["resampling"]
        moment = {
            "seed": seed,
            "packet_moment_relative_error": diagnostic["packet_moment_relative_error"],
            "lambda": diagnostic["conditional_influence_lambda"],
            "oracle_residual": diagnostic["oracle_residual"],
        }
        moments.append(moment)
        for packet in ("symmetric", "realized"):
            for path in sorted((base/"theory"/packet).glob("j*.json.gz")):
                value = _load(path)
                individual.append({key: value[key] for key in (
                    "seed", "candidate_index", "packet", "absolute_margin",
                    "oracle_predicted_reduction", "conditional_predicted_reduction",
                    "observed_spurious_energy_reduction", "zero_contrast", "a_wins")})
    for moment in moments:
        one = [r for r in individual if r["seed"] == moment["seed"] and r["packet"] == "symmetric"]
        margins = np.asarray([r["absolute_margin"] for r in one])
        moment.update({
            "margin_mean": float(margins.mean()),
            "margin_sd": float(margins.std(ddof=1)),
            "margin_p90": float(np.quantile(margins, .9)),
            "zero_contrast_count": int(sum(r["zero_contrast"] for r in one)),
        })
    _csv(output/"individual_influence.csv", individual)
    _csv(output/"packet_moments.csv", moments)
    symmetric = [r for r in individual if r["packet"] == "symmetric"]
    predicted = np.asarray([r["oracle_predicted_reduction"] for r in symmetric])
    observed = np.asarray([r["observed_spurious_energy_reduction"] for r in symmetric])
    report = {
        "vector_relative_error": float(np.linalg.norm(predicted-observed)/np.linalg.norm(observed)),
        "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(observed))),
        "packet_moment_relative_error_mean": float(np.mean([
            r["packet_moment_relative_error"] for r in moments])),
        "margins": {
            "mean": float(np.mean([r["absolute_margin"] for r in symmetric])),
            "median": float(np.median([r["absolute_margin"] for r in symmetric])),
            "p90": float(np.quantile([r["absolute_margin"] for r in symmetric], .9)),
        },
        "zero_contrast_fraction": float(np.mean([r["zero_contrast"] for r in symmetric])),
    }
    lo, hi = min(predicted.min(), observed.min()), max(predicted.max(), observed.max())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(predicted, observed, s=6, alpha=.25, rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set(xlabel="Oracle predicted reduction", ylabel="Observed symmetric-packet reduction",
           title="Single-packet influence at weight 1/N")
    ax.grid(alpha=.2)
    _save(fig, output/"figures"/"influence_identity")

    curves = []
    for seed in seeds:
        anchor = sorted((r for r in symmetric if r["seed"] == seed),
                        key=lambda r: r["absolute_margin"])
        by_mode = {(r["seed"], r["candidate_index"], r["packet"]): r for r in individual}
        for number, members in enumerate(np.array_split(anchor, bins), start=1):
            for packet in ("symmetric", "realized"):
                actual = [by_mode[(seed, r["candidate_index"], packet)][
                    "observed_spurious_energy_reduction"] for r in members]
                oracle = [r["oracle_predicted_reduction"] for r in members]
                curves.append({
                    "seed": seed, "bin": number, "packet": packet,
                    "count": len(members),
                    "margin_mean": float(np.mean([r["absolute_margin"] for r in members])),
                    "observed_mean": float(np.mean(actual)),
                    "observed_sd_individual": float(np.std(actual, ddof=1)),
                    "oracle_mean": float(np.mean(oracle)),
                    "oracle_sd_individual": float(np.std(oracle, ddof=1)),
                    "analytic_conditional_mean": float(np.mean([
                        r["conditional_predicted_reduction"] for r in members])),
                })
    _csv(output/"expected_influence_bins_by_seed.csv", curves)
    aggregate = []
    for packet in ("symmetric", "realized"):
        for number in range(1, bins+1):
            members = [r for r in curves if r["packet"] == packet and r["bin"] == number]
            row = {"packet": packet, "bin": number}
            for key in ("margin_mean", "observed_mean", "oracle_mean", "analytic_conditional_mean",
                        "observed_sd_individual", "oracle_sd_individual"):
                row[key], row[key+"_ci95"] = _estimate([r[key] for r in members])
            aggregate.append(row)
    _csv(output/"expected_influence_bins_aggregate.csv", aggregate)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    members = sorted((r for r in aggregate if r["packet"] == "realized"),
                     key=lambda r: r["bin"])
    x = np.array([r["margin_mean"] for r in members])
    for key, label, color in (
        ("observed_mean", "Observed", "#d55e00"),
        ("oracle_mean", "Mean oracle A", "#0072b2"),
        ("analytic_conditional_mean", "Gaussian curve", "#444444")):
        y = np.array([r[key] for r in members])
        half = np.array([r[key+"_ci95"] for r in members])
        ax.plot(x, y, marker="o", ms=3, label=label, color=color)
        ax.fill_between(x, y-half, y+half, alpha=.12, color=color)
    for seed in seeds:
        one = sorted((r for r in curves
                      if r["seed"] == seed and r["packet"] == "realized"),
                     key=lambda r: r["bin"])
        ax.plot([r["margin_mean"] for r in one],
                [r["observed_mean"] for r in one],
                color="#d55e00", alpha=.16, linewidth=.7)
    ax.set(xlabel="Absolute strict margin",
           ylabel="Population spurious-energy reduction",
           title="Expected influence for realized tie orientations")
    ax.grid(alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, output/"figures"/"expected_influence")
    return report


def _budgets(root: Path, seeds: list[int], output: Path) -> dict:
    endpoint = []
    strict = []
    for seed in seeds:
        foundation = _load(root/f"seed_{seed:02d}"/"foundation.json.gz")
        m = foundation["strict_metrics"]
        strict.append({"seed": seed, **{f"accuracy_{k}": v for k,v in m["accuracy"].items()},
                       **{f"energy_{k}": v for k,v in m["energy"].items() if isinstance(v, (int,float))}})
        for path in (root/f"seed_{seed:02d}"/"resampling").glob("*/*.json.gz"):
            value = _load(path)
            m = value["metrics"]
            endpoint.append({
                "seed": seed, "method": value["method"], "draw": value["draw"],
                "budget": value["budget"],
                **{f"accuracy_{k}": v for k,v in m["accuracy"].items()},
                **{f"energy_{k}": v for k,v in m["energy"].items() if isinstance(v,(int,float))},
            })
    _csv(output/"endpoints.csv", endpoint)
    _csv(output/"strict.csv", strict)
    grouped = defaultdict(list)
    for row in endpoint:
        grouped[(row["seed"],row["method"],row["budget"])].append(row)
    per_seed = []
    keys = [k for k in endpoint[0] if k.startswith(("accuracy_", "energy_"))]
    for (seed, method, budget), members in sorted(grouped.items()):
        per_seed.append({"seed": seed, "method": method, "budget": budget,
                         **{k: float(np.mean([r[k] for r in members])) for k in keys}})
    _csv(output/"seed_method_budget.csv", per_seed)
    grouped = defaultdict(list)
    for row in per_seed:
        grouped[(row["method"],row["budget"])].append(row)
    summary = []
    for (method,budget), members in sorted(grouped.items()):
        result = {"method": method,"budget":budget}
        for key in keys:
            result[key+"_mean"], result[key+"_ci95"] = _estimate([r[key] for r in members])
        summary.append(result)
    _csv(output/"summary.csv", summary)
    specifications = (
        ("accuracy_P", "Population expected BT accuracy on P (%)",
         "P accuracy by budget (95% CI across seeds)"),
        ("accuracy_Q_clean", "Population expected BT accuracy on Q_clean (%)",
         "Q_clean accuracy by budget (95% CI across seeds)"),
        ("accuracy_Q_mixed", "Population expected BT accuracy on Q_adv (%)",
         "Q_adv accuracy by budget (95% CI across seeds)"),
        ("energy_spurious_energy", "Population spurious energy E",
         "Budget versus spurious energy (95% CI across seeds)"),
    )
    for metric, ylabel, title in specifications:
        fig, ax = plt.subplots(figsize=(6.2,4.2))
        multiplier = 100 if metric.startswith("accuracy_") else 1
        for method in ("UTL", "PM", "Oracle"):
            rows = sorted((r for r in summary if r["method"] == method), key=lambda r:r["budget"])
            x = [r["budget"] for r in rows]
            y = multiplier*np.array([r[metric+"_mean"] for r in rows])
            half = multiplier*np.array([r[metric+"_ci95"] for r in rows])
            ax.plot(x,y,marker="o",markersize=3,label=LABELS[method],color=COLORS[method])
            ax.fill_between(x,y-half,y+half,color=COLORS[method],alpha=.13)
        ax.axhline(multiplier*np.mean([r[metric] for r in strict]),color="black",
                   linestyle="--",label="Strict only")
        ax.set(xlabel="Tie budget", ylabel=ylabel, title=title)
        ax.grid(alpha=.2)
        ax.legend(frameon=False,fontsize=8)
        _save(fig, output/"figures"/metric)
    return {"endpoint_cells":len(endpoint), "strict_seeds":len(strict)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--root",type=Path)
    parser.add_argument("--seeds",type=int,nargs="*")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = args.root or Path(config["output_root"])
    output = root/"aggregate"
    seeds = args.seeds if args.seeds else list(map(int,config["seeds"]))
    _clear_figures(output)
    theory = _theory(root,seeds,int(config["theory_margin_bins"]),output)
    budgets = _budgets(root,seeds,output)
    result = {"theory":theory,"budgets":budgets}
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
