#!/usr/bin/env python
"""Build and score several immutable width/seed foundations sequentially."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"src"))

from stl.nonlinear.campaign import (  # noqa: E402
    atomic_torch_save, build_foundation, config_digest, config_from_dict,
    implementation_digest, score_foundation,
)


def load(path: Path, width: int, device: str):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.pop("name", None)
    raw.pop("endpoint_initialization", None)
    raw.pop("output_root", None)
    widths = tuple(raw.pop("widths"))
    if width not in widths:
        raise ValueError(f"width {width} is not present in the configuration")
    raw.update({"width": width, "device": device})
    return config_from_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cells", nargs="+", required=True,
                        help="WIDTH:SEED cells")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path,
                        help="width-specific output directory")
    args = parser.parse_args()
    experiment = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for text in args.cells:
        width, seed = map(int, text.split(":"))
        tag = f"w{width}:s{seed}"
        print(f"[cells] launching {tag}", flush=True)
        cfg = load(args.config, width, args.device)
        root = (args.output_root if args.output_root is not None else
                ROOT/experiment.get(
                    "output_root", "results/nonlinear/paper_aligned")/f"w{width}")
        foundation_path = root/"foundations"/f"seed_{seed:02d}.pt"
        score_path = root/"scores"/f"seed_{seed:02d}.pt"
        if foundation_path.exists():
            foundation = torch.load(
                foundation_path, map_location="cpu", weights_only=False)
            if (foundation.get("config_digest") != config_digest(cfg)
                    or foundation.get("implementation_digest")
                    != implementation_digest()):
                raise RuntimeError(
                    f"{foundation_path} does not match the active experiment")
        else:
            foundation = build_foundation(cfg, seed)
            atomic_torch_save(foundation, foundation_path)
        if score_path.exists():
            scores = torch.load(score_path, map_location="cpu", weights_only=False)
            if (scores.get("config_digest") != config_digest(cfg)
                    or scores.get("implementation_digest")
                    != foundation["implementation_digest"]
                    or scores.get("strict_digest") != foundation["strict_digest"]):
                raise RuntimeError(
                    f"{score_path} does not match its active foundation")
        else:
            scores = score_foundation(cfg, foundation)
            atomic_torch_save(scores, score_path)
        print(json.dumps({
            "cell": text, "state": "completed",
            "foundation_seconds": foundation["runtime_seconds"],
            "scoring_seconds": scores["runtime_seconds"],
            "worst_residual": max(
                item["relative_residual"]
                for item in scores["diagnostics"]["curvature"]
                    ["systems"].values()),
        }), flush=True)
        print(f"[cell] completed {tag}", flush=True)


if __name__ == "__main__":
    main()
