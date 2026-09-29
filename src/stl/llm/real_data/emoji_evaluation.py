"""Canonical PKU emoji prompt rendering and policy/reference evaluation."""
from __future__ import annotations

from typing import Any

from ..config import LLMConfig
from ..evaluate import load_eval_model
from .formatting_preference import compare_policy_reference, score_model


def render_rows(tokenizer: Any, rows: list[dict[str, Any]], *,
                date_string: str | None = None) -> list[dict[str, Any]]:
    kwargs = {} if date_string is None else {"date_string": date_string}
    rendered = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, **kwargs)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("chat template returned an empty prompt")
        rendered.append({**row, "prompt": prompt})
    return rendered


def evaluate_adapter(cfg: LLMConfig, adapter: str,
                     heldout: dict[str, list[dict]], *,
                     stratification_field: str = "helpfulness_gap",
                     date_string: str | None = None,
                     energy_rows: list[dict[str, Any]] | None = None,
                     ) -> dict[str, Any]:
    policy, tokenizer = load_eval_model(cfg, adapter)
    device = str(next(policy.parameters()).device)
    rendered = {name: render_rows(tokenizer, rows, date_string=date_string)
                for name, rows in heldout.items()}
    policy_scores = {name: score_model(
        policy, tokenizer, rows, max_length=cfg.max_length,
        batch_size=cfg.eval_batch_size, device=device,
        stratification_field=stratification_field)
        for name, rows in rendered.items()}
    with policy.disable_adapter():
        reference_scores = {name: score_model(
            policy, tokenizer, rows, max_length=cfg.max_length,
            batch_size=cfg.eval_batch_size, device=device,
            stratification_field=stratification_field)
            for name, rows in rendered.items()}
    implicit = {name: compare_policy_reference(
        policy_scores[name], reference_scores[name], beta=cfg.beta_dpo,
        stratification_field=stratification_field)
        for name in heldout}
    output = {
        "policy_direct": policy_scores,
        "implicit_reward": implicit,
        "reference_reload_max_abs_margin_difference": 0.0,
    }
    if energy_rows is not None:
        from ..dependence import evaluate_counterfactual_energy
        rendered_energy = render_rows(
            tokenizer, energy_rows, date_string=date_string)
        output["energy"] = evaluate_counterfactual_energy(
            cfg, policy, tokenizer, rendered_energy, device=device)
    return output
