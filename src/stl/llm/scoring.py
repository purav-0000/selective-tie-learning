"""Counterfactual PM and Practical-influence scores over LoRA parameters.

Computed at the strict-model checkpoint:

  PM_j = | beta [ (logpi_pol(a) - logpi_pol(b)) - (logpi_ref(a) - logpi_ref(b)) ] |
         the implicit DPO reward gap.
  A_j = < v_s, (G+lambda I)^{-1} g_j >, where g_j is the exact symmetric
            packet gradient divided by N, v_s = grad_theta E[m^2] over N_v
            probe anchors, and G is the strict-data generalized Gauss--Newton
            matrix over LoRA parameters. The low-rank system is solved exactly
            in data space with Woodbury.

Influence scores are stored signed and ranked in descending order.

The reference is the adapter-disabled policy. Practical influence uses an
exact Woodbury solve in the sampled GGN row space.
"""
from __future__ import annotations

import json

from .config import LLMConfig
from .loglik import forward_logprob_sums
from .tokenization import encode_conditional


def load_policy(cfg: LLMConfig, lora_path: str, device: str = "cuda:0"):
    """Load a trainable adapter and its frozen base reference."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    from .dpo import load_tokenizer

    dev_map = {"": int(device.split(":")[1])} if ":" in device else device
    tok = load_tokenizer(cfg)

    try:
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model, revision=cfg.base_model_revision,
            device_map=dev_map, dtype=torch.bfloat16,
            attn_implementation=cfg.attn_implementation)
    except (ImportError, ValueError) as error:
        raise RuntimeError(
            f"required attention backend {cfg.attn_implementation!r} is "
            "unavailable during scoring") from error

    policy = PeftModel.from_pretrained(base, lora_path, is_trainable=True)
    policy.eval()                         # disable dropout -> deterministic grads
    from .loglik import enable_scoring_layer_checkpointing
    checkpointed = enable_scoring_layer_checkpointing(policy)
    print(f"[score] eval-mode activation checkpointing: {checkpointed} decoder blocks", flush=True)
    return policy, tok


def _lora_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _flatten(tensors):
    import torch
    return torch.cat([t.reshape(-1) for t in tensors])


def _pair_margin(policy, tok, prompt, resp_a, resp_b, beta, device, max_length):
    """Differentiable implicit-reward margin m for one (a, b) candidate."""
    import torch
    encoded = [encode_conditional(tok, prompt, r, max_length)
               for r in (resp_a, resp_b)]
    pad_id = tok.pad_token_id
    with torch.no_grad(), policy.disable_adapter():
        rp = forward_logprob_sums(policy, encoded, device, pad_id)
    # Reference first: do not keep policy activations alive during its forward.
    lp = forward_logprob_sums(policy, encoded, device, pad_id)
    return beta * ((lp[0] - lp[1]) - (rp[0] - rp[1]))


def _probe_direction(policy, tok, val_anchors, beta, device, max_length,
                     batch_size: int = 1):
    """Probe direction ``v_s = grad mean(m^2)``, plus margin diagnostics.

    The mean signed margin diagnoses systematic original-versus-transformed
    preference. Counterfactual energy itself uses the squared margin and is
    invariant to packet orientation.
    """
    import torch
    params = _lora_params(policy)
    accum = [torch.zeros_like(p) for p in params]
    margins = []
    if batch_size < 1:
        raise ValueError("probe batch_size must be positive")
    pad_id = tok.pad_token_id
    for lo in range(0, len(val_anchors), batch_size):
        batch = val_anchors[lo:lo + batch_size]
        encoded = []
        for anchor in batch:
            encoded.extend([
                encode_conditional(tok, anchor["prompt"],
                                   anchor["response_a"], max_length),
                encode_conditional(tok, anchor["prompt"],
                                   anchor["response_b"], max_length),
            ])
        with torch.no_grad(), policy.disable_adapter():
            rp = forward_logprob_sums(policy, encoded, device, pad_id)
        lp = forward_logprob_sums(policy, encoded, device, pad_id)
        batch_margins = beta * (
            (lp[0::2] - lp[1::2]) - (rp[0::2] - rp[1::2]))
        grads = torch.autograd.grad(batch_margins.square().sum(), params)
        for acc, g in zip(accum, grads):
            acc.add_(g.detach())
        margins.extend(batch_margins.detach().float().cpu().tolist())
    v = _flatten([a / max(len(val_anchors), 1) for a in accum])
    n = max(len(margins), 1)
    stats = {
        "n_anchors": len(margins),
        "mean_margin": sum(margins) / n,
        "mean_abs_margin": sum(abs(m) for m in margins) / n,
        "frac_positive_margin": sum(m > 0 for m in margins) / n,
        "mean_square_margin": sum(m * m for m in margins) / n,
    }
    stats["gradient_norm"] = float(v.float().norm().item())
    return v.detach(), stats


def _packet_loss(margin, cand, *, symmetric_packet: bool):
    """Return the protocol-matched logical packet loss.

    Candidate construction fixes one orientation. Realized
    experiments score that loss; symmetric experiments instead score the
    exact average of both orientations, making the gradient independent of
    the fixed winner.
    """
    import torch.nn.functional as F
    if symmetric_packet:
        return 0.5 * (F.softplus(-margin) + F.softplus(margin))
    chosen = cand.get("metadata", {}).get("tie_chosen", cand.get("chosen"))
    if chosen is None:
        raise ValueError("candidate does not declare its realized tie winner")
    if chosen == cand["response_a"]:
        return F.softplus(-margin)
    if chosen == cand["response_b"]:
        return F.softplus(margin)
    raise ValueError("tie winner is not one of the candidate responses")


def _packet_gradient(policy, tok, cand, beta, device, max_length, *,
                     symmetric_packet: bool = False):
    """Gradient of the protocol-matched logical packet plus A-minus-B margin."""
    import torch
    params = _lora_params(policy)
    m = _pair_margin(policy, tok, cand["prompt"], cand["response_a"],
                     cand["response_b"], beta, device, max_length)
    L_Q = _packet_loss(m, cand, symmetric_packet=symmetric_packet)
    g = _flatten([g.detach() for g in torch.autograd.grad(L_Q, params)])
    return g, m.item()


class LoRAGGN:
    """Exact strict-data DPO GGN in the sampled-row subspace.

    A row is ``sqrt(sigmoid(m)sigmoid(-m)) * grad(m)``. This differs from an
    empirical Fisher row, which would be the gradient of the realized loss
    and would square an outcome-dependent first derivative. The transient row
    matrix is stored on CPU when it does not fit safely beside the model.
    """

    def __init__(self, policy, tok, strict_subsample, beta, device,
                 max_length, damping_fraction, dtype):
        import torch
        self.damping_fraction = damping_fraction
        self.damping = None
        self.dtype, self.N = dtype, len(strict_subsample)
        params = _lora_params(policy)
        d_lora = sum(p.numel() for p in params)
        bytes_per_value = torch.empty((), dtype=dtype).element_size()
        gb = self.N * d_lora * bytes_per_value / 1e9
        storage_device = "cpu" if gb > 10.0 else device
        print(f"  [GGN] {self.N} rows, d_LoRA={d_lora}, ~{gb:.2f} GB, "
              f"storage={storage_device}")
        G = torch.zeros((self.N, d_lora), dtype=dtype, device=storage_device)
        raw_norms = torch.zeros(self.N, dtype=torch.float64)
        raw_max_abs = 0.0
        nonfinite = 0
        for i, ex in enumerate(strict_subsample):
            m = _pair_margin(policy, tok, ex["prompt"], ex["chosen"],
                             ex["rejected"], beta, device, max_length)
            curvature = (torch.sigmoid(m) * torch.sigmoid(-m)).detach()
            grads = torch.autograd.grad(m, params)
            flat = _flatten([g.detach() for g in grads]).float()
            flat.mul_(curvature.sqrt().float())
            finite = torch.isfinite(flat)
            nonfinite += int((~finite).sum().item())
            if not bool(finite.all()):
                raise FloatingPointError(
                    f"strict GGN row {i} contains non-finite values")
            raw_norms[i] = flat.double().norm().cpu()
            raw_max_abs = max(raw_max_abs, float(flat.abs().max().item()))
            G[i] = flat.to(dtype)
        self.G = G
        self._gram64 = None
        self.raw_norms = raw_norms
        self.raw_max_abs = raw_max_abs
        self.nonfinite = nonfinite

    def __call__(self, v):
        """Stable damped-GGN product with compensated chunk accumulation."""
        import torch
        scores = torch.zeros(self.N, dtype=torch.float64, device=v.device)
        chunk = 16_384
        for start in range(0, v.numel(), chunk):
            stop = min(start + chunk, v.numel())
            block = self.G[:, start:stop].to(v.device, dtype=torch.float32)
            scores.add_((block @ v[start:stop].float()).double())
        out = torch.empty_like(v, dtype=torch.float32)
        reconstruction_chunk = 262_144
        for start in range(0, v.numel(), reconstruction_chunk):
            stop = min(start + reconstruction_chunk, v.numel())
            block = self.G[:, start:stop].to(v.device, dtype=torch.float32)
            out[start:stop] = block.T @ scores.float() / self.N
        if self.damping is None:
            raise RuntimeError("solve must determine scale-aware damping first")
        return out + self.damping * v.float()

    def solve(self, v, tol):
        """Solve the damped GGN system exactly in the N-dimensional data space.

        For F = G^T G / N + lambda I, Woodbury gives
          F^-1 v = lambda^-1 [v - G^T (G G^T + N lambda I)^-1 G v].
        Short matrix products run in fp32 and their partials accumulate in
        fp64; the small solve, refinement, eigensystem, and reconstruction use
        fp64. This is both more stable and cheaper than parameter-space CG
        because the sampled row count is tiny relative to the LoRA dimension.
        """
        import torch
        v = v.float()
        gram64 = (torch.zeros((self.N, self.N), dtype=torch.float64,
                              device=v.device) if self._gram64 is None
                  else self._gram64.to(v.device))
        gv64 = torch.zeros(self.N, dtype=torch.float64, device=v.device)
        # Short products bound fp32 dot-product error; summing their partials
        # in fp64 is important when strict-DPO gradient norms span orders of
        # magnitude. G itself may be fp32 in the high-accuracy campaign.
        chunk = 16_384
        for start in range(0, v.numel(), chunk):
            stop = min(start + chunk, v.numel())
            block = self.G[:, start:stop].to(v.device, dtype=torch.float32)
            if self._gram64 is None:
                gram64.add_((block @ block.T).double())
            gv64.add_((block @ v[start:stop]).double())

        gram64 = gram64.cpu()
        self._gram64 = gram64
        gv64 = gv64.cpu()
        eig = torch.linalg.eigvalsh(gram64) / self.N
        lambda_max = max(float(eig[-1].item()), 1e-12)
        self.damping = self.damping_fraction * lambda_max
        eye = torch.eye(self.N, dtype=torch.float64)
        system = gram64 + (self.N * self.damping) * eye
        factor = torch.linalg.cholesky(system)
        coeff = torch.cholesky_solve(gv64[:, None], factor).squeeze(1)
        for _ in range(3):
            system_residual = gv64 - system @ coeff
            if system_residual.norm() <= 1e-12 * (gv64.norm() + 1e-30):
                break
            coeff += torch.cholesky_solve(
                system_residual[:, None], factor).squeeze(1)
        threshold = max(float(eig[-1].item()) * 1e-10, 1e-14)
        positive = eig[eig > threshold]

        coeff_device = coeff.to(v.device)
        x = torch.empty_like(v, dtype=torch.float32)
        reconstruction_chunk = 65_536
        for start in range(0, v.numel(), reconstruction_chunk):
            stop = min(start + reconstruction_chunk, v.numel())
            block = self.G[:, start:stop].to(v.device, dtype=torch.float64)
            solved = ((v[start:stop].double() - block.T @ coeff_device)
                      / self.damping)
            x[start:stop] = solved.float()

        residual = self(x) - v
        relative_residual = float(
            (residual.double().norm() / (v.double().norm() + 1e-30)).item())
        projection_sq = float((gv64 @ torch.linalg.pinv(gram64) @ gv64).item())
        diagnostics = {
            "solver": "woodbury",
            "operator": "generalized_gauss_newton",
            "ggn_storage_dtype": str(self.dtype),
            "damping": self.damping,
            "damping_fraction_of_lambda_max": self.damping_fraction,
            "system_relative_residual": float(
                ((system @ coeff - gv64).norm() /
                 (gv64.norm() + 1e-30)).item()),
            "relative_residual": relative_residual,
            "converged": bool(relative_residual < tol),
            "effective_rank": int(positive.numel()),
            "min_positive_eigenvalue": (
                float(positive[0].item()) if positive.numel() else 0.0),
            "max_eigenvalue": float(eig[-1].item()),
            "damped_condition_estimate": float(
                (eig[-1].item() + self.damping) / self.damping),
            "probe_rowspace_fraction": float(max(0.0, min(
                1.0, projection_sq / (v.double().square().sum().item() + 1e-30)))),
            "raw_gradient_norm_min": float(self.raw_norms.min().item()),
            "raw_gradient_norm_median": float(self.raw_norms.median().item()),
            "raw_gradient_norm_max": float(self.raw_norms.max().item()),
            "raw_gradient_max_abs": self.raw_max_abs,
            "nonfinite_gradient_values": self.nonfinite,
        }
        return x.detach(), diagnostics

def _load_ggn_subsample(cfg: LLMConfig, strict_file: str) -> list[dict]:
    """Select the seed-fixed strict rows used by GGN scoring."""
    import torch

    with open(strict_file, encoding="utf-8") as handle:
        strict_rows = [json.loads(line) for line in handle]
    generator = torch.Generator().manual_seed(cfg.seed + 2)
    indices = torch.randperm(
        len(strict_rows), generator=generator)[:cfg.N_ggn].tolist()
    return [strict_rows[index] for index in indices]
