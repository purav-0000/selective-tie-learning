"""Finite empirical banks and matched counterfactual packet construction."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .dgp import GenericNonlinearDGP, LatentPairBatch, PreferenceBatch


@dataclass(frozen=True)
class PreferenceLatents:
    """Strict preferences together with their causal and nuisance latents."""
    strict: PreferenceBatch
    c_w: torch.Tensor
    s_w: torch.Tensor
    q_w: torch.Tensor
    c_l: torch.Tensor
    s_l: torch.Tensor
    q_l: torch.Tensor

    @property
    def nuisance_bank(self) -> torch.Tensor:
        """Both sides of every strict pair, never winners alone."""
        return torch.cat((self.s_w, self.s_l), dim=0)


@dataclass(frozen=True)
class PacketPool:
    """Candidate feature pairs and their fixed endpoint orientations."""
    first: torch.Tensor
    second: torch.Tensor
    anchor_c: torch.Tensor
    donor_index: torch.Tensor
    a_wins: torch.Tensor
    source_indices: torch.Tensor
    nuisance_bank: torch.Tensor


@dataclass(frozen=True)
class ProbeBank:
    """Held-out winner-preserving pairs used for counterfactual energy."""
    resampling_first: torch.Tensor
    resampling_second: torch.Tensor
    resampling_redraws: torch.Tensor
    anchor_c: torch.Tensor
    nuisance_bank: torch.Tensor
    donor_indices: torch.Tensor


def sample_preference_latents(
    dgp: GenericNonlinearDGP, n: int, *, seed: int,
) -> PreferenceLatents:
    """Sample hard-BT pairs while preserving both alternatives' latents."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    q_a, c_a, s_a = dgp._sample_option(n, "P", generator)
    q_b, c_b, s_b = dgp._sample_option(n, "P", generator)
    phi_a, phi_b = dgp._features(c_a, s_a), dgp._features(c_b, s_b)
    probability = torch.sigmoid(dgp.beta_teacher * (q_a-q_b))
    a_wins = torch.rand(n, generator=generator) < probability
    mask = a_wins[:, None]
    device_mask = mask.to(dgp.device)

    def orient(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_mask = device_mask if a.device == dgp.device else mask.to(a.device)
        return torch.where(local_mask, a, b), torch.where(local_mask, b, a)

    phi_w, phi_l = orient(phi_a, phi_b)
    c_w, c_l = orient(c_a.to(dgp.device), c_b.to(dgp.device))
    s_w, s_l = orient(s_a.to(dgp.device), s_b.to(dgp.device))
    q_w = torch.where(a_wins, q_a, q_b).to(dgp.device)
    q_l = torch.where(a_wins, q_b, q_a).to(dgp.device)
    return PreferenceLatents(
        PreferenceBatch(phi_w, phi_l, q_w, q_l),
        c_w, s_w, q_w, c_l, s_l, q_l)


def _draw_donors(
    bank_size: int, anchors: int, redraws: int, *, seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randint(
        bank_size, (anchors, redraws), generator=generator, dtype=torch.int64)


def make_packet_pools(
    dgp: GenericNonlinearDGP, latents: PreferenceLatents, *,
    n_candidates: int, seed: int,
) -> dict[str, PacketPool]:
    """Subsample strict sources, then redraw from the full strict bank."""
    n = len(latents.c_w)
    if not 1 <= n_candidates <= n:
        raise ValueError("candidate count must fit the strict source set")
    source_rng = torch.Generator(device="cpu").manual_seed(int(seed))
    source_indices = torch.randperm(n, generator=source_rng)[:n_candidates].sort().values
    source_device = source_indices.to(latents.c_w.device)
    bank = latents.nuisance_bank
    donors = _draw_donors(len(bank), n_candidates, 1, seed=seed+1)[:, 0]
    orientation_rng = torch.Generator(device="cpu").manual_seed(int(seed) + 2)
    a_wins = torch.rand(n_candidates, generator=orientation_rng) < 0.5
    donor_s = bank[donors.to(bank.device)]
    resampling = PacketPool(
        latents.strict.phi_w[source_device],
        dgp._features(latents.c_w[source_device], donor_s),
        latents.c_w[source_device], donors, a_wins, source_indices, bank)
    return {"resampling": resampling}


def make_probe_bank(
    dgp: GenericNonlinearDGP, latents: PreferenceLatents, *,
    nuisance_bank: torch.Tensor, redraws: int, seed: int,
) -> ProbeBank:
    """Form held-out winner-preserving ties using the supplied donor bank."""
    bank = nuisance_bank
    if bank.ndim != 2 or bank.shape[1] != latents.s_w.shape[1]:
        raise ValueError("probe nuisance bank has incompatible shape")
    n = len(latents.c_w)
    donors = _draw_donors(len(bank), n, redraws, seed=seed)
    donor_values = bank[donors.to(bank.device)]
    c = latents.c_w[:, None, :].expand(-1, redraws, -1)
    features = dgp._features(
        c.reshape(-1, c.shape[-1]),
        donor_values.reshape(-1, donor_values.shape[-1]),
    ).reshape(n, redraws, -1)
    return ProbeBank(
        resampling_first=latents.strict.phi_w,
        resampling_second=features[:, 0],
        resampling_redraws=features,
        anchor_c=latents.c_w,
        nuisance_bank=bank,
        donor_indices=donors,
    )


def sample_evaluations(
    dgp: GenericNonlinearDGP, environments: tuple[str, ...], n: int,
    seed_for,
) -> dict[str, LatentPairBatch]:
    """Sample independent evaluation pairs for each environment."""
    return {
        name: dgp.sample_latent_pairs(n, environment=name, seed=seed_for(name))
        for name in environments
    }
