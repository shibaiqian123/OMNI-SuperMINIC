"""Finite scalar quantization used by the OmniMimic motion-token actor.

The quantizer uses fixed, symmetric bin centres. For an even level count L,
the centres are ``-1 + 2*i/(L-1)`` for ``i in [0, L)``; inputs are clipped to
the closed interval before nearest-centre rounding. The training path uses a
straight-through estimator (STE), while ``quantize_deterministic`` exposes the
same forward rule without a gradient shortcut for deployment/tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class FSQCfg:
    num_tokens: int = 2
    token_dim: int = 32
    levels: int = 32
    use_ste: bool = True

    @property
    def latent_dim(self) -> int:
        return self.num_tokens * self.token_dim

    def validate(self) -> None:
        if self.num_tokens < 1 or self.token_dim < 1:
            raise ValueError("FSQ num_tokens and token_dim must be positive")
        if self.levels < 2:
            raise ValueError("FSQ levels must be at least 2")


@dataclass
class QuantizedLatent:
    prequant: torch.Tensor
    z_q: torch.Tensor
    indices: torch.Tensor


class FSQ(nn.Module):
    """Per-component finite scalar quantizer with a fixed codebook grid."""

    def __init__(self, cfg: FSQCfg | None = None):
        super().__init__()
        self.cfg = cfg or FSQCfg()
        self.cfg.validate()
        self.register_buffer("centres", torch.linspace(-1.0, 1.0, self.cfg.levels))

    def _check(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim < 2:
            raise ValueError(f"FSQ expected rank >= 2 with a batch dimension, got shape {tuple(z.shape)}")
        if z.shape[-1] != self.cfg.latent_dim:
            raise ValueError(f"FSQ expected last dimension {self.cfg.latent_dim}, got {z.shape[-1]}")
        if not torch.is_floating_point(z):
            raise TypeError("FSQ input must be floating point")
        return z.reshape(*z.shape[:-1], self.cfg.num_tokens, self.cfg.token_dim)

    def quantize_deterministic(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(z_q, indices)`` using the deployment-exact forward rule."""
        tokens = self._check(z)
        bounded = tokens.clamp(-1.0, 1.0)
        scaled = (bounded + 1.0) * (self.cfg.levels - 1) / 2.0
        # Explicit half-up rule avoids backend-dependent tie behaviour.
        indices = torch.floor(scaled + 0.5).to(torch.long).clamp(0, self.cfg.levels - 1)
        z_q = self.centres.to(dtype=z.dtype)[indices]
        return z_q, indices

    def quantize(self, z: torch.Tensor) -> QuantizedLatent:
        z_q, indices = self.quantize_deterministic(z)
        if self.cfg.use_ste:
            z_q_out = z.reshape_as(z_q) + (z_q - z.reshape_as(z_q)).detach()
        else:
            z_q_out = z_q
        return QuantizedLatent(prequant=z, z_q=z_q_out, indices=indices)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.quantize(z).z_q


def indices_to_latent(indices: torch.Tensor, cfg: FSQCfg | None = None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Decode integer FSQ indices to flattened token values."""
    cfg = cfg or FSQCfg()
    cfg.validate()
    if indices.shape[-2:] != (cfg.num_tokens, cfg.token_dim):
        raise ValueError(f"expected indices [..., {cfg.num_tokens}, {cfg.token_dim}], got {tuple(indices.shape)}")
    if indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("FSQ indices must be an integer tensor")
    if bool(((indices < 0) | (indices >= cfg.levels)).any()):
        raise ValueError("FSQ indices out of range")
    centres = torch.linspace(-1.0, 1.0, cfg.levels, device=indices.device, dtype=dtype)
    return centres[indices].reshape(*indices.shape[:-2], cfg.latent_dim)
