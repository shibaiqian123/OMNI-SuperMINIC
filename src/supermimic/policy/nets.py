"""Policy (goal encoder + shared decoder) and critic (paper Table A.4)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from supermimic.policy.fsq import FSQ, FSQCfg
from supermimic.world_model.dynamics import MLP
from supermimic.world_model.heads import OBS_TERMS

GOAL_DIM = 680
LATENT_DIM = 32
ACT_DIM = 29


@dataclass
class PolicyCfg:
    encoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    decoder_hidden: tuple[int, ...] = (2048, 1024, 512, 256)
    latent_dim: int = LATENT_DIM
    history: int = 10
    # Fixed per-term observation scales (divide): joint_pos, joint_vel, base_ang_vel, projected_gravity, actions.
    obs_term_scale: tuple[float, ...] = (1.0, 5.0, 2.0, 1.0, 1.0)
    # Fixed goal feature scales: joint_pos 29, joint_vel 29, grav 3, ang_vel_b 3, lin_vel_h 3, height 1.
    goal_term_scale: tuple[float, ...] = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
    decoder_last_gain: float = 0.01
    goal_dim: int = GOAL_DIM  # 680 (+9 with anchor_obs)
    fsq_num_tokens: int = 2
    fsq_token_dim: int = 32
    fsq_levels: int = 32


@dataclass
class CriticCfg:
    hidden: tuple[int, ...] = (256, 128)


def _obs_scale_vector(cfg: PolicyCfg) -> torch.Tensor:
    parts = []
    for (name, d), s in zip(OBS_TERMS, cfg.obs_term_scale):
        parts.append(torch.full((cfg.history * d,), 1.0 / s))
    return torch.cat(parts)  # term-major layout matches the flattened history


def _goal_scale_vector(cfg: PolicyCfg, goal_dim: int) -> torch.Tensor:
    sizes = (29, 29, 3, 3, 3, 1)
    per_frame = torch.cat([torch.full((n,), 1.0 / s) for n, s in zip(sizes, cfg.goal_term_scale)])
    if goal_dim < 68:
        raise ValueError(f"goal_dim must include at least one 68D motion frame, got {goal_dim}")
    if goal_dim % 68 == 0:
        future_frames, anchor_dim = goal_dim // 68, 0
    elif (goal_dim - 9) % 68 == 0:
        future_frames, anchor_dim = (goal_dim - 9) // 68, 9
    else:
        raise ValueError(f"goal_dim {goal_dim} does not match K*68 or K*68+9")
    return torch.cat([per_frame.repeat(future_frames), torch.ones(anchor_dim)])


class MotionEncoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, goal_dim: int | None = None):
        super().__init__()
        goal_dim = goal_dim or cfg.goal_dim
        self.mlp = MLP(goal_dim, cfg.encoder_hidden, cfg.latent_dim)
        self.register_buffer("scale", _goal_scale_vector(cfg, goal_dim))

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(goal * self.scale))


class Decoder(nn.Module):
    def __init__(self, cfg: PolicyCfg, obs_dim: int = 930):
        super().__init__()
        self.mlp = MLP(cfg.latent_dim + obs_dim, cfg.decoder_hidden, ACT_DIM, last_gain=cfg.decoder_last_gain)
        self.register_buffer("scale", _obs_scale_vector(cfg))

    def forward(self, z: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        if z.ndim == 3:
            z = z.reshape(z.shape[0], -1)
        if z.ndim != 2 or obs_hist.ndim != 2:
            raise ValueError(f"Decoder expects rank-2 inputs, got token={tuple(z.shape)} obs={tuple(obs_hist.shape)}")
        return self.mlp(torch.cat([z, obs_hist * self.scale], dim=-1))


class Policy(nn.Module):
    """Motion encoder, optional FSQ, and shared action decoder.

    ``variant=continuous`` preserves the original 32D actor. ``continuous64``
    widens the latent without quantization; ``fsq64`` uses a 2x32 FSQ token.
    """

    def __init__(self, cfg: PolicyCfg, variant: str | None = None):
        super().__init__()
        self.cfg = cfg
        variant = variant or ("continuous" if cfg.latent_dim == 32 else "continuous64")
        self.variant = variant
        if variant not in {"continuous", "continuous64", "fsq64"}:
            raise ValueError(f"unknown policy variant: {variant!r}")
        expected = 32 if variant == "continuous" else 64
        if cfg.latent_dim != expected:
            raise ValueError(f"{variant} requires latent_dim={expected}, got {cfg.latent_dim}")
        self.encoders = nn.ModuleDict({"motion": MotionEncoder(cfg)})
        self.fsq = FSQ(FSQCfg(cfg.fsq_num_tokens, cfg.fsq_token_dim, cfg.fsq_levels)) if variant == "fsq64" else None
        if self.fsq is not None and cfg.latent_dim != self.fsq.cfg.latent_dim:
            raise ValueError("fsq latent dimensions do not match PolicyCfg.latent_dim")
        self.decoder = Decoder(cfg)

    def encode(self, goal: torch.Tensor, interface: str = "motion") -> torch.Tensor:
        if interface not in self.encoders:
            raise ValueError(f"unknown policy interface {interface!r}; available={list(self.encoders)}")
        if goal.ndim != 2 or goal.shape[-1] != self.cfg.goal_dim:
            raise ValueError(f"goal must have shape [B,{self.cfg.goal_dim}], got {tuple(goal.shape)}")
        return self.encoders[interface](goal)

    def quantize(self, latent: torch.Tensor, *, return_info: bool = False):
        if self.fsq is None:
            if latent.ndim != 2 or latent.shape[-1] != self.cfg.latent_dim:
                raise ValueError(f"latent must have shape [B,{self.cfg.latent_dim}], got {tuple(latent.shape)}")
            return latent
        info = self.fsq.quantize(latent)
        return info if return_info else info.z_q

    def decode(self, obs_hist: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        obs_dim = self.cfg.history * sum(d for _, d in OBS_TERMS)
        if obs_hist.ndim != 2 or obs_hist.shape[-1] != obs_dim:
            raise ValueError(f"obs_hist must have shape [B,{obs_dim}], got {tuple(obs_hist.shape)}")
        return self.decoder(token, obs_hist)

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor, interface: str = "motion") -> torch.Tensor:
        return self.decode(obs_hist, self.quantize(self.encode(goal, interface)))


class Critic(nn.Module):
    """V(canonical state 71, goal 680, xi 33, root error 4) -> scalar (one per interface)."""

    def __init__(self, cfg: CriticCfg, feat_dim: int = 71, goal_dim: int | None = None, xi_dim: int = 33, pcfg: PolicyCfg | None = None):
        super().__init__()
        pcfg = pcfg or PolicyCfg()
        goal_dim = goal_dim or pcfg.goal_dim
        self.mlp = MLP(feat_dim + goal_dim + xi_dim + 4, cfg.hidden, 1)
        self.register_buffer("goal_scale", _goal_scale_vector(pcfg, goal_dim))

    def forward(self, feat: torch.Tensor, goal: torch.Tensor, xi_norm: torch.Tensor, e_root: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([feat, goal * self.goal_scale, xi_norm, e_root], dim=-1)).squeeze(-1)
