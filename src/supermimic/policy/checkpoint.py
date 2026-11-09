"""Versioned policy checkpoints shared by training and evaluation.

Legacy means a payload without policy_schema AND without a config schema version.
Missing legacy fields use NJU defaults, never shape-based architecture inference.
FSQ and widened policies carry explicit variants and quantizer metadata.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
import math

import torch

from supermimic.policy.nets import Policy, PolicyCfg

SCHEMA_VERSION = 1


def _plain(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _config(cfg):
    cfg = _plain(cfg)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("policy"), dict):
        raise ValueError("config.policy: expected a configuration mapping")
    return cfg


def policy_metadata(cfg) -> dict:
    """Validate a resolved training config and describe its policy interface."""
    cfg = _config(cfg)
    version = cfg.get("policy_schema_version", SCHEMA_VERSION)
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"config.policy_schema_version: unsupported {version!r}")
    variant = cfg.get("policy_variant", "continuous")
    if variant not in ("continuous", "continuous64", "fsq64"):
        raise ValueError(f"config.policy_variant: unsupported {variant!r}")
    try:
        pc = PolicyCfg(**cfg["policy"])
    except TypeError as exc:
        raise ValueError(f"config.policy: {exc}") from exc
    env = cfg.get("env", {})
    if not isinstance(env, dict):
        raise ValueError("config.env: expected a mapping")
    future = env.get("future_frames", 10)
    anchor = env.get("anchor_obs", False)
    locked = env.get("lock_waist", False)
    if type(future) is not int or future < 1:
        raise ValueError("config.env.future_frames: must be a positive integer")
    if type(anchor) is not bool or type(locked) is not bool:
        raise ValueError("config.env.anchor_obs / lock_waist: must be boolean")
    expected_goal = 68 * future + (9 if anchor else 0)
    if type(pc.goal_dim) is not int or pc.goal_dim != expected_goal:
        raise ValueError(f"config.policy.goal_dim: {pc.goal_dim!r} != {expected_goal} for future_frames={future}, anchor_obs={anchor}")
    if type(pc.latent_dim) is not int or pc.latent_dim < 1:
        raise ValueError("config.policy.latent_dim: must be a positive integer")
    required_latent = 32 if variant == "continuous" else 64
    if pc.latent_dim != required_latent:
        raise ValueError(f"config.policy.latent_dim: {variant} requires {required_latent}, got {pc.latent_dim}")
    if any(type(v) is not int or v < 1 for v in (pc.fsq_num_tokens, pc.fsq_token_dim, pc.fsq_levels)):
        raise ValueError("config.policy.fsq_*: expected positive integers")
    if pc.fsq_levels < 2:
        raise ValueError("config.policy.fsq_levels: must be at least 2")
    if variant == "fsq64" and pc.fsq_num_tokens * pc.fsq_token_dim != pc.latent_dim:
        raise ValueError("config.policy.fsq_num_tokens * fsq_token_dim must equal latent_dim")
    if type(pc.history) is not int or pc.history != 10:
        raise ValueError("config.policy.history: current Decoder requires 10 frames (930D)")
    for name, n in (("obs_term_scale", 5), ("goal_term_scale", 6)):
        values = getattr(pc, name)
        if len(values) != n or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"config.policy.{name}: expected {n} finite positive scales")
    for name in ("encoder_hidden", "decoder_hidden"):
        if any(type(n) is not int or n <= 0 for n in getattr(pc, name)):
            raise ValueError(f"config.policy.{name}: widths must be positive integers")
    return {
        "version": version, "variant": variant, "policy": _plain(pc),
        "interfaces": ["motion"],
        "fsq": ({"num_tokens": pc.fsq_num_tokens, "token_dim": pc.fsq_token_dim,
                 "levels": pc.fsq_levels, "range": [-1.0, 1.0],
                 "round": "nearest-half-up"} if variant == "fsq64" else None),
        "observation": {"frame_dim": 93, "history": 10, "layout": "term-major"},
        "goal": {"future_frames": future, "anchor_obs": anchor,
                 "mode": "pose9d" if anchor else "none", "layout": "frame-major+anchor"},
        "robot": {"name": "g1", "action_dim": 29, "lock_waist": locked,
                  "lock_waist_refs": env.get("lock_waist_refs", "feasible") if locked else None},
    }


def _compare(actual, expected, path):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{path}: missing or unknown fields")
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path}: checkpoint {actual!r} != requested {expected!r}")


def build_policy(cfg) -> Policy:
    meta = policy_metadata(cfg)
    return Policy(PolicyCfg(**meta["policy"]), variant=meta["variant"])


def checkpoint_metadata(payload) -> dict:
    if not isinstance(payload, Mapping) or "config" not in payload:
        raise ValueError("checkpoint.config: missing")
    cfg = _config(payload["config"])
    meta = policy_metadata(cfg)
    if "policy_schema" in payload:
        # Versioned configs must not fall back to legacy defaults if truncated.
        for key in ("policy_schema_version", "policy_variant", "env"):
            if key not in cfg:
                raise ValueError(f"checkpoint.config.{key}: missing")
        # C02 versioned continuous checkpoints predate the three FSQ config
        # fields. They have no effect on that architecture, so defaults remain
        # the only allowed compatibility fill.
        optional_continuous = {"fsq_num_tokens", "fsq_token_dim", "fsq_levels"}
        for field in fields(PolicyCfg):
            if meta["variant"] == "continuous" and field.name in optional_continuous:
                continue
            if field.name not in cfg["policy"]:
                raise ValueError(f"checkpoint.config.policy.{field.name}: missing")
        for key in ("future_frames", "anchor_obs", "lock_waist", "lock_waist_refs"):
            if key not in cfg["env"]:
                raise ValueError(f"checkpoint.config.env.{key}: missing")
        _compare(_plain(payload["policy_schema"]), meta, "policy_schema")
    elif "policy_schema_version" in cfg or "policy_variant" in cfg:
        raise ValueError("checkpoint.policy_schema: missing for versioned config")
    return meta


def load_policy_state(payload, cfg=None, *, policy: Policy | None = None) -> Policy:
    """Validate before copying tensors; keep parameter objects for optimizer resume."""
    meta = checkpoint_metadata(payload)
    if cfg is not None:
        _compare(meta, policy_metadata(cfg), "policy_schema")
    if policy is None:
        policy = build_policy(payload["config"])
    else:
        _compare(_plain(policy.cfg), meta["policy"], "policy.cfg")
        if policy.variant != meta["variant"]:
            raise ValueError(f"policy.variant: {policy.variant!r} != {meta['variant']!r}")
    state = payload.get("policy")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint.policy: missing state dictionary")
    expected = policy.state_dict()
    missing, extra = expected.keys() - state.keys(), state.keys() - expected.keys()
    if missing or extra:
        raise ValueError(f"checkpoint.policy: missing keys={sorted(missing)}, unexpected keys={sorted(extra)}")
    for key, tensor in expected.items():
        value = state[key]
        if not isinstance(value, torch.Tensor) or value.shape != tensor.shape:
            raise ValueError(f"checkpoint.policy.{key}: expected shape {tuple(tensor.shape)}, got {getattr(value, 'shape', None)}")
        if key.endswith(".scale") and not torch.equal(value.cpu(), tensor.cpu()):
            raise ValueError(f"checkpoint.policy.{key}: scale buffer conflicts with config")
    policy.load_state_dict(state, strict=True)
    return policy
