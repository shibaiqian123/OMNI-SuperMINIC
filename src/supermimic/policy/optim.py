"""Optimizer bundles for Blink runtime experiments.

Muon is optional and only selected when explicitly requested. Hidden Linear
weights use Muon; first/last layers, vectors and biases stay on Adam.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _linears(module: nn.Module) -> list[nn.Linear]:
    return [m for m in module.modules() if isinstance(m, nn.Linear)]


class MuonAdam:
    def __init__(self, model: nn.Module, mlps: list[nn.Module], lr: float, betas=(0.9, 0.95), fused: bool = False):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("runtime.pi_opt/wm_opt=muon requires torch.optim.Muon")
        hidden = {id(layer.weight) for mlp in mlps for layer in _linears(mlp)[1:-1]}
        muon_params = [p for p in model.parameters() if id(p) in hidden]
        adam_params = [p for p in model.parameters() if id(p) not in hidden and p.requires_grad]
        if not muon_params:
            raise ValueError("Muon optimizer received no hidden Linear weights")
        self.muon = torch.optim.Muon(muon_params, lr=lr, weight_decay=0.0, adjust_lr_fn="match_rms_adamw")
        self.adam = torch.optim.Adam(adam_params, lr=lr, betas=betas, fused=fused)

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adam.param_groups

    def zero_grad(self, set_to_none: bool = True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}

    def load_state_dict(self, state):
        self.muon.load_state_dict(state["muon"])
        self.adam.load_state_dict(state["adam"])


def build_optimizer(model: nn.Module, mlps: list[nn.Module], *, kind: str, lr: float, betas=(0.9, 0.95), fused: bool = False):
    if kind == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, fused=fused)
    if kind == "muon":
        return MuonAdam(model, mlps, lr=lr, betas=betas, fused=fused)
    raise ValueError(f"unsupported optimizer kind: {kind!r}")
