"""G1 sagittal-plane reflection for World Model training windows."""

from __future__ import annotations

import torch

from supermimic.data.motion_convert import G1_JOINT_NAMES
from supermimic.world_model.state_spec import StateSpec


class G1Mirror:
    """Mirror flat 71D states, 29D actions and 33D simulation context."""

    def __init__(self, device: torch.device | str):
        names = list(G1_JOINT_NAMES)
        partners = []
        signs = []
        for name in names:
            partner = name.replace("left_", "@", 1).replace("right_", "left_", 1).replace("@", "right_", 1)
            partners.append(names.index(partner))
            signs.append(-1.0 if ("_roll_" in name or "_yaw_" in name) else 1.0)
        self.perm = torch.tensor(partners, dtype=torch.long, device=device)
        self.sign = torch.tensor(signs, dtype=torch.float32, device=device)

    def joints(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., self.perm] * self.sign.to(dtype=x.dtype)

    def state(self, state: torch.Tensor) -> torch.Tensor:
        p, q, qj, v, w, qd = StateSpec.split(state)
        # Reflection y -> -y: quaternion (w,x,y,z) -> (w,-x,y,-z).
        q_m = q * q.new_tensor([1.0, -1.0, 1.0, -1.0])
        p_m = p * p.new_tensor([1.0, -1.0, 1.0])
        v_m = v * v.new_tensor([1.0, -1.0, 1.0])
        w_m = w * w.new_tensor([-1.0, 1.0, -1.0])
        return torch.cat([p_m, q_m, self.joints(qj), v_m, w_m, self.joints(qd)], dim=-1)

    def xi(self, xi: torch.Tensor) -> torch.Tensor:
        out = xi.clone()
        # xi = friction[1] | encoder bias[29] | torso COM offset[3].
        out[..., 1:30] = self.joints(xi[..., 1:30])
        out[..., 31] = -xi[..., 31]
        return out
