import torch

from supermimic.world_model.mirror import G1Mirror


def test_mirror_is_involution_and_preserves_quaternion_norm():
    mirror = G1Mirror("cpu")
    state = torch.randn(4, 71)
    state[:, 3:7] = state[:, 3:7] / state[:, 3:7].norm(dim=-1, keepdim=True)
    twice = mirror.state(mirror.state(state))
    torch.testing.assert_close(twice, state, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(mirror.state(state)[:, 3:7].norm(dim=-1), torch.ones(4), atol=1e-6, rtol=0)


def test_mirror_xi_is_involution():
    mirror = G1Mirror("cpu")
    xi = torch.randn(4, 33)
    torch.testing.assert_close(mirror.xi(mirror.xi(xi)), xi, atol=1e-6, rtol=1e-6)
