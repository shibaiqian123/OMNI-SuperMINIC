import torch

from supermimic.world_model.state_spec import StateSpec, StateSpecCfg


def test_legacy_and_semi_implicit_zero_delta_are_finite_and_distinct_for_acceleration():
    legacy = StateSpec(StateSpecCfg(dynamics_mode="legacy"), "cpu")
    structured = StateSpec(StateSpecCfg(dynamics_mode="semi_implicit"), "cpu")
    state = torch.zeros(2, 71)
    state[:, 3] = 1.0
    state[:, 7] = 0.2
    state[:, 36] = 1.0
    delta = torch.zeros(2, 70)
    delta[:, 0] = 1.0
    a = legacy.integrate(state, delta)
    b = structured.integrate(state, delta)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.equal(a[:, 0:3], b[:, 0:3])


def test_legacy_default_matches_explicit_legacy_mode():
    default = StateSpec(StateSpecCfg(), "cpu")
    legacy = StateSpec(StateSpecCfg(dynamics_mode="legacy"), "cpu")
    state = torch.randn(3, 71)
    state[:, 3:7] = state[:, 3:7] / state[:, 3:7].norm(dim=-1, keepdim=True)
    delta = torch.randn(3, 70)
    torch.testing.assert_close(default.integrate(state, delta), legacy.integrate(state, delta))
