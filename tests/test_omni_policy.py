"""OmniMimic G1 policy variants and split-interface tests."""

import pytest
import torch

from supermimic.policy.nets import Policy, PolicyCfg


def cfg(latent: int, goal_dim: int = 680) -> PolicyCfg:
    return PolicyCfg(encoder_hidden=(32, 16), decoder_hidden=(32, 16), latent_dim=latent, goal_dim=goal_dim)


@pytest.mark.parametrize("variant,latent,token_shape", [
    ("continuous", 32, (5, 32)),
    ("continuous64", 64, (5, 64)),
    ("fsq64", 64, (5, 2, 32)),
])
@pytest.mark.parametrize("goal_dim", [680, 689])
def test_split_interface_matches_forward(variant, latent, token_shape, goal_dim):
    torch.manual_seed(0)
    policy = Policy(cfg(latent, goal_dim), variant)
    obs, goal = torch.randn(5, 930), torch.randn(5, goal_dim)
    encoded = policy.encode(goal)
    token = policy.quantize(encoded)
    assert token.shape == token_shape
    action = policy.decode(obs, token)
    assert action.shape == (5, 29)
    torch.testing.assert_close(policy(obs, goal), action, rtol=0, atol=0)


def test_fsq_actor_backpropagates_to_encoder_and_decoder():
    policy = Policy(cfg(64), "fsq64")
    loss = policy(torch.randn(4, 930), torch.randn(4, 680)).square().mean()
    loss.backward()
    for prefix in ("encoders.motion", "decoder"):
        grads = [p.grad for name, p in policy.named_parameters() if name.startswith(prefix)]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(bool((g != 0).any()) for g in grads)


@pytest.mark.parametrize("variant,latent", [("continuous", 64), ("continuous64", 32), ("fsq64", 32)])
def test_variant_latent_mismatch_rejected(variant, latent):
    with pytest.raises(ValueError, match="requires latent_dim"):
        Policy(cfg(latent), variant)


def test_invalid_inputs_rejected():
    policy = Policy(cfg(64), "fsq64")
    with pytest.raises(ValueError, match="goal"):
        policy(torch.zeros(2, 930), torch.zeros(2, 679))
    with pytest.raises(ValueError, match="obs_hist"):
        policy(torch.zeros(2, 929), torch.zeros(2, 680))
    with pytest.raises(ValueError, match="interface"):
        policy.encode(torch.zeros(2, 680), "smpl")


def test_legacy_constructor_and_parameter_names_stay_stable():
    policy = Policy(cfg(32))
    assert policy.variant == "continuous"
    assert policy.fsq is None
    assert "encoders.motion.mlp.net.0.weight" in policy.state_dict()
    assert "decoder.mlp.net.0.weight" in policy.state_dict()
