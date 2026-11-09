"""FSQ numerical contract and gradient tests."""

import pytest
import torch

from supermimic.policy.fsq import FSQ, FSQCfg, indices_to_latent


def test_all_levels_and_index_roundtrip():
    cfg = FSQCfg(num_tokens=2, token_dim=32, levels=32)
    fsq = FSQ(cfg)
    values = fsq.centres.repeat(2).reshape(1, 64)
    info = fsq.quantize(values)
    assert info.z_q.shape == (1, 2, 32)
    assert info.indices.shape == (1, 2, 32)
    assert torch.equal(info.indices[0, 0], torch.arange(32))
    assert torch.equal(info.indices[0, 1], torch.arange(32))
    torch.testing.assert_close(indices_to_latent(info.indices, cfg), values, rtol=0, atol=0)


def test_saturation_and_half_up_rule():
    fsq = FSQ()
    step = 2.0 / 31.0
    z = torch.zeros(1, 64)
    z[0, :6] = torch.tensor([-9.0, 9.0, -1.0, 1.0, -1.0 + 0.5 * step, -1.0 + 1.5 * step])
    q, idx = fsq.quantize_deterministic(z)
    assert idx[0, 0, :6].tolist() == [0, 31, 0, 31, 1, 2]
    assert q.min() >= -1 and q.max() <= 1


def test_ste_gradient_is_finite_and_nonzero():
    z = torch.linspace(-1.5, 1.5, 128).reshape(2, 64).requires_grad_()
    out = FSQ()(z)
    out.square().mean().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    assert bool((z.grad != 0).any())


@pytest.mark.parametrize("shape,match", [((64,), "rank"), ((2, 63), "last dimension"), ((2, 65), "last dimension")])
def test_wrong_latent_shape_rejected(shape, match):
    with pytest.raises(ValueError, match=match):
        FSQ()(torch.zeros(shape))


def test_state_dict_roundtrip():
    old = FSQ(FSQCfg(levels=16))
    state = old.state_dict()
    new = FSQ(FSQCfg(levels=16))
    new.load_state_dict(state, strict=True)
    x = torch.randn(4, 64)
    torch.testing.assert_close(new(x), old(x), rtol=0, atol=0)
