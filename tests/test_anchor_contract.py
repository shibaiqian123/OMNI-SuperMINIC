"""Existing 9D torso-anchor contract used by the G1 FSQ actor."""

from types import SimpleNamespace

import torch

from supermimic.world_model.heads import anchor_relative_obs


def test_anchor_identity_and_known_yaw():
    pos_robot = torch.zeros(1, 3)
    quat_robot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pos_ref = torch.tensor([[1.0, 2.0, 3.0]])
    same = anchor_relative_obs(pos_robot, quat_robot, pos_ref, quat_robot)
    torch.testing.assert_close(same, torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]))

    half = 2**-0.5
    yaw90 = torch.tensor([[half, 0.0, 0.0, half]])
    turned = anchor_relative_obs(pos_robot, quat_robot, pos_ref, yaw90)
    torch.testing.assert_close(turned[0, 3:], torch.tensor([0.0, -1.0, 1.0, 0.0, 0.0, 0.0]), atol=1e-6, rtol=0)


def test_quaternion_sign_does_not_change_anchor_features():
    q = torch.tensor([[0.9238795, 0.0, 0.0, 0.3826834]])
    args = (torch.zeros(1, 3), q, torch.ones(1, 3))
    a = anchor_relative_obs(*args, q)
    b = anchor_relative_obs(args[0], -q, args[2], -q)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_actor_exception_restores_frozen_parameter_flags(monkeypatch):
    from supermimic.imagination.policy_trainer import PolicyTrainer, PolicyTrainerCfg

    class WorldModel(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.spec = SimpleNamespace(normalize_xi=lambda value: value)

    trainer = PolicyTrainer.__new__(PolicyTrainer)
    trainer.cfg = PolicyTrainerCfg(batch_size=2, history=1)
    trainer.wm = WorldModel()
    trainer.critic = torch.nn.Linear(2, 1)
    trainer.policy = torch.nn.Linear(2, 2)
    trainer.lib = None
    trainer.step_fn = None
    list(trainer.wm.parameters())[0].requires_grad_(False)
    before_wm = [p.requires_grad for p in trainer.wm.parameters()]
    before_critic = [p.requires_grad for p in trainer.critic.parameters()]
    buffer = SimpleNamespace(sample_starts=lambda *_: SimpleNamespace(xi=torch.zeros(2, 3)))

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic actor failure")

    monkeypatch.setattr("supermimic.imagination.policy_trainer.imagine", fail)
    try:
        trainer.update(buffer)
    except RuntimeError as exc:
        assert "synthetic" in str(exc)
    else:
        raise AssertionError("expected the synthetic actor failure")
    assert [p.requires_grad for p in trainer.wm.parameters()] == before_wm
    assert [p.requires_grad for p in trainer.critic.parameters()] == before_critic
