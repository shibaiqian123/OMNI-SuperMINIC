"""CPU checkpoint compatibility; small real networks, no training dataset needed."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest
import torch

from supermimic.policy.checkpoint import build_policy, load_policy_state, policy_metadata
from supermimic.policy.nets import Policy, PolicyCfg


def config(anchor=False, latent=32):
    return {
        "policy_schema_version": 1, "policy_variant": "continuous",
        "policy": asdict(PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,),
                                   goal_dim=689 if anchor else 680, latent_dim=latent)),
        "env": {"anchor_obs": anchor, "future_frames": 10,
                "lock_waist": False, "lock_waist_refs": "feasible"},
    }


def checkpoint(cfg, legacy=False):
    # Construct with the original public Policy constructor, not the new builder.
    model = Policy(PolicyCfg(**cfg["policy"]))
    saved = deepcopy(cfg)
    if legacy:
        saved.pop("policy_schema_version")
        saved.pop("policy_variant")
    ck = {"config": saved, "policy": model.state_dict()}
    if not legacy:
        ck["policy_schema"] = policy_metadata(cfg)
    return model, ck


@pytest.mark.parametrize("anchor", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_output_roundtrip(tmp_path, anchor, legacy):
    cfg = config(anchor)
    old, ck = checkpoint(cfg, legacy)
    # JSON conversion changes tuples into lists, as in external config tooling.
    ck["config"] = json.loads(json.dumps(ck["config"]))
    path = tmp_path / "checkpoint.pt"
    torch.save(ck, path)
    new = load_policy_state(torch.load(path, weights_only=False), cfg)
    obs, goal = torch.randn(3, 930), torch.randn(3, cfg["policy"]["goal_dim"])
    torch.testing.assert_close(new(obs, goal), old(obs, goal), rtol=0, atol=0)


def test_legacy_missing_fields_use_baseline_defaults():
    cfg = config()
    old, ck = checkpoint(cfg, True)
    ck["config"].pop("env")
    for key in ("latent_dim", "goal_dim", "history", "obs_term_scale", "goal_term_scale"):
        ck["config"]["policy"].pop(key)
    new = load_policy_state(ck)
    for key, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("mutation,match", [
    (lambda c: c["policy"].update(latent_dim=64), "latent_dim"),
    (lambda c: c["env"].update(anchor_obs=True), "goal_dim"),
    (lambda c: c["env"].update(lock_waist=True), "lock_waist"),
    (lambda c: c.update(policy_variant="fsq"), "policy_variant"),
    (lambda c: c.update(policy_schema_version=2), "schema_version"),
    (lambda c: c["policy"].update(history=5), "history"),
])
def test_requested_config_conflicts(mutation, match):
    cfg = config()
    _, ck = checkpoint(cfg)
    mutation(cfg)
    with pytest.raises(ValueError, match=match):
        load_policy_state(ck, cfg)


@pytest.mark.parametrize("case,match", [
    ("missing_encoder", "encoders.motion"), ("wrong_tensor", "expected shape"),
    ("schema_version", "version"), ("missing_schema", "policy_schema"),
    ("missing_config_field", "latent_dim"), ("scale", "scale buffer"),
    ("fsq_metadata", "fsq"), ("latent64_as32", "expected shape"),
])
def test_invalid_checkpoints_rejected_without_mutation(case, match):
    cfg = config()
    _, ck = checkpoint(cfg)
    target = build_policy(cfg)
    before = deepcopy(target.state_dict())
    if case == "missing_encoder":
        del ck["policy"][next(k for k in ck["policy"] if k.startswith("encoders.motion"))]
    elif case == "wrong_tensor":
        ck["policy"]["decoder.mlp.net.0.weight"] = torch.zeros(1, 1)
    elif case == "schema_version":
        ck["policy_schema"]["version"] = 99
    elif case == "missing_schema":
        del ck["policy_schema"]
    elif case == "missing_config_field":
        del ck["config"]["policy"]["latent_dim"]
    elif case == "scale":
        ck["policy"]["decoder.scale"] = torch.zeros(930)
    elif case == "fsq_metadata":
        ck["policy_schema"]["fsq"] = {"levels": 32}
    elif case == "latent64_as32":
        _, other = checkpoint(config(latent=64))
        ck["policy"] = other["policy"]
    with pytest.raises(ValueError, match=match):
        load_policy_state(ck, cfg, policy=target)
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)


def test_optimizer_save_resume_next_step(tmp_path):
    cfg = config()
    old = build_policy(cfg)
    opt = torch.optim.Adam(old.parameters(), lr=1e-3)
    obs, goal = torch.randn(4, 930), torch.randn(4, 680)

    def step(model, optimizer):
        optimizer.zero_grad()
        model(obs, goal).square().mean().backward()
        optimizer.step()

    step(old, opt)
    path = tmp_path / "resume.pt"
    torch.save({"config": cfg, "policy_schema": policy_metadata(cfg),
                "policy": old.state_dict(), "policy_opt": opt.state_dict()}, path)
    ck = torch.load(path, weights_only=False)
    new = build_policy(cfg)
    resumed_opt = torch.optim.Adam(new.parameters(), lr=0.1)
    parameter_ids = [id(p) for p in new.parameters()]
    load_policy_state(ck, cfg, policy=new)
    assert parameter_ids == [id(p) for p in new.parameters()]
    resumed_opt.load_state_dict(ck["policy_opt"])
    step(old, opt)
    step(new, resumed_opt)
    for key, value in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[key], value, rtol=0, atol=0)


def test_benchmark_uses_same_loader(tmp_path):
    from supermimic.eval.benchmark import load_policy_from_checkpoint

    cfg = config(True)
    old, ck = checkpoint(cfg)
    path = tmp_path / "eval.pt"
    torch.save(ck, path)
    new, saved = load_policy_from_checkpoint(path, "cpu")
    assert saved == cfg
    assert not new.training
    obs, goal = torch.randn(2, 930), torch.randn(2, 689)
    torch.testing.assert_close(new(obs, goal), old(obs, goal), rtol=0, atol=0)


def test_trainer_save_load_preserves_all_states(tmp_path):
    from types import SimpleNamespace
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    class Sampler:
        def __init__(self):
            self.bin_return = torch.tensor([3.0, 4.0])
            self.reset_called = False

        def sampler_state(self):
            return {"bin_return": self.bin_return.clone()}

        def load_sampler_state(self, ck):
            self.bin_return.copy_(ck["bin_return"])
            return "loaded"

        def reset_sampler_state(self):
            self.reset_called = True

    cfg = TrainConfig(device="cpu")
    cfg.policy = PolicyCfg(**config()["policy"])
    cfg.resolve_policy_config()

    def trainer():
        t = Trainer.__new__(Trainer)  # exercise checkpoint methods without a simulator
        t.cfg, t.device, t.is_main, t.log_dir = cfg, "cpu", True, tmp_path
        t.policy = build_policy(cfg)
        t.wm, t.critic = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
        t.wm_trainer = SimpleNamespace(opt=torch.optim.Adam(t.wm.parameters()))
        t.policy_trainer = SimpleNamespace(opt=torch.optim.Adam(t.policy.parameters()),
                                           critic_opt=torch.optim.Adam(t.critic.parameters()))
        t.command, t.iteration, t.num_updates = Sampler(), 17, 9
        return t

    old = trainer()
    for model, opt in ((old.wm, old.wm_trainer.opt), (old.critic, old.policy_trainer.critic_opt)):
        model(torch.ones(1, 2)).sum().backward()
        opt.step()
    old.policy(torch.ones(1, 930), torch.ones(1, 680)).sum().backward()
    old.policy_trainer.opt.step()
    old.save()
    new = trainer()
    new.command.bin_return.zero_()
    ids = [id(p) for p in new.policy.parameters()]
    new.load(tmp_path / "ckpt_latest.pt")
    assert ids == [id(p) for p in new.policy.parameters()]
    assert (new.iteration, new.num_updates) == (17, 9)
    torch.testing.assert_close(new.command.bin_return, old.command.bin_return)
    for name in ("policy", "wm", "critic"):
        for key, value in getattr(old, name).state_dict().items():
            torch.testing.assert_close(getattr(new, name).state_dict()[key], value)
    for a, b in ((old.wm_trainer.opt, new.wm_trainer.opt),
                 (old.policy_trainer.opt, new.policy_trainer.opt),
                 (old.policy_trainer.critic_opt, new.policy_trainer.critic_opt)):
        assert a.state_dict()["param_groups"] == b.state_dict()["param_groups"]
        for key, state in a.state_dict()["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(b.state_dict()["state"][key][field], value)
    cfg.resume_reset_sampler = True
    new.load(tmp_path / "ckpt_latest.pt")
    assert new.command.reset_called


def test_resolve_anchor_and_reject_explicit_conflict():
    from supermimic.config import TrainConfig

    cfg = TrainConfig()
    cfg.env.anchor_obs = True
    cfg.resolve_policy_config()
    assert cfg.policy.goal_dim == 689
    cfg.policy.goal_dim = 686
    with pytest.raises(ValueError, match="goal_dim"):
        cfg.resolve_policy_config()
