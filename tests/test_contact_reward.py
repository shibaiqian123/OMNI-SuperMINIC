import torch

from supermimic.envs.mdp.rewards import TrackingRewardCfg, tracking_reward_terms


def test_contact_reward_agrees_and_penalizes_slip():
    n, b = 2, 7
    pos = torch.zeros(n, b, 3)
    pos[:, 3, 2] = 0.02
    pos[:, 6, 2] = 0.02
    ref_pos = pos.clone()
    vel = torch.zeros(n, b, 3)
    ref_vel = vel.clone()
    vel[0, 3, 0] = 1.0
    q = torch.zeros(n, b, 4)
    q[..., 0] = 1.0
    cfg = TrackingRewardCfg(w_contact=1.0)
    common = dict(robot_body_pos_w=pos, robot_body_quat_w=q, robot_body_lin_vel_w=vel,
                  robot_body_ang_vel_w=vel, ref_body_pos_w=ref_pos, ref_body_quat_w=q,
                  ref_body_lin_vel_w=ref_vel, ref_body_ang_vel_w=ref_vel, anchor_index=0,
                  action=torch.zeros(n, 29), prev_action=torch.zeros(n, 29),
                  joint_pos=torch.zeros(n, 29), soft_limits=torch.tensor([[-1., 1.]]).expand(29, 2), cfg=cfg)
    out = tracking_reward_terms(**common)
    assert out["contact"].shape == (n,)
    assert out["contact"][1] > out["contact"][0]
