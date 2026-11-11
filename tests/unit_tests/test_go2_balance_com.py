"""Centre-of-mass offset from the front-foot support: observation and penalty."""

import math

import numpy as np
import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2balance import Go2Balance
from gym.envs.go2.go2balance_com_config import (
    Go2BalanceComCfg,
    Go2BalanceComRunnerCfg,
)
from gym.envs.go2.go2balance_config import Go2BalanceRunnerCfg
from gym.utils.task_registry import task_registry
from gym.utils.torch_quat import quat_apply


@pytest.fixture
def env():
    # Five rigid tilts at rest: -20, -10, 0, 10 and 20 degrees.
    cfg = Go2BalanceComCfg()
    cfg.env.num_envs = 5
    cfg.seed = 17
    cfg.tilt_test.enabled = True
    cfg.tilt_test.grid = [5, 1]
    cfg.tilt_test.angle_degrees = [-20.0, 20.0]
    cfg.tilt_test.angular_velocity = [0.0, 0.0]
    task_registry.convert_frequencies_to_params(cfg, Go2BalanceComRunnerCfg())
    backend = MuJocoCPUBackend()
    try:
        env = Go2Balance(cfg, "cpu", True, backend)
        env._reset_idx(torch.ones(5, dtype=torch.bool))
        yield env
    finally:
        backend.close()


def load_front_feet(env, left, right):
    env.contact_forces.zero_()
    env.contact_forces[:, env.feet_indices[0], 2] = left
    env.contact_forces[:, env.feet_indices[1], 2] = right


def test_link_offsets_reproduce_mujocos_centre_of_mass(env):
    backend = env._backend
    offset = backend.link_com_offset.expand(5, -1, -1)
    points = env._rigid_body_pos + quat_apply(env._rigid_body_quat, offset)
    mass = backend.link_mass
    com = (mass[..., None] * points).sum(dim=1) / mass.sum(dim=1, keepdim=True)
    native = np.array(
        [
            backend._datas[i].subtree_com[int(backend._model_for_env(i).jnt_bodyid[0])]
            for i in range(5)
        ]
    )
    torch.testing.assert_close(
        com, torch.tensor(native, dtype=torch.float), atol=1e-5, rtol=0
    )


def test_offset_follows_the_tilt_and_its_sign(env):
    load_front_feet(env, 80.0, 80.0)
    across, along = env.com_cop_offset.unbind(dim=1)
    # A rigid tilt swings the centre of mass on a fixed arm about the feet.
    arm = (across[4] - across[0]) / (2 * math.sin(math.radians(20)))
    assert 0.4 < arm < 0.7
    nominal = torch.asin(across[2] / arm)
    torch.testing.assert_close(
        across, arm * torch.sin(env.initial_tilt + nominal), atol=1e-4, rtol=0
    )
    assert abs(across[2]) < 0.01
    assert along.abs().max() < 1e-4


def test_support_point_is_load_weighted_between_the_front_feet(env):
    feet = env._rigid_body_pos[:, env.feet_indices[:2]]
    spacing = (feet[:, 0] - feet[:, 1]).norm(dim=1)
    load_front_feet(env, 80.0, 80.0)
    shared = env.com_cop_offset.clone()
    load_front_feet(env, 160.0, 0.0)
    left_only = env.com_cop_offset.clone()
    # All load on the left foot puts the centre of mass half a stance to its right.
    torch.testing.assert_close(
        left_only[:, 1], shared[:, 1] - spacing / 2, atol=1e-5, rtol=0
    )
    torch.testing.assert_close(left_only[:, 0], shared[:, 0], atol=1e-5, rtol=0)
    # Airborne: fall back to the midpoint instead of dividing by zero load.
    load_front_feet(env, 0.0, 0.0)
    torch.testing.assert_close(env.com_cop_offset, shared, atol=1e-5, rtol=0)


def test_rear_foot_load_does_not_move_the_support_point(env):
    load_front_feet(env, 80.0, 80.0)
    front_only = env.com_cop_offset.clone()
    env.contact_forces[:, env.feet_indices[2:], 2] = 200.0
    torch.testing.assert_close(env.com_cop_offset, front_only)


def test_penalty_grows_with_offset_and_saturates(env):
    load_front_feet(env, 80.0, 80.0)
    offset = env.com_cop_offset.norm(dim=1)
    reward = env._reward_balance_com_offset()
    scale = env.cfg.reward_settings.com_offset_scale
    torch.testing.assert_close(reward, -offset / scale)
    assert reward[2] > -0.05 and reward[0] < -0.5
    env.cfg.reward_settings.com_offset_scale = 0.05
    assert env._reward_balance_com_offset().tolist() == pytest.approx(
        [-1, -1, -offset[2].item() / 0.05, -1, -1]
    )


def test_task_adds_the_offset_without_changing_go2balance():
    _, train_cfg = task_registry.get_cfgs("go2balance_com")
    base = Go2BalanceRunnerCfg()
    assert train_cfg.actor.obs == base.actor.obs + ["com_cop_offset"]
    assert train_cfg.critic.obs == base.critic.obs + ["com_cop_offset"]
    assert train_cfg.critic.reward.weights.balance_com_offset > 0
    assert train_cfg.critic.reward.termination_weight.termination == 20.0
    # Existing balance checkpoints keep their input size and reward set.
    assert "com_cop_offset" not in base.actor.obs
    assert not hasattr(base.critic.reward.weights, "balance_com_offset")
    assert task_registry.get_task_class("go2balance_com") is Go2Balance


def test_scaled_observation_is_two_wide(env):
    load_front_feet(env, 80.0, 80.0)
    observed = env.get_state("com_cop_offset")
    assert observed.shape == (5, 2)
    torch.testing.assert_close(observed, env.com_cop_offset / 0.1)
