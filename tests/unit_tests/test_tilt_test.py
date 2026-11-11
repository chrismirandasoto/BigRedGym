"""Tilt-test initial conditions: grid assignment and planted-foot kinematics."""

import math
from types import SimpleNamespace

import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2_config import FOOT_RADIUS
from gym.envs.go2.go2balance import Go2Balance
from gym.envs.go2.go2balance_config import Go2BalanceCfg, Go2BalanceRunnerCfg
from gym.envs.go2.tilt_test import (
    sample_tilt,
    tilt_about_front_feet,
    validate_tilt_test,
)
from gym.utils.task_registry import task_registry
from gym.utils.torch_quat import quat_apply


def settings(**changes):
    values = dict(
        enabled=True,
        angle_degrees=[-20.0, 20.0],
        angular_velocity=[-2.0, 2.0],
        sampling="grid",
        grid=[3, 3],
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_grid_assigns_one_cell_per_robot_and_wraps():
    angle, rate = sample_tilt(torch.arange(10), settings(), "cpu")
    torch.testing.assert_close(
        torch.rad2deg(angle),
        torch.tensor([-20.0, -20, -20, 0, 0, 0, 20, 20, 20, -20]),
        atol=1e-4,
        rtol=0,
    )
    torch.testing.assert_close(
        rate, torch.tensor([-2.0, 0, 2, -2, 0, 2, -2, 0, 2, -2]), atol=1e-6, rtol=0
    )
    # A robot's cell depends only on its id, not on which robots reset with it.
    alone = sample_tilt(torch.tensor([5]), settings(), "cpu")
    assert alone[0].item() == pytest.approx(0, abs=1e-6)
    assert alone[1].item() == pytest.approx(2)


def test_single_cell_axis_uses_the_middle_of_its_range():
    cfg = settings(grid=[1, 3], angle_degrees=[4.0, 8.0])
    angle, rate = sample_tilt(torch.arange(3), cfg, "cpu")
    torch.testing.assert_close(
        torch.rad2deg(angle), torch.full((3,), 6.0), atol=1e-4, rtol=0
    )
    assert rate.tolist() == pytest.approx([-2, 0, 2])


def test_uniform_draws_stay_inside_both_ranges():
    torch.manual_seed(0)
    angle, rate = sample_tilt(torch.arange(500), settings(sampling="uniform"), "cpu")
    assert angle.abs().max() <= math.radians(20) and rate.abs().max() <= 2
    assert angle.std() > 0.1 and rate.std() > 0.5


@pytest.mark.parametrize(
    "changes",
    [
        dict(angle_degrees=[10.0, -10.0]),
        dict(angular_velocity=[0.0, float("inf")]),
        dict(sampling="sobol"),
        dict(grid=[0, 3]),
        dict(grid=[3]),
    ],
)
def test_invalid_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        validate_tilt_test(settings(**changes))


def test_rigid_tilt_keeps_front_feet_planted_and_stationary():
    # Level body, front feet 0.5 m below the base and 0.2 m ahead of it.
    feet = torch.tensor(
        [[[0.2, 0.1, -0.5], [0.2, -0.1, -0.5], [-0.2, 0.1, -0.5], [-0.2, -0.1, -0.5]]]
    )
    root = torch.zeros(1, 13)
    root[0, 2] = 0.5
    root[0, 6] = 1
    angle, rate = torch.tensor([math.radians(30)]), torch.tensor([2.0])
    tilted = tilt_about_front_feet(root, feet, angle, rate)
    world = tilted[:, None, :3] + quat_apply(
        tilted[:, None, 3:7].expand(-1, 4, -1), feet
    )
    torch.testing.assert_close(
        world[0, :2], feet[0, :2] + root[0, :3], atol=1e-6, rtol=0
    )
    velocity = tilted[:, None, 7:10] + torch.cross(
        tilted[:, None, 10:13].expand(-1, 4, -1), world - tilted[:, None, :3], dim=-1
    )
    torch.testing.assert_close(velocity[0, :2], torch.zeros(2, 3), atol=1e-6, rtol=0)
    # Rear feet sit 0.4 m from the support line and swing about it.
    assert velocity[0, 2:].norm(dim=1).tolist() == pytest.approx([0.8, 0.8])
    # Positive tilt pitches nose-down about the body's left axis.
    torch.testing.assert_close(
        tilted[0, 10:13], torch.tensor([0.0, 2.0, 0.0]), atol=1e-6, rtol=0
    )
    nose = quat_apply(tilted[:, 3:7], torch.tensor([[1.0, 0.0, 0.0]]))[0]
    assert nose[2].item() == pytest.approx(-0.5)
    # The base swings forward and up over the pivot; hand-computed.
    assert tilted[0, :3].tolist() == pytest.approx([0.2768, 0, 0.5330], abs=1e-4)


@pytest.fixture
def env():
    cfg = Go2BalanceCfg()
    cfg.env.num_envs = 9
    cfg.seed = 17
    cfg.tilt_test.enabled = True
    cfg.tilt_test.grid = [3, 3]
    cfg.tilt_test.angle_degrees = [-20.0, 20.0]
    cfg.tilt_test.angular_velocity = [-2.0, 2.0]
    task_registry.convert_frequencies_to_params(cfg, Go2BalanceRunnerCfg())
    backend = MuJocoCPUBackend()
    try:
        yield Go2Balance(cfg, "cpu", True, backend)
    finally:
        backend.close()


def test_native_reset_matches_the_requested_tilt_with_stationary_feet(env):
    env._reset_idx(torch.ones(9, dtype=torch.bool))
    angle, rate = env.initial_tilt, env.initial_tilt_rate
    assert torch.rad2deg(angle).tolist() == pytest.approx(
        [-20, -20, -20, 0, 0, 0, 20, 20, 20], abs=1e-4
    )
    assert rate.tolist() == pytest.approx([-2, 0, 2] * 3)
    # MuJoCo's own kinematics, not the reset's analytic leg model.
    feet = env._rigid_body_pos[:, env.feet_indices]
    speed = env._rigid_body_lin_vel[:, env.feet_indices].norm(dim=-1)
    height = FOOT_RADIUS + env.cfg.handstand_curriculum.ground_clearance
    torch.testing.assert_close(
        feet[:, :2, 2], torch.full((9, 2), height), atol=1e-4, rtol=0
    )
    assert speed[:, :2].max() < 1e-4
    assert torch.all(speed[rate != 0, 2:] > 1)
    # Nose-down top pose is gravity (+1, 0, 0); over-rotation turns the back down.
    expected = torch.stack((angle.cos(), torch.zeros(9), angle.sin()), dim=1)
    torch.testing.assert_close(env.projected_gravity, expected, atol=1e-4, rtol=0)
    torch.testing.assert_close(env.base_ang_vel[:, 1], rate, atol=1e-4, rtol=0)
    assert env.base_ang_vel[:, [0, 2]].abs().max() < 1e-4
    assert env.handstand_stage.tolist() == [2] * 9


def test_partial_reset_leaves_other_robots_untouched(env):
    env._reset_idx(torch.ones(9, dtype=torch.bool))
    before = env.root_states.clone()
    mask = torch.zeros(9, dtype=torch.bool)
    mask[7] = True
    env._reset_idx(mask)
    torch.testing.assert_close(env.root_states[~mask], before[~mask])
    assert math.degrees(env.initial_tilt[7]) == pytest.approx(20, abs=1e-4)
    assert env.initial_tilt_rate[7].item() == pytest.approx(0, abs=1e-6)
