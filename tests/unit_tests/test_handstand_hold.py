"""go2handstandhold: hand-over resets, control path, sensors, export parity.

Environment tests need resources/handstand/handover_states.npz (tracked in
git, made by scripts/handstand_handover_states.py).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gym import GYM_ROOT_DIR
from gym.envs.go2.handstand_hold_features import (
    ACTION_CLIP,
    ACTION_SCALE,
    FRAME,
    features,
    frame_center_and_scale,
    joint_offset,
)
from go2_deploy.handstand.hold_policy import CaptureHoldDeployPolicy, mlp_from_state
from tests.unit_tests.test_handstand_deploy import random_capture_policy

DATASET = Path(GYM_ROOT_DIR) / "resources/handstand/handover_states.npz"
N = 4
ACTIVE = 8  # of random_capture_policy


def meta():
    if not DATASET.exists():
        pytest.skip("hand-over dataset not generated")
    with np.load(DATASET, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def fake_meta():
    rng = np.random.default_rng(0)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return dict(
        nominal=np.tile([0.0, 0.73, -1.47], 4),
        hold_target=np.tile([0.0, 0.73, -1.47], 4) + rng.normal(0, 0.3, 12),
        speed=np.tile([30.1, 30.1, 15.7], 4),
        target_rotation=rotation * np.sign(np.linalg.det(rotation)),
    )


# ── Shared feature definition and deploy export ──────────────────────────


def test_features_put_the_held_handstand_at_zero():
    m = fake_meta()
    center, scale = frame_center_and_scale(
        m["nominal"], m["hold_target"], m["speed"], m["target_rotation"]
    )
    offset = m["hold_target"] - m["nominal"]
    rotation = torch.tensor(m["target_rotation"])
    held = torch.cat(
        [
            rotation.T @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64),
            torch.zeros(3),
            torch.tensor(offset),
            torch.zeros(12),
            torch.tensor(offset) / 2,
        ]
    ).float()
    out = features(held.expand(3, FRAME), center, scale)
    assert out.shape == (3 * FRAME,)
    assert out.abs().max() < 1e-6
    # Joint velocity features are rad/s / 5.
    moving = held.clone()
    moving[18:30] = 5.0 / torch.tensor(m["speed"]).float()
    torch.testing.assert_close(
        features(moving[None], center, scale)[18:30], torch.ones(12)
    )
    # The action maps to an offset around the held posture, clipped.
    action = torch.full((12,), 10.0)
    torch.testing.assert_close(
        joint_offset(action, m["nominal"], m["hold_target"]),
        torch.tensor(offset).float() + ACTION_SCALE * ACTION_CLIP,
    )


def hold_policy(history=3):
    torch.manual_seed(3)
    network = torch.nn.Sequential(
        torch.nn.Linear(FRAME * (history + 1), 32),
        torch.nn.ELU(),
        torch.nn.Linear(32, 12),
    )
    m = fake_meta()
    center, scale = frame_center_and_scale(
        m["nominal"], m["hold_target"], m["speed"], m["target_rotation"]
    )
    capture = random_capture_policy()
    policy = CaptureHoldDeployPolicy(
        capture.state_dict(), network, center, scale, m["nominal"], m["hold_target"]
    ).eval()
    return policy, capture, network, center, scale, m


def test_deploy_module_is_the_kick_then_the_trained_hold():
    policy, capture, network, center, scale, m = hold_policy()
    assert policy.history == 3 and policy.num_obs == 47 + 3 * FRAME
    assert policy.handover == ACTIVE
    generator = torch.Generator().manual_seed(4)
    traced = torch.jit.trace(policy, torch.zeros(policy.num_obs))
    for step in range(20):
        obs = 0.3 * torch.randn(policy.num_obs, generator=generator)
        obs[42] = step * 0.02 / 8
        with torch.no_grad():
            if step < ACTIVE:
                expected = capture(obs[None, :47])[0]
            else:
                frames = torch.cat([obs[None, :42], obs[47:].view(3, FRAME)])
                action = network(features(frames, center, scale))
                offset = joint_offset(action, m["nominal"], m["hold_target"])
                expected = (offset / 2).clamp(-1, 1)
            torch.testing.assert_close(policy(obs), expected, atol=1e-6, rtol=0)
            torch.testing.assert_close(traced(obs), expected, atol=1e-6, rtol=0)


def test_actor_network_is_rebuilt_from_a_qgym_checkpoint():
    from learning.modules.actor import Actor

    actor = Actor(5 * FRAME, 12, [64, 32], activation="elu", normalize_obs=False)
    rebuilt = mlp_from_state(actor.state_dict())
    x = torch.randn(7, 5 * FRAME)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(x), actor.act_inference(x))


# ── Training environment ─────────────────────────────────────────────────


def make_env(**hold):
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.envs.go2.go2handstandhold import Go2HandstandHold
    from gym.envs.go2.go2handstandhold_config import (
        Go2HandstandHoldCfg,
        Go2HandstandHoldRunnerCfg,
    )
    from gym.utils.task_registry import task_registry

    meta()  # skips without the dataset
    cfg = Go2HandstandHoldCfg()
    cfg.seed = 5
    cfg.env.num_envs = N
    cfg.push_robots.toggle = False
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    for key, value in hold.items():
        setattr(cfg.handstand_hold, key, value)
    task_registry.convert_frequencies_to_params(cfg, Go2HandstandHoldRunnerCfg())
    backend = MuJocoCPUBackend()
    return Go2HandstandHold(cfg, "cpu", True, backend), backend


@pytest.fixture
def env():
    env, backend = make_env(ramp_steps=10**9)  # errors stay at zero
    try:
        yield env
    finally:
        backend.close()


ALL = torch.ones(N, dtype=torch.bool)


@torch.inference_mode()
def test_resets_place_the_robot_exactly_in_a_recorded_state(env):
    env._reset_idx(ALL)
    d = env._starts
    i = env._sample
    torch.testing.assert_close(env.dof_pos, d["qpos"][i, 7:])
    torch.testing.assert_close(env.dof_vel, d["qvel"][i, 6:])
    torch.testing.assert_close(env.root_states[:, 2], d["qpos"][i, 2])
    torch.testing.assert_close(env.hold_target, d["target"][i])
    torch.testing.assert_close(env._history, d["history"][i])
    # Body angular velocity is MuJoCo's body-frame free-joint velocity.
    torch.testing.assert_close(env.base_ang_vel, d["qvel"][i, 3:6], atol=1e-5, rtol=0)


@torch.inference_mode()
def test_frame_is_what_go2_deploy_reports(env):
    env._reset_idx(ALL)
    m = meta()
    frame = env._frame
    nominal = torch.tensor(m["nominal"]).float()
    speed = torch.tensor(m["speed"]).float()
    torch.testing.assert_close(frame[:, :3], env.projected_gravity)
    torch.testing.assert_close(frame[:, 3:6], env.base_ang_vel)
    torch.testing.assert_close(frame[:, 6:18], env.dof_pos - nominal)
    torch.testing.assert_close(frame[:, 18:30], env.dof_vel / speed)
    torch.testing.assert_close(frame[:, 30:42], (env.hold_target - nominal) / 2)
    assert env.hold_obs.shape == (N, FRAME * (env.cfg.handstand_hold.history + 1))
    assert torch.isfinite(env.hold_privileged).all()


@torch.inference_mode()
def test_targets_follow_the_deploy_clip_and_slew(env):
    env._reset_idx(ALL)
    m = meta()
    nominal = torch.tensor(m["nominal"]).float()
    lower, upper = torch.tensor(m["joint_bounds"]).float().T
    limit = 0.7 * torch.tensor(m["speed"]).float() * 0.02
    target = env.hold_target.clone()
    torch.manual_seed(6)
    for _ in range(5):
        env.hold_action = 3 * torch.randn(N, 12)
        offset = joint_offset(env.hold_action, m["nominal"], m["hold_target"])
        desired = (nominal + 2 * (offset / 2).clamp(-1, 1)).clamp(lower, upper)
        target = target + (desired - target).clamp(-limit, limit)
        env.step()
        torch.testing.assert_close(env.hold_target, target, atol=1e-5, rtol=0)
        # History: newest past frame first.
        assert env._history.shape[1] == env.cfg.handstand_hold.history


@torch.inference_mode()
def test_errors_ramp_in_and_stay_in_range():
    env, backend = make_env(ramp_steps=1)
    try:
        env.common_step_counter = 10
        env._reset_idx(ALL)
        h = env.cfg.handstand_hold
        assert env._delay.max() <= h.command_delay_substeps
        assert (env._strength >= h.strength[0]).all() and (env._strength <= 1).all()
        assert env._zero_offset.abs().max() <= h.zero_offset
        assert env._gyro_bias.abs().max() <= h.gyro_bias
        angle = torch.acos(
            ((env._attitude.diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2).clamp(-1, 1)
        )
        assert angle.max() <= h.attitude_bias * 3**0.5 + 1e-6
        assert env._gyro_bias.abs().max() > 0
        # Sensors now differ from the truth.
        assert (env._frame[:, 3:6] - env.base_ang_vel).abs().max() > 0
        env.common_step_counter = 0
        env._reset_idx(ALL)
        assert env._delay.max() == 0 and env._zero_offset.abs().max() == 0
    finally:
        backend.close()


@torch.inference_mode()
def test_rewards_are_bounded_and_registered(env):
    from gym.envs.go2.go2handstandhold_config import Go2HandstandHoldRunnerCfg

    env._reset_idx(ALL)
    env.step()
    weights = Go2HandstandHoldRunnerCfg.critic.reward.weights
    for name in vars(weights):
        if name.startswith("_"):
            continue
        value = getattr(env, f"_reward_{name}")()
        assert value.shape == (N,) and torch.isfinite(value).all(), name
    for name in ["strict_hold", "upright", "com_over_support", "heights"]:
        value = getattr(env, f"_reward_{name}")()
        assert (value >= 0).all() and (value <= 1).all(), name


@torch.inference_mode()
def test_a_diverged_world_is_reset_without_touching_the_others(env):
    env._reset_idx(ALL)
    env.step()
    healthy = env.hold_obs[1:].clone()
    env.dof_vel[0, 3] = float("nan")
    env.root_states[0, 8] = float("inf")
    env.torques[0, 0] = float("nan")
    env._guard_diverged()
    env._update_physics_summaries()
    env._measure_frame(None)
    env._check_terminations_and_timeouts()
    assert env.terminated[0] and not env._diverged[1:].any()
    assert (
        torch.isfinite(env.hold_obs).all() and torch.isfinite(env.hold_privileged).all()
    )
    torch.testing.assert_close(env.hold_obs[1:], healthy)
    from gym.envs.go2.go2handstandhold_config import Go2HandstandHoldRunnerCfg

    for name in vars(Go2HandstandHoldRunnerCfg.critic.reward.weights):
        if not name.startswith("_"):
            assert torch.isfinite(getattr(env, f"_reward_{name}")()).all(), name
    assert env.diverged_resets == 1
    env._reset_idx(env.terminated)
    assert torch.isfinite(env.dof_vel).all() and torch.isfinite(env.root_states).all()


# ── v2: gate on every physics step, unbroken hold ────────────────────────


@pytest.fixture
def env_v2():
    from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
    from gym.envs.go2.go2handstandhold_v2 import Go2HandstandHoldV2
    from gym.envs.go2.go2handstandhold_v2_config import (
        Go2HandstandHoldV2Cfg,
        Go2HandstandHoldV2RunnerCfg,
    )
    from gym.utils.task_registry import task_registry

    meta()  # skips without the dataset
    cfg = Go2HandstandHoldV2Cfg()
    cfg.seed = 5
    cfg.env.num_envs = N
    cfg.push_robots.toggle = False
    cfg.domain_randomization.startup.contact_friction_range = None
    cfg.domain_randomization.startup.link_mass_scale_range = None
    cfg.domain_randomization.episode.scale_ranges = {}
    cfg.handstand_hold.ramp_steps = 10**9
    task_registry.convert_frequencies_to_params(cfg, Go2HandstandHoldV2RunnerCfg())
    backend = MuJocoCPUBackend()
    env = Go2HandstandHoldV2(cfg, "cpu", True, backend)
    env._reset_idx(ALL)
    try:
        yield env
    finally:
        backend.close()


@torch.inference_mode()
def test_v2_a_step_holds_only_if_every_physics_step_passes(env_v2):
    env = env_v2
    passes = [torch.ones(N, dtype=torch.bool)] * 10
    passes[6] = torch.tensor([True, False, True, True])  # world 1 fails once
    calls = []

    def gate():
        calls.append(None)
        return passes[(len(calls) - 1) % 10]

    env._strict_hold = gate
    env.step()
    env.step()
    assert len(calls) == 20  # once per physics step
    torch.testing.assert_close(
        env.hold_seconds, torch.tensor([2, 0, 2, 2]) * env.dt, atol=1e-6, rtol=0
    )
    rewards = env._reward_unbroken_hold()
    assert rewards[1] == 0 and (rewards[[0, 2, 3]] > 0).all()
    env._reset_idx(torch.tensor([True, False, False, False]))
    assert env.hold_seconds[0] == 0 and env.hold_seconds[2] > 0


@torch.inference_mode()
def test_v2_a_joint_past_its_limit_on_any_physics_step_ends_the_episode(env_v2):
    env = env_v2
    calls = []
    unsafe = torch.tensor([False, False, True, False])

    def joint_unsafe():
        calls.append(None)
        return unsafe if len(calls) == 4 else torch.zeros(N, dtype=torch.bool)

    env._joint_unsafe = joint_unsafe
    env.step()
    assert env.terminated[2] and not env._gate_all[2]


@torch.inference_mode()
def test_v2_rewards_and_critic_inputs_are_finite(env_v2):
    from gym.envs.go2.go2handstandhold_v2_config import Go2HandstandHoldV2RunnerCfg

    env_v2.step()
    assert env_v2.hold_progress.shape == (N, 2)
    for name in vars(Go2HandstandHoldV2RunnerCfg.critic.reward.weights):
        if not name.startswith("_"):
            value = getattr(env_v2, f"_reward_{name}")()
            assert value.shape == (N,) and torch.isfinite(value).all(), name
    # The worst physics step is at least as bad as the last one.
    assert (env_v2._slip_peak >= env_v2._front_slip - 1e-6).all()
    assert (env_v2._front_load_low <= env_v2._loads[:, :2].amin(1) + 1e-6).all()


def test_v2_starts_from_v1s_actor():
    from gym.envs.go2.go2handstandhold_v2_config import Go2HandstandHoldV2RunnerCfg

    path = Go2HandstandHoldV2RunnerCfg.runner.init_actor_from.format(
        GYM_ROOT_DIR=GYM_ROOT_DIR
    )
    state = torch.load(path, weights_only=True)["actor_state_dict"]
    assert state["NN.0.weight"].shape[1] == FRAME * 5  # v1's observation
    assert state["NN.6.weight"].shape[0] == 12
