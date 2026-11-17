"""Sensor/actuator error model and the trainable capture policy."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.handstand_robustness.randomization import (
    PROFILES,
    SWEEPABLE,
    RandomizedEnvironment,
    SensorErrors,
    draw,
    scaled,
)
from gym.envs.go2.handstand_replay import build_model, load_trajectory


def fixture_trajectory():
    path = os.getenv("GO2_HANDSTAND_TRAJECTORY")
    digest = os.getenv("GO2_HANDSTAND_SHA256")
    if not path or not digest:
        pytest.skip("Provide external hash-locked reference")
    model = build_model()
    return model, load_trajectory(Path(path), digest, model)


def test_draw_is_reproducible_bounded_and_independent_of_other_errors():
    ranges = PROFILES["both"]
    first, again, other = draw(ranges, 5), draw(ranges, 5), draw(ranges, 6)
    assert first == again and first != other
    assert np.abs(first["gyro_bias"]).max() <= ranges["gyro_bias"]
    assert len(first["zero_offset"]) == 12
    assert 0 <= first["gyro_noise"] <= ranges["gyro_noise"]
    assert 0.95 <= first["strength"] <= 1.0
    assert 0.95 <= first["kp_scale"] <= 1.05
    assert first["command_delay_substeps"] in (0, 1, 2)
    # Start-state scatter passes through for the environment to draw itself.
    assert first["joint_position"] == 0.003
    assert draw({}, 5) == {}
    with pytest.raises(KeyError, match="unknown"):
        draw({"gyro_bais": 0.1}, 0)
    assert set(SWEEPABLE) <= set(PROFILES["both"]) | {"accel_bias", "height_error"}


def test_scaled_grows_every_error_but_not_the_start_scatter():
    double = scaled(PROFILES["both"], 2.0)
    assert double["gyro_bias"] == 2 * PROFILES["both"]["gyro_bias"]
    assert double["strength"] == pytest.approx([0.9, 1.0])
    assert double["joint_position"] == PROFILES["both"]["joint_position"]
    assert scaled(PROFILES["both"], 0.0)["kp_scale"] == 0


def test_sensor_errors_apply_bias_noise_and_attitude_error():
    errors = SensorErrors(
        dict(
            gyro_bias=[0.01, 0, 0],
            accel_bias=[0, 0.1, 0],
            orientation_bias=[0, 0.02, 0],
            zero_offset=[0.005] * 12,
            height_error=[0.003],
        ),
        seed=1,
    )
    assert errors.active
    np.testing.assert_allclose(errors.gyro(np.zeros(3)), [0.01, 0, 0])
    np.testing.assert_allclose(errors.accel(np.zeros(3)), [0, 0.1, 0])
    q, dq = errors.joints(np.zeros(12), np.ones(12))
    np.testing.assert_allclose(q, 0.005)
    np.testing.assert_allclose(dq, 1.0)
    tilt = errors.rotation(np.eye(3))
    assert np.arccos(tilt[2, 2]) == pytest.approx(0.02)
    assert errors.height_error == 0.003
    assert not SensorErrors({}, seed=1).active
    noisy = SensorErrors(dict(gyro_noise=0.01), seed=1)
    samples = np.array([noisy.gyro(np.zeros(3)) for _ in range(2000)])
    assert samples.std() == pytest.approx(0.01, rel=0.1)


def test_no_errors_reproduces_the_nominal_environment_exactly():
    from experiments.handstand_robustness.capture import CaptureTrainingEnvironment

    model, t = fixture_trajectory()
    condition = {"joint_position": 0.003, "joint_velocity": 0.03}
    nominal = CaptureTrainingEnvironment(model, t, 11, condition, horizon=30)
    randomized = RandomizedEnvironment(
        build_model(), t, 11, draw(condition, 11), horizon=30
    )
    for k in range(30):
        np.testing.assert_array_equal(nominal.observation(), randomized.observation())
        nominal.step(t["actions"][k])
        randomized.step(t["actions"][k])
    np.testing.assert_array_equal(nominal.d.qpos, randomized.d.qpos)


def test_each_error_changes_what_it_should():
    model, t = fixture_trajectory()

    def after(condition, steps=20):
        env = RandomizedEnvironment(build_model(), t, 3, condition, horizon=steps)
        for k in range(steps):
            env.step(t["actions"][k])
        return env

    base = after({})
    reference = base.observation()
    # Sensor errors change the observation, never the physics.
    for condition, channels in [
        (dict(gyro_bias=[0.02, 0, 0]), slice(3, 6)),
        (dict(orientation_bias=[0, 0.02, 0]), slice(0, 3)),
        (dict(accel_bias=[0.3, 0, 0]), slice(44, 47)),
        (dict(height_error=[0.01]), slice(43, 44)),
    ]:
        env = after(condition)
        np.testing.assert_array_equal(env.d.qpos, base.d.qpos)
        assert np.abs(env.observation()[channels] - reference[channels]).max() > 1e-3
    drifted = after(dict(accel_bias=[0.3, 0, 0]))
    assert drifted.max_observer_velocity_error == pytest.approx(0.3 * 0.4, rel=0.05)
    # Actuator errors change the physics.
    for condition in [
        dict(kp_scale=0.8),
        dict(kd_scale=1.3),
        dict(command_delay_substeps=5),
        dict(joint_damping=0.2),
        dict(zero_offset=[0.02] * 12),
    ]:
        assert np.abs(after(condition).d.qpos - base.d.qpos).max() > 1e-5


def test_trainable_policy_starts_as_the_baseline_and_exports():
    from experiments.handstand_robustness.randomized_training import RobustCapturePolicy
    from go2_deploy.handstand.policy import CaptureDeployPolicy
    from tests.unit_tests.test_handstand_deploy import random_capture_policy

    base = random_capture_policy()
    state = base.state_dict()
    balance = {
        k: state["balance_" + k].numpy() for k in ["weight", "bias", "lower", "upper"]
    }
    robust = RobustCapturePolicy(
        state["actions"], state["centers"], balance, int(state["active_steps"])
    )
    robust.load_state_dict(state)
    assert "balance_weight" in dict(robust.named_parameters())
    deploy = CaptureDeployPolicy(robust.state_dict())
    generator = torch.Generator().manual_seed(8)
    for step in range(30):
        obs = 0.3 * torch.randn(1, 47, generator=generator)
        obs[0, 42] = step * 0.02 / 8
        with torch.no_grad():
            expected = base(obs)
            torch.testing.assert_close(robust(obs), expected)
            # The stochastic form used by PPO, at its mean, is the same law.
            mean = robust.action(obs, robust.latent_mean(obs))
            torch.testing.assert_close(mean, expected, atol=1e-6, rtol=0)
            torch.testing.assert_close(deploy(obs[0]), expected[0], atol=1e-6, rtol=0)
            assert robust.value(obs).shape == (1,)
    # Both phases receive gradient.
    obs = 0.3 * torch.randn(8, 47, generator=generator)
    obs[:, 42] = torch.tensor([0, 1, 2, 3, 10, 12, 15, 25]) * 0.02 / 8
    robust.distribution(obs).log_prob(torch.zeros(8, 12)).sum().backward()
    assert (
        robust.gains.grad.abs().sum() > 0 and robust.balance_weight.grad.abs().sum() > 0
    )
