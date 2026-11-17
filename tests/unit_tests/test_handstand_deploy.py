"""Robot-side handstand code: kinematics, estimator, maneuver, policy, hooks.

Everything except the last test runs without the released artifacts. That one
flies the whole maneuver through the deploy stack in simulation and needs the
hash-locked reference (GO2_HANDSTAND_TRAJECTORY / GO2_HANDSTAND_SHA256) plus a
capture checkpoint (GO2_HANDSTAND_CHECKPOINT).
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from go2_deploy.handstand.estimator import GRAVITY, HandstandEstimator
from go2_deploy.handstand.kinematics import KinematicModel, extract_tree, quat_to_matrix
from go2_deploy.handstand.maneuver import Maneuver, Stage
from go2_deploy.handstand.policy import CaptureDeployPolicy
from go2_deploy.state import State
from go2_deploy.utility.import_custom_impl import get_custom_impl_module
from gym.envs.go2.handstand_replay import build_model

JOINTS = [
    f"{leg}_{part}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for part in ("hip", "thigh", "calf")
]
UNITREE = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]


@pytest.fixture(scope="module")
def robot():
    """MuJoCo Go2, its exported tree, and index helpers."""
    model = build_model()
    data = mujoco.MjData(model)
    joints = [model.joint(n).id for n in JOINTS]
    feet = [
        np.flatnonzero(model.geom_bodyid == model.body(f"{leg}_foot").id)[0]
        for leg in ("FL", "FR", "RL", "RR")
    ]
    return SimpleNamespace(
        m=model,
        d=data,
        qids=model.jnt_qposadr[joints],
        vids=model.jnt_dofadr[joints],
        bounds=model.jnt_range[joints],
        base=model.body("base").id,
        feet=feet,
        tree=extract_tree(model, JOINTS, feet),
    )


def random_pose(robot, rng):
    d = robot.d
    d.qpos[:3] = rng.normal(size=3)
    quat = rng.normal(size=4)
    d.qpos[3:7] = quat / np.linalg.norm(quat)
    d.qpos[robot.qids] = rng.uniform(robot.bounds[:, 0], robot.bounds[:, 1])
    d.qvel[:] = rng.normal(size=robot.m.nv)
    mujoco.mj_forward(robot.m, d)
    return d.xmat[robot.base].reshape(3, 3)


def assets(robot):
    # The posture the deploy config is built around, as a bundle would carry it.
    nominal = np.array(
        json.loads(Path("configs/handstand_reference.json").read_text())["nominal"]
    )
    return dict(
        nominal=nominal.tolist(),
        initial_target=nominal.tolist(),
        initial_joints=(nominal + 0.001).tolist(),
        standing_target=(nominal + np.tile([0.0, 0.0, 0.09], 4)).tolist(),
        joint_bounds=robot.bounds.tolist(),
        speed=np.tile([30.1, 30.1, 15.7], 4).tolist(),
        kinematics=robot.tree,
        height_offset=0.0,
        imu_offset=[-0.02557, 0.0, 0.04232],
    )


# ── Kinematics ───────────────────────────────────────────────────────────


def test_kinematic_tree_matches_mujoco_everywhere_in_the_joint_range(robot):
    model = KinematicModel(json.loads(json.dumps(robot.tree)))  # as the robot loads it
    rng = np.random.default_rng(0)
    for _ in range(50):
        rotation = random_pose(robot, rng)
        d = robot.d
        centre, jacobian = model.com(d.qpos[robot.qids])
        expected = np.zeros((3, robot.m.nv))
        mujoco.mj_jacSubtreeCom(robot.m, d, expected, robot.base)
        np.testing.assert_allclose(
            rotation @ centre, d.subtree_com[robot.base] - d.qpos[:3], atol=1e-12
        )
        np.testing.assert_allclose(
            rotation @ jacobian, expected[:, robot.vids], atol=1e-12
        )
        feet = d.qpos[:3] + model.feet(d.qpos[robot.qids]) @ rotation.T
        np.testing.assert_allclose(feet, d.geom_xpos[robot.feet], atol=1e-12)
        np.testing.assert_allclose(quat_to_matrix(d.qpos[3:7]), rotation, atol=1e-12)


# ── Estimator ────────────────────────────────────────────────────────────


def test_com_estimate_is_the_true_com_when_the_root_estimate_is_true(robot):
    """Height and world velocity of the COM from IMU-frame quantities only."""
    estimator = HandstandEstimator(KinematicModel(robot.tree))
    rng = np.random.default_rng(1)
    for _ in range(20):
        random_pose(robot, rng)
        d = robot.d
        mujoco.mj_subtreeVel(robot.m, d)
        estimator.velocity = d.qvel[:3].copy()
        estimator.height = float(d.qpos[2])
        estimate = estimator.com(
            d.qpos[robot.qids], d.qvel[robot.vids], d.qvel[3:6], d.qpos[3:7]
        )
        np.testing.assert_allclose(
            estimate[0] * 0.6, d.subtree_com[robot.base, 2], atol=1e-10
        )
        np.testing.assert_allclose(
            estimate[1:], d.subtree_linvel[robot.base], atol=1e-10
        )


def rigid_motion(steps=400, dt=0.002):
    """A tumbling, accelerating body: origin acceleration, attitude, rates."""
    rng = np.random.default_rng(2)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    rate = np.array([0.3, -0.5, 0.2])
    samples = []
    for k in range(steps):
        alpha = 3.0 * np.array([np.sin(0.05 * k), np.cos(0.03 * k), 0.5])
        acceleration = np.array([0.5 * np.sin(0.02 * k), 0.2, 1.0 * np.cos(0.04 * k)])
        rate = rate + dt * alpha
        step = np.zeros(4)
        mujoco.mju_axisAngle2Quat(
            step, rate / np.linalg.norm(rate), np.linalg.norm(rate) * dt
        )
        new = np.zeros(4)
        mujoco.mju_mulQuat(new, quat, step)
        quat = new / np.linalg.norm(new)
        samples.append((acceleration, quat.copy(), rate.copy(), alpha))
    return samples, dt, rng


def test_integrating_an_ideal_imu_recovers_the_root_velocity_and_height():
    samples, dt, _ = rigid_motion()
    estimator = HandstandEstimator(model=None)
    estimator.height = 0.34
    velocity, height = np.zeros(3), 0.34
    for acceleration, quat, rate, _ in samples:
        rotation = quat_to_matrix(quat)
        force = rotation.T @ (acceleration + [0, 0, GRAVITY])
        estimator.update(force, rate, quat, dt)
        velocity = velocity + dt * acceleration
        height += dt * velocity[2]
    np.testing.assert_allclose(estimator.velocity, velocity, atol=1e-12)
    assert estimator.height == pytest.approx(height, abs=1e-12)


def test_lever_arm_correction_removes_the_imu_mounting_offset():
    samples, dt, _ = rigid_motion()
    offset = np.array([-0.02557, 0.0, 0.04232])
    corrected = HandstandEstimator(model=None, imu_offset=offset)
    ignored = HandstandEstimator(model=None)
    velocity = np.zeros(3)
    for acceleration, quat, rate, alpha in samples:
        rotation = quat_to_matrix(quat)
        at_origin = rotation.T @ (acceleration + [0, 0, GRAVITY])
        at_imu = (
            at_origin + np.cross(alpha, offset) + np.cross(rate, np.cross(rate, offset))
        )
        corrected.update(at_imu, rate, quat, dt)
        ignored.update(at_imu, rate, quat, dt)
        velocity = velocity + dt * acceleration
    # The angular acceleration is a finite difference of the gyro: exact here
    # except for the first sample, which has no predecessor.
    assert np.abs(corrected.velocity - velocity).max() < 1e-3
    assert (
        np.abs(ignored.velocity - velocity).max()
        > 5 * np.abs(corrected.velocity - velocity).max()
    )


def test_stationary_calibration_finds_gyro_and_accelerometer_bias(robot):
    estimator = HandstandEstimator(KinematicModel(robot.tree))
    quat = np.array([0.999, 0.02, -0.03, 0.01])
    quat /= np.linalg.norm(quat)
    rest = quat_to_matrix(quat).T @ [0, 0, GRAVITY]
    gyro_bias, accel_bias = (
        np.array([0.004, -0.002, 0.001]),
        np.array([0.05, -0.02, 0.08]),
    )
    estimator.clear_calibration(500)
    assert not estimator.calibration_ready()
    for _ in range(500):
        estimator.add_calibration_sample(rest + accel_bias, gyro_bias, quat)
    assert estimator.calibration_ready()
    assert estimator.calibration_drift() < 1e-12
    q = np.tile([0.0, 0.7334, -1.4655], 4)
    estimator.start(q, quat)
    np.testing.assert_allclose(estimator.gyro_bias, gyro_bias, atol=1e-12)
    np.testing.assert_allclose(estimator.accel_bias, accel_bias, atol=1e-12)
    # With the biases removed a stationary robot stays stationary.
    for _ in range(500):
        estimator.update(rest + accel_bias, gyro_bias, quat, 0.002)
    assert np.abs(estimator.velocity).max() < 1e-9
    np.testing.assert_allclose(estimator.rate(gyro_bias), 0, atol=1e-12)


def test_calibration_drift_flags_a_body_that_is_still_settling():
    estimator = HandstandEstimator(model=None)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    estimator.clear_calibration(500)
    for k in range(500):
        moving = np.array([0.02 if k < 250 else 0.0, 0.0, GRAVITY])
        estimator.add_calibration_sample(moving, np.zeros(3), quat)
    assert estimator.calibration_drift() == pytest.approx(0.02)


def test_standing_height_puts_the_lowest_foot_on_the_ground(robot):
    estimator = HandstandEstimator(KinematicModel(robot.tree))
    d = robot.d
    mujoco.mj_resetData(robot.m, d)
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[robot.qids] = np.tile([0.0, 0.7334, -1.4655], 4)
    d.qpos[robot.qids[2]] -= 0.1  # one leg slightly longer
    mujoco.mj_forward(robot.m, d)
    lowest = d.geom_xpos[robot.feet, 2].min()
    height = estimator.standing_height(d.qpos[robot.qids], d.qpos[3:7])
    assert height == pytest.approx(d.qpos[2] - lowest + 0.022, abs=1e-12)


# ── Maneuver ─────────────────────────────────────────────────────────────


def still(maneuver, seconds, gyro=np.zeros(3), q=None):
    """Stand still for a while, feeding the IMU at 500 Hz; last applied offset."""
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    q = maneuver.initial_joints if q is None else q
    offset = None
    for _ in range(round(seconds / 0.02)):
        for _ in range(10):
            maneuver.on_imu(np.array([0, 0, GRAVITY]), gyro, quat, 0.002)
        offset = maneuver.step(np.zeros(12), q, np.zeros(12), gyro, quat)
    return offset


def make_maneuver(robot):
    a = assets(robot)
    return Maneuver(a, HandstandEstimator(KinematicModel(robot.tree))), a


def test_prepare_moves_slowly_to_the_standing_target_then_waits_for_a_trigger(robot):
    maneuver, a = make_maneuver(robot)
    start = np.asarray(a["standing_target"]) + 0.3
    maneuver.activate(start)
    assert maneuver.stage == Stage.PREPARE and not maneuver.trigger()
    first = maneuver.step(
        np.ones(12), start, np.zeros(12), np.zeros(3), np.array([1.0, 0, 0, 0])
    )
    # The policy's output is ignored while preparing: 0.5 rad/s for 20 ms.
    np.testing.assert_allclose(first + maneuver.nominal, start - 0.01, atol=1e-12)
    still(maneuver, 6.0)
    assert maneuver.stage == Stage.READY
    np.testing.assert_allclose(maneuver.target, a["standing_target"], atol=1e-12)
    assert maneuver.elapsed() == 0
    # Standing does not start the handstand; the operator does.
    still(maneuver, 1.0)
    assert maneuver.stage == Stage.READY
    assert maneuver.trigger()
    still(maneuver, 0.02)
    assert maneuver.stage == Stage.RUN and maneuver.step_count == 0
    # Training starts with the target at the start posture, not the standing one.
    np.testing.assert_allclose(maneuver.target, a["initial_target"], atol=1e-12)


@pytest.mark.parametrize(
    "disturbance, reason",
    [
        (dict(gyro=np.array([0.0, 0.3, 0.0])), "rotating"),
        (dict(q=np.tile([0.0, 0.9, -1.4655], 4)), "posture"),
    ],
)
def test_ready_requires_a_still_robot_in_the_training_posture(
    robot, disturbance, reason
):
    maneuver, a = make_maneuver(robot)
    maneuver.activate(np.asarray(a["standing_target"]))
    still(maneuver, 7.0, **disturbance)
    assert maneuver.stage == Stage.PREPARE
    assert reason in maneuver.status
    assert not maneuver.trigger()


def test_running_targets_follow_the_training_clip_and_slew(robot):
    """Same filter as Environment.step: clip to limits, then 0.7 * speed * dt."""
    maneuver, a = make_maneuver(robot)
    quat = np.array([1.0, 0, 0, 0])
    maneuver.start_now(maneuver.initial_joints, quat, height=0.34)
    nominal, speed = maneuver.nominal, maneuver.speed
    lower, upper = robot.bounds.T
    target = np.asarray(a["initial_target"]).copy()
    rng = np.random.default_rng(3)
    for k in range(50):
        action = rng.uniform(-1.5, 1.5, 12)  # the policy clamps; the filter must too
        desired = np.clip(nominal + 2 * np.clip(action, -1, 1), lower, upper)
        target += np.clip(desired - target, -0.7 * speed * 0.02, 0.7 * speed * 0.02)
        maneuver.on_imu(np.array([0, 0, GRAVITY]), np.zeros(3), quat, 0.002)
        offset = maneuver.step(
            2 * action, maneuver.initial_joints, np.zeros(12), np.zeros(3), quat
        )
        np.testing.assert_allclose(offset + nominal, target, atol=1e-12)
        assert maneuver.step_count == k + 1
        assert maneuver.elapsed() == pytest.approx((k + 1) * 0.02 / 8)
    assert (target >= lower - 1e-12).all() and (target <= upper + 1e-12).all()


def test_lost_or_invalid_imu_data_during_the_maneuver_is_a_fault(robot):
    quat = np.array([1.0, 0, 0, 0])
    maneuver, _ = make_maneuver(robot)
    maneuver.start_now(maneuver.initial_joints, quat, height=0.34)
    maneuver.on_imu(np.array([0, 0, GRAVITY]), np.zeros(3), quat, 0.2)
    assert "gap" in maneuver.fault

    maneuver.start_now(maneuver.initial_joints, quat, height=0.34)
    maneuver.on_imu(np.array([0, 0, np.nan]), np.zeros(3), quat, 0.002)
    assert "non-finite" in maneuver.fault

    maneuver.start_now(maneuver.initial_joints, quat, height=0.34)
    for _ in range(5):  # commands keep going out but no IMU sample arrives
        maneuver.step(
            np.zeros(12), maneuver.initial_joints, np.zeros(12), np.zeros(3), quat
        )
    assert "no IMU" in maneuver.fault


# ── Policy ───────────────────────────────────────────────────────────────


def random_capture_policy(steps=20, active=8):
    from experiments.handstand_robustness.capture import CapturePolicy

    generator = torch.Generator().manual_seed(4)
    balance = dict(
        weight=0.5 * torch.randn(12, 34, generator=generator).numpy(),
        bias=np.zeros(12),
        lower=-np.ones(12),
        upper=np.ones(12),
    )
    policy = CapturePolicy(
        0.9 * (2 * torch.rand(steps, 12, generator=generator) - 1),
        0.3 * torch.randn(steps, 47, generator=generator),
        balance,
        active_steps=active,
    )
    with torch.no_grad():
        policy.gains.copy_(0.5 * torch.randn(steps, 12, 46, generator=generator))
    return policy.eval()


def test_deploy_policy_equals_the_training_policy_at_every_phase_step(tmp_path):
    """Including the feedback hand-over and the held last step, after tracing."""
    reference = random_capture_policy()
    policy = CaptureDeployPolicy(reference.state_dict()).eval()
    traced = torch.jit.trace(policy, torch.zeros(47))
    torch.jit.save(traced, tmp_path / "policy.pt")
    traced = torch.jit.load(tmp_path / "policy.pt")
    generator = torch.Generator().manual_seed(5)
    for step in range(30):  # 20 recorded steps, hand-over at 8, then held
        obs = 0.3 * torch.randn(47, generator=generator)
        obs[42] = step * 0.02 / 8
        with torch.no_grad():
            expected = reference(obs[None])[0]
            torch.testing.assert_close(policy(obs), expected, atol=1e-6, rtol=0)
            torch.testing.assert_close(traced(obs), expected, atol=1e-6, rtol=0)
    assert expected.abs().max() <= 1


def test_deploy_policy_rejects_other_checkpoint_kinds():
    state = random_capture_policy().state_dict()
    state["gains"] = state["gains"][:, :, :42]
    with pytest.raises(ValueError, match="capture_residual_v1"):
        CaptureDeployPolicy(state)


# ── Framework hooks ──────────────────────────────────────────────────────


@pytest.fixture
def handstand(robot, tmp_path, monkeypatch):
    """The custom implementation with a controller-shaped object around it."""
    module = get_custom_impl_module("handstand_capture.py")
    path = tmp_path / "handstand_assets.json"
    path.write_text(json.dumps(assets(robot)))
    monkeypatch.setenv("GO2_HANDSTAND_ASSETS", str(path))
    custom = module.CustomDeployConfig
    sizes = dict(projected_gravity=3, dof_pos_obs=12, dof_vel=12, dof_pos_target=12)
    sizes.update(custom.custom_obs_sizes)
    cfg = SimpleNamespace(
        obs_vector=custom.obs_vector,
        obs_sizes=sizes,
        DeployScaling=custom.DeployScaling,
        default_dof_pos=torch.tensor(custom.default_dof_pos, dtype=torch.float32),
        ctrl_freq=custom.ctrl_freq,
        exp_moving_avg=custom.exp_moving_avg,
        bundle_dir=None,
        bundle_meta=None,
    )
    stops = []
    controller = SimpleNamespace(
        cfg=cfg,
        obs_vec_size=sum(sizes[name] for name in cfg.obs_vector),
        last_action=torch.zeros(2, 12),
        last_lowstate_msg=None,
        remote_controller=SimpleNamespace(Y=[0, 0]),
        _state=State.RECOVERY,
        request_emergency_stop=lambda: stops.append(True),
        stops=stops,
    )
    for key, value in module.get_custom_buffers_dict().items():
        setattr(controller, key, value)
    return module, controller


def message(
    q,
    dq,
    tick,
    gyro=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
    accel=(0.0, 0.0, GRAVITY),
):
    """LowState-shaped message from policy-order joints."""
    return SimpleNamespace(
        tick=tick,
        # Plain Python floats, as the Unitree message holds them.
        imu_state=SimpleNamespace(
            quaternion=[float(x) for x in quat],
            gyroscope=[float(x) for x in gyro],
            accelerometer=[float(x) for x in accel],
        ),
        motor_state=[
            SimpleNamespace(q=float(q[i]), dq=float(dq[i]), ddq=0.0) for i in UNITREE
        ],
    )


def test_handstand_config_is_47_channels_at_50_hz_without_smoothing(handstand):
    module, controller = handstand
    custom = module.CustomDeployConfig
    assert controller.obs_vec_size == 47
    assert custom.obs_vector == [
        "projected_gravity",
        "hs_gyro",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "hs_elapsed",
        "hs_com",
    ]
    assert (custom.ctrl_freq, custom.kp, custom.kd, custom.exp_moving_avg) == (
        50,
        60.0,
        3.0,
        False,
    )
    assert custom.DeployScaling.dof_pos_target == 2.0
    assert custom.DeployScaling.dof_vel == [30.1, 30.1, 15.7] * 4
    pieces = set(module.get_custom_obs_pieces())
    # hs_history is offered for history policies but not in the default vector.
    assert pieces == {"hs_gyro", "hs_elapsed", "hs_com", "hs_history"}
    assert "hs_history" not in custom.obs_vector


def test_observation_layout_matches_the_training_contract(handstand, monkeypatch):
    """Every slice of the 47 holds the quantity, order and scale training used."""
    from go2_deploy.utility import deploy_utility

    module, controller = handstand
    monkeypatch.setattr(deploy_utility, "custom_impl", module)
    maneuver = module._maneuver(controller)
    nominal = maneuver.nominal
    rng = np.random.default_rng(6)
    q, dq = nominal + rng.normal(0, 0.1, 12), rng.normal(0, 2.0, 12)
    gyro = np.array([0.2, -0.4, 0.1])
    quat = np.array([0.98, 0.1, -0.15, 0.05])
    quat /= np.linalg.norm(quat)
    maneuver.start_now(q, quat, height=0.34)
    maneuver.estimator.velocity = np.array([0.3, -0.1, 0.2])
    maneuver.step_count = 37
    maneuver.target = nominal + rng.normal(0, 0.2, 12)
    controller.last_action[0] = torch.tensor(
        maneuver.target - nominal, dtype=torch.float32
    )
    msg = message(q, dq, tick=2, gyro=gyro, quat=quat)

    obs = deploy_utility.lowstate_to_obs(controller, msg).numpy()

    rotation = quat_to_matrix(quat)
    np.testing.assert_allclose(obs[0:3], rotation.T @ [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(obs[3:6], gyro, atol=1e-6)
    np.testing.assert_allclose(obs[6:18], q - nominal, atol=1e-6)
    np.testing.assert_allclose(
        obs[18:30], dq / np.tile([30.1, 30.1, 15.7], 4), atol=1e-6
    )
    np.testing.assert_allclose(obs[30:42], (maneuver.target - nominal) / 2, atol=1e-6)
    assert obs[42] == pytest.approx(37 * 0.02 / 8, abs=1e-7)
    np.testing.assert_allclose(
        obs[43:47], maneuver.estimator.com(q, dq, gyro, quat), atol=1e-6
    )
    assert round(float(obs[42]) * 400) == 37  # the policy's table index


def test_hooks_run_the_operator_sequence_and_integrate_every_sample_once(handstand):
    module, controller = handstand
    maneuver = module._maneuver(controller)
    q, dq = maneuver.initial_joints, np.zeros(12)
    tick = 0

    def period(action=None, y=0):
        nonlocal tick
        for _ in range(10):
            tick += 2
            controller.remote_controller.Y = [y, 0]
            controller.last_lowstate_msg = message(q, dq, tick)
            module.on_lowstate(controller, controller.last_lowstate_msg, tick * 1e-3)
        module.update_custom_buffers(controller, tick * 1e-3)
        if controller._state == State.CUSTOM_CTRL:
            given = torch.zeros(12) if action is None else action
            controller.last_action[0] = module.process_action(controller, given)

    period()
    assert maneuver.stage == Stage.IDLE
    controller._state = State.CUSTOM_CTRL
    period(y=1)  # Y before the robot is ready does nothing
    assert maneuver.stage == Stage.PREPARE
    for _ in range(400):
        period()
    assert maneuver.stage == Stage.READY
    module.on_key(controller, "x")
    assert not maneuver.triggered
    period(y=1)
    assert maneuver.stage == Stage.RUN and maneuver.step_count == 0
    height = maneuver.estimator.height
    # A duplicate message (same tick) must not be integrated twice.
    module.on_lowstate(controller, controller.last_lowstate_msg, tick * 1e-3)
    period(action=torch.full((12,), 0.2))
    assert maneuver.step_count == 1
    assert maneuver.estimator.height == pytest.approx(height, abs=1e-9)
    applied = controller.last_action[0].numpy()
    np.testing.assert_allclose(applied, maneuver.target - maneuver.nominal, atol=1e-6)
    # Leaving custom control ends the maneuver; nothing is integrated after.
    controller._state = State.RECOVERY
    period()
    assert maneuver.stage == Stage.IDLE


def test_a_fault_requests_the_emergency_stop(handstand):
    module, controller = handstand
    maneuver = module._maneuver(controller)
    q, dq = maneuver.initial_joints, np.zeros(12)
    controller._state = State.CUSTOM_CTRL
    controller.last_lowstate_msg = message(q, dq, 2)
    maneuver.start_now(q, np.array([1.0, 0, 0, 0]), height=0.34)
    module.on_lowstate(controller, message(q, dq, 2), 0.002)
    module.on_lowstate(controller, message(q, dq, 402), 0.402)  # 0.4 s without data
    module.update_custom_buffers(controller, 0.41)
    assert controller.stops == [True]


def test_handstand_refuses_smoothing_or_another_control_rate(handstand):
    module, controller = handstand
    controller.cfg.exp_moving_avg = True
    with pytest.raises(ValueError, match="50 Hz"):
        module._maneuver(controller)


def test_default_custom_implementation_is_still_the_trot_template(monkeypatch):
    monkeypatch.delenv("GO2_CUSTOM_IMPL", raising=False)
    assert get_custom_impl_module().CUSTOM_IMPL_NAME == "go2trot_base.py"
    with pytest.raises(ImportError):
        get_custom_impl_module("missing.py")


# ── Whole maneuver through the deploy stack ──────────────────────────────

SIL = """
import json, sys
from pathlib import Path
from experiments.handstand_robustness.deploy_sil import (
    DeployedEnvironment, make_controller, run_deployed)
from gym.envs.go2.handstand_replay import build_model, load_trajectory
trajectory = load_trajectory(Path(sys.argv[1]), sys.argv[2], build_model())
controller = make_controller(sys.argv[3])
env = DeployedEnvironment(build_model(), trajectory, controller, 301000)
result = run_deployed(env, height="training")
print(json.dumps({k: result[k] for k in (
    "complete_success", "strict_hold_seconds", "observation_error", "target_error")}))
"""


def test_deploy_stack_flies_the_nominal_handstand_in_simulation(tmp_path):
    """Bundle export, MainController's control loop and the 47 channels, end to end."""
    trajectory = os.getenv("GO2_HANDSTAND_TRAJECTORY")
    digest = os.getenv("GO2_HANDSTAND_SHA256")
    checkpoint = os.getenv("GO2_HANDSTAND_CHECKPOINT")
    if not (trajectory and digest and checkpoint):
        pytest.skip("Provide the hash-locked reference and a capture checkpoint")
    env = os.environ | {"GO2_CUSTOM_IMPL": "handstand_capture.py"}
    subprocess.run(
        [
            sys.executable,
            "scripts/export_handstand_bundle.py",
            "--name",
            "bundle",
            "--checkpoint",
            checkpoint,
            "--trajectory",
            trajectory,
            "--out_root",
            str(tmp_path),
        ],
        check=True,
        env=env,
        capture_output=True,
    )
    bundle = tmp_path / "bundle"
    meta = json.loads((bundle / "deploy_meta.json").read_text())
    assert (meta["num_obs"], meta["ctrl_freq"], meta["exp_moving_avg"]) == (
        47,
        50,
        False,
    )
    assert meta["custom_impl"] == "handstand_capture.py"
    done = subprocess.run(
        [sys.executable, "-c", SIL, trajectory, digest, str(bundle)],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["complete_success"] and result["strict_hold_seconds"] >= 5
    # The deploy observation is the training observation, to float32 rounding.
    assert result["observation_error"] < 1e-5
    assert result["target_error"] < 1e-5


# ── Drift correction from planted feet ───────────────────────────────────


def standing_robot(robot):
    """The Go2 standing on flat ground with its feet touching."""
    d = robot.d
    mujoco.mj_resetData(robot.m, d)
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[robot.qids] = np.tile([0.0, 0.7334, -1.4655], 4)
    mujoco.mj_forward(robot.m, d)
    d.qpos[2] += 0.022 - d.geom_xpos[robot.feet, 2].min()
    mujoco.mj_forward(robot.m, d)
    return d


def test_planted_feet_give_the_root_height_and_velocity(robot):
    """A body moving over feet that stay put: the legs reveal its motion."""
    estimator = HandstandEstimator(KinematicModel(robot.tree))
    d = standing_robot(robot)
    q = d.qpos[robot.qids].copy()
    # Joint velocities that keep all four contact points still while the base
    # moves. A contact point is the bottom of the foot sphere and turns with
    # the calf: its velocity is linear @ dq + (angular @ dq) x down.
    base_velocity = np.array([0.05, -0.02, 0.03])
    _, linear, angular = estimator.model.feet_motion(q)
    x, y, z = 0.0, 0.0, -0.022
    cross_down = -np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])  # spin -> spin x down
    contact = np.concatenate([linear[f] + cross_down @ angular[f] for f in range(4)])
    dq = np.linalg.lstsq(contact, np.tile(-base_velocity, 4), rcond=None)[0]
    planted = np.ones(4, dtype=bool)
    velocity, height = estimator.root_from_feet(
        q, dq, np.zeros(3), d.qpos[3:7], planted
    )
    # Mean over the feet, which differ by a fraction of a millimetre.
    assert height == pytest.approx(d.qpos[2], abs=5e-4)
    np.testing.assert_allclose(velocity, base_velocity, atol=1e-9)

    estimator.velocity, estimator.height = np.array([0.5, 0.5, 0.5]), 0.5
    estimator.correct(
        q, dq, np.zeros(3), d.qpos[3:7], planted, height_gain=1.0, velocity_gain=0.0
    )
    assert estimator.height == pytest.approx(height) and estimator.velocity[0] == 0.5
    estimator.correct(
        q, dq, np.zeros(3), d.qpos[3:7], planted, height_gain=0.0, velocity_gain=0.5
    )
    np.testing.assert_allclose(estimator.velocity, 0.5 * (0.5 + velocity), atol=1e-12)
    # No planted foot, no information.
    estimator.correct(
        q, dq, np.zeros(3), d.qpos[3:7], np.zeros(4, dtype=bool), 1.0, 1.0
    )
    np.testing.assert_allclose(estimator.velocity, 0.5 * (0.5 + velocity), atol=1e-12)


def test_drift_correction_is_off_by_default_and_waits_for_settled_contact(robot):
    quat = np.array([1.0, 0, 0, 0])
    d = standing_robot(robot)
    q, load = d.qpos[robot.qids].copy(), np.full(4, 40.0)

    maneuver, a = make_maneuver(robot)
    maneuver.start_now(q, quat, height=0.5)
    for _ in range(10):
        maneuver.correct_drift(q, np.zeros(12), np.zeros(3), quat, load)
    assert maneuver.estimator.height == 0.5  # purely inertial, as trained

    maneuver = Maneuver(
        a | dict(foot_correction_seconds=0.3),
        HandstandEstimator(KinematicModel(robot.tree)),
    )
    maneuver.start_now(q, quat, height=0.5)
    maneuver.estimator.velocity = np.array([0.2, 0.0, 0.0])
    maneuver.correct_drift(q, np.zeros(12), np.zeros(3), quat, load)
    maneuver.correct_drift(q, np.zeros(12), np.zeros(3), quat, load)
    assert maneuver.estimator.height == 0.5  # two periods of contact: not yet
    for _ in range(100):
        maneuver.correct_drift(q, np.zeros(12), np.zeros(3), quat, load)
    assert maneuver.estimator.height == pytest.approx(d.qpos[2], abs=5e-4)
    assert maneuver.estimator.velocity[0] == 0.2  # velocity only once held
    maneuver.step_count = maneuver.cfg.foot_velocity_from_step
    for _ in range(100):
        maneuver.correct_drift(q, np.zeros(12), np.zeros(3), quat, load)
    assert abs(maneuver.estimator.velocity[0]) < 1e-3
    # Unloaded feet (flight) leave the estimate alone.
    maneuver.estimator.height = 0.5
    maneuver.correct_drift(q, np.zeros(12), np.zeros(3), quat, np.zeros(4))
    maneuver.correct_drift(q, np.zeros(12), np.zeros(3), quat, load)
    assert maneuver.estimator.height == 0.5
