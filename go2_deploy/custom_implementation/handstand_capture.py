"""Go2 handstand (capture_residual_v1) for the go2_deploy framework.

Select with GO2_CUSTOM_IMPL=handstand_capture.py, or deploy a bundle from
scripts/export_handstand_bundle.py (the bundle names this file).

The policy takes 47 numbers, in this order:

    projected_gravity  3   gravity direction in the body frame
    hs_gyro            3   body angular velocity, rad/s, gyro bias removed
    dof_pos_obs       12   joint angle minus the nominal posture, rad
    dof_vel           12   joint velocity / joint speed limit
    dof_pos_target    12   (previous applied target - nominal) / 2
    hs_elapsed         1   commands issued since the trigger * 0.02 / 8
    hs_com             4   estimated COM height / 0.6, COM velocity xyz

Policies trained with history (capture_history_v1) take one more piece, which
their bundle adds to the observation vector:

    hs_history    42 * K   the first 42 values above as they were 1..K control
                           periods ago, most recent first; zero before then

Sequence on the robot: A enters custom control and the robot moves to the
training start posture. When the terminal shows "ready", press Y (or type h)
to start the handstand. X or Enter is the emergency stop at any time.

Simulation-validated only. See docs/go2_handstand_deploy.md before use.
"""

import json
import os
import threading

import numpy as np
import torch

from go2_deploy.handstand.assets import load_assets
from go2_deploy.handstand.estimator import HandstandEstimator
from go2_deploy.handstand.kinematics import KinematicModel
from go2_deploy.handstand.maneuver import Maneuver, ManeuverConfig, Stage
from go2_deploy.state import State

# Unitree reports FR, FL, RR, RL; the policy uses FL, FR, RL, RR. Same table
# as deploy_utility.UNITREE_TO_QGYM_JOINT_IDX, which cannot be imported here
# because deploy_utility imports this module's config.
UNITREE_TO_POLICY = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

_REFERENCE = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "handstand_reference.json"
)
with open(_REFERENCE) as _f:
    _reference = json.load(_f)


class CustomDeployConfig:
    task_name = "go2handstand_capture"

    # Training ran position targets at 50 Hz into a 60 / 3 joint PD loop.
    ctrl_freq = 50
    kp = 60.0
    kd = 3.0
    # The policy's previous-target input must be the target actually sent,
    # so no smoothing between the policy and the motors.
    exp_moving_avg = False

    # Joint observations and actions are relative to this posture.
    default_dof_pos = _reference["nominal"]

    obs_vector = [
        "projected_gravity",
        "hs_gyro",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "hs_elapsed",
        "hs_com",
    ]

    custom_obs_sizes = {
        "hs_gyro": 3,
        "hs_elapsed": 1,
        "hs_com": 4,
        "hs_history": 4 * 42,  # only in bundles of history policies
    }

    # Observations are divided by these; unlisted pieces are not scaled.
    # dof_pos_target is also the radians per unit of policy output.
    class DeployScaling:
        dof_vel = _reference["speed_rad_s"]
        dof_pos_target = 2.0


def get_custom_buffers_dict():
    return {
        "handstand": None,  # Maneuver, built on first use from the bundle
        "handstand_lock": threading.Lock(),
        "handstand_last_tick": None,
        "handstand_last_time": None,
        "handstand_last_status": None,
        "handstand_y": 0,
        "handstand_history": None,  # tensor (K, 42), most recent first
    }


def _maneuver(main_controller):
    """The Maneuver for this controller, built once from the bundle's assets."""
    if main_controller.handstand is None:
        cfg = main_controller.cfg
        if cfg.exp_moving_avg or cfg.ctrl_freq != CustomDeployConfig.ctrl_freq:
            raise ValueError("handstand needs 50 Hz control without action smoothing")
        assets = load_assets(cfg)
        estimator = HandstandEstimator(
            KinematicModel(assets["kinematics"]),
            imu_offset=assets["imu_offset"],
            height_offset=assets["height_offset"],
        )
        main_controller.handstand = Maneuver(assets, estimator, ManeuverConfig())
    return main_controller.handstand


def _joints(lowstate_msg):
    """Joint angles and velocities in policy order, float64."""
    order = UNITREE_TO_POLICY
    states = lowstate_msg.motor_state
    q = np.array([states[i].q for i in order], dtype=float)
    dq = np.array([states[i].dq for i in order], dtype=float)
    return q, dq


def _foot_force(lowstate_msg):
    """Foot loads in newtons, FL, FR, RL, RR; None if the message has none."""
    force = getattr(lowstate_msg, "foot_force", None)
    if force is None:
        return None
    # Unitree reports FR, FL, RR, RL.
    return np.array([force[1], force[0], force[3], force[2]], dtype=float)


def _imu(lowstate_msg):
    imu = lowstate_msg.imu_state
    return (
        np.array(imu.accelerometer, dtype=float),
        np.array(imu.gyroscope, dtype=float),
        np.array(imu.quaternion, dtype=float),  # w, x, y, z
    )


# Called for every LowState_ message (500 Hz): integrate the IMU, read the trigger
def on_lowstate(main_controller, lowstate_msg, t):
    with main_controller.handstand_lock:
        maneuver = _maneuver(main_controller)
        # Prefer the robot's own millisecond tick over arrival time.
        tick = getattr(lowstate_msg, "tick", None)
        last_tick = main_controller.handstand_last_tick
        last_time = main_controller.handstand_last_time
        if tick is not None and last_tick is not None and tick > last_tick:
            dt = (tick - last_tick) * 1e-3
        elif last_time is not None:
            dt = t - last_time
        else:
            dt = maneuver.cfg.imu_dt
        main_controller.handstand_last_tick = tick
        main_controller.handstand_last_time = t
        if dt <= 0:
            return  # duplicate message
        accel, gyro, quat = _imu(lowstate_msg)
        maneuver.on_imu(accel, gyro, quat, dt)
        # Y on the remote starts the handstand (rising edge).
        pressed = main_controller.remote_controller.Y[0]
        if pressed and not main_controller.handstand_y:
            _trigger(main_controller, maneuver)
        main_controller.handstand_y = pressed


# Optional keyboard hook: h starts the handstand
def on_key(main_controller, key):
    if key != "h":
        return False
    with main_controller.handstand_lock:
        _trigger(main_controller, _maneuver(main_controller))
    return True


def _trigger(main_controller, maneuver):
    if maneuver.trigger():
        print("Handstand triggered")
    elif main_controller._state == State.CUSTOM_CTRL:
        print(f"Handstand not started: {maneuver.status}")


# Called at the control frequency, before the observation is built
def update_custom_buffers(main_controller, t):
    with main_controller.handstand_lock:
        maneuver = _maneuver(main_controller)
        active = main_controller._state == State.CUSTOM_CTRL
        if active and maneuver.stage == Stage.IDLE:
            q, _ = _joints(main_controller.last_lowstate_msg)
            maneuver.activate(q)
            print("Handstand: moving to the start posture")
        elif not active and maneuver.stage != Stage.IDLE:
            maneuver.deactivate()
        if maneuver.fault and active:
            print(f"Handstand fault: {maneuver.fault}")
            maneuver.fault = None
            main_controller.request_emergency_stop()
        if active and main_controller.last_lowstate_msg is not None:
            msg = main_controller.last_lowstate_msg
            q, dq = _joints(msg)
            _, gyro, quat = _imu(msg)
            maneuver.correct_drift(q, dq, gyro, quat, _foot_force(msg))
        status = maneuver.status.split(",")[0]
        if active and status != main_controller.handstand_last_status:
            print(f"Handstand: {maneuver.status}")
            main_controller.handstand_last_status = status


# Called in custom control with the policy's offset (radians from nominal);
# returns the offset actually applied, which the framework stores and sends
def process_action(main_controller, action):
    with main_controller.handstand_lock:
        maneuver = _maneuver(main_controller)
        msg = main_controller.last_lowstate_msg
        q, dq = _joints(msg)
        _, gyro, quat = _imu(msg)
        was_running = maneuver.stage == Stage.RUN
        offset = maneuver.step(action.numpy(), q, dq, gyro, quat)
        history = _history(main_controller)
        if history is not None:
            if was_running:
                # The observation this action came from becomes the newest frame.
                history[1:] = history[:-1].clone()
                history[0] = main_controller.last_obs[:HISTORY_FRAME]
            else:
                history.zero_()
    return torch.tensor(offset, dtype=torch.float32)


HISTORY_FRAME = 42


def _history(main_controller):
    """Past observation frames, if this bundle's policy uses them."""
    cfg = main_controller.cfg
    if "hs_history" not in cfg.obs_vector:
        return None
    if main_controller.handstand_history is None:
        frames = cfg.obs_sizes["hs_history"] // HISTORY_FRAME
        if (
            cfg.obs_vector[-1] != "hs_history"
            or frames * HISTORY_FRAME != cfg.obs_sizes["hs_history"]
        ):
            raise ValueError("hs_history must be last and a multiple of 42 values")
        main_controller.handstand_history = torch.zeros(frames, HISTORY_FRAME)
    return main_controller.handstand_history


def get_custom_obs_pieces():
    return {
        "hs_gyro": _get_obs_hs_gyro,
        "hs_elapsed": _get_obs_hs_elapsed,
        "hs_com": _get_obs_hs_com,
        "hs_history": _get_obs_hs_history,
    }


def _get_obs_hs_gyro(main_controller, lowstate_msg):
    with main_controller.handstand_lock:
        estimator = _maneuver(main_controller).estimator
        return torch.tensor(estimator.rate(_imu(lowstate_msg)[1]), dtype=torch.float32)


def _get_obs_hs_elapsed(main_controller, lowstate_msg):
    with main_controller.handstand_lock:
        return torch.tensor([_maneuver(main_controller).elapsed()], dtype=torch.float32)


def _get_obs_hs_com(main_controller, lowstate_msg):
    with main_controller.handstand_lock:
        estimator = _maneuver(main_controller).estimator
        q, dq = _joints(lowstate_msg)
        _, gyro, quat = _imu(lowstate_msg)
        return torch.tensor(estimator.com(q, dq, gyro, quat), dtype=torch.float32)


def _get_obs_hs_history(main_controller, lowstate_msg):
    with main_controller.handstand_lock:
        return _history(main_controller).flatten().clone()
