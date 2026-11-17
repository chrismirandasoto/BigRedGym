"""Sensor and actuator errors for the capture task; simulation only.

A *range* dictionary says how large each error may be; `draw` turns it into
one concrete *condition* for an episode, which is what the environment takes
and what result files record. Unlisted errors are zero, so an empty range
reproduces the nominal environment exactly.

Sensor errors (what the policy and the estimator are told):
    gyro_bias               rad/s    constant offset per axis
    gyro_noise              rad/s    white noise per reading
    accel_bias              m/s^2    constant offset per axis
    accel_noise             m/s^2    white noise per 2 ms sample
    orientation_bias        rad      constant attitude error (roll, pitch, yaw)
    encoder_noise           rad      white noise per joint reading
    encoder_velocity_noise  rad/s    white noise per joint-velocity reading
    height_error            m        error in the calibrated standing height

Actuator errors (what the motors and the world do):
    zero_offset             rad      joint zero error; shifts the motor's own
                                     position loop and the encoder reading
    kp_scale, kd_scale      factor   joint PD gains
    strength                factor   available torque
    command_delay_substeps  2 ms     delay before a new target reaches the motor
    joint_damping           N m s/rad  extra viscous friction in every joint
    friction                factor   ground friction
    mass                    factor   every link's mass and inertia

The magnitudes in PROFILES are engineering guesses for a Go2, not
measurements of one.
"""

import numpy as np

from experiments.handstand_robustness.capture import (
    CaptureTrainingEnvironment,
    InertialObserver,
)
from experiments.handstand_robustness.core import Environment

# Symmetric errors: drawn uniformly in [-value, +value].
SYMMETRIC = {
    "gyro_bias": 3,
    "accel_bias": 3,
    "orientation_bias": 3,
    "height_error": 1,
    "zero_offset": 12,
}
# Noise levels and one-sided quantities: drawn uniformly in [0, value].
ONE_SIDED = (
    "gyro_noise",
    "accel_noise",
    "encoder_noise",
    "encoder_velocity_noise",
    "joint_damping",
)
# Multipliers: drawn uniformly in [1 - value, 1 + value], or in [low, high].
SWEEPABLE = tuple(SYMMETRIC) + ONE_SIDED
FACTORS = ("kp_scale", "kd_scale", "strength", "friction", "mass")
RESET_KEYS = ("joint_position", "joint_velocity")

PROFILES = {
    # The start-state scatter the released 67/100 evaluation used.
    "release": dict(joint_position=0.003, joint_velocity=0.03),
    "sensor": dict(
        joint_position=0.003,
        joint_velocity=0.03,
        gyro_bias=0.005,
        gyro_noise=0.005,
        accel_bias=0.03,
        accel_noise=0.05,
        orientation_bias=0.005,
        encoder_noise=0.0005,
        encoder_velocity_noise=0.05,
        height_error=0.003,
    ),
    "actuator": dict(
        joint_position=0.003,
        joint_velocity=0.03,
        zero_offset=0.003,
        kp_scale=0.05,
        kd_scale=0.05,
        strength=[0.95, 1.0],
        command_delay_substeps=[0, 2],
        joint_damping=0.02,
        friction=0.1,
    ),
}
PROFILES["both"] = PROFILES["sensor"] | PROFILES["actuator"]
# What a policy meant for the robot should tolerate at scale 1. The robot
# calibrates constant gyro and accelerometer bias while standing, so those
# matter less than their size suggests; attitude error is not calibrated.
PROFILES["deploy"] = dict(
    gyro_bias=0.01,
    gyro_noise=0.005,
    accel_bias=0.1,
    accel_noise=0.05,
    orientation_bias=0.01,
    encoder_noise=0.0005,
    encoder_velocity_noise=0.05,
    zero_offset=0.005,
    kp_scale=0.1,
    kd_scale=0.1,
    strength=[0.9, 1.0],
    command_delay_substeps=[0, 3],
    joint_damping=0.02,
    friction=0.2,
)


def scaled(ranges, factor):
    """The same ranges with every error `factor` times as large."""
    out = {}
    for key, value in ranges.items():
        if key in RESET_KEYS:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            centre = 1.0 if key in FACTORS else 0.0
            out[key] = [centre + factor * (v - centre) for v in value]
        else:
            out[key] = factor * value
    return out


def draw(ranges, seed):
    """One concrete condition from `ranges`; the same seed gives the same one."""
    known = set(SYMMETRIC) | set(ONE_SIDED) | set(FACTORS) | set(RESET_KEYS)
    unknown = set(ranges) - known - {"command_delay_substeps"}
    if unknown:
        raise KeyError(f"unknown randomization ranges: {sorted(unknown)}")
    # Its own stream, so enabling an error never changes the start state that
    # the environment draws from the episode seed.
    rng = np.random.default_rng([int(seed), 7])
    condition = {key: ranges[key] for key in RESET_KEYS if key in ranges}
    for key in sorted(ranges):
        value = ranges[key]
        if key in SYMMETRIC:
            condition[key] = (value * rng.uniform(-1, 1, SYMMETRIC[key])).tolist()
        elif key in ONE_SIDED:
            condition[key] = float(value * rng.uniform(0, 1))
        elif key in FACTORS:
            low, high = (
                value if isinstance(value, (list, tuple)) else (1 - value, 1 + value)
            )
            condition[key] = float(rng.uniform(low, high))
        elif key == "command_delay_substeps":
            low, high = value if isinstance(value, (list, tuple)) else (0, value)
            # Whole physics steps. A fractional range is rounded at random, so
            # scaling the range down makes a delay rarer instead of removing it.
            real = rng.uniform(low, high)
            condition[key] = int(
                np.floor(real) + (rng.uniform() < real - np.floor(real))
            )
    return condition


def _rotation_vector(vector):
    """Rotation matrix of a rotation vector (Rodrigues)."""
    angle = np.linalg.norm(vector)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = np.asarray(vector) / angle
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class SensorErrors:
    """Turns true quantities into what the robot's sensors would report."""

    def __init__(self, condition, seed):
        get = condition.get
        self.gyro_bias = np.asarray(get("gyro_bias", np.zeros(3)), dtype=float)
        self.accel_bias = np.asarray(get("accel_bias", np.zeros(3)), dtype=float)
        self.tilt = _rotation_vector(get("orientation_bias", np.zeros(3)))
        self.zero_offset = np.asarray(get("zero_offset", np.zeros(12)), dtype=float)
        self.height_error = float(np.ravel(get("height_error", 0.0))[0])
        self.gyro_noise = float(get("gyro_noise", 0.0))
        self.accel_noise = float(get("accel_noise", 0.0))
        self.encoder_noise = float(get("encoder_noise", 0.0))
        self.velocity_noise = float(get("encoder_velocity_noise", 0.0))
        self.rng = np.random.default_rng([int(seed), 11])
        self.active = bool(
            self.gyro_bias.any()
            or self.accel_bias.any()
            or not np.allclose(self.tilt, np.eye(3))
            or self.zero_offset.any()
            or self.height_error
            or self.gyro_noise
            or self.accel_noise
            or self.encoder_noise
            or self.velocity_noise
        )

    def _noise(self, level, size):
        return level * self.rng.normal(size=size) if level else 0.0

    def rotation(self, rotation):
        return rotation @ self.tilt

    def gyro(self, rate):
        return rate + self.gyro_bias + self._noise(self.gyro_noise, 3)

    def accel(self, force):
        return force + self.accel_bias + self._noise(self.accel_noise, 3)

    def joints(self, q, dq):
        return (
            q + self.zero_offset + self._noise(self.encoder_noise, 12),
            dq + self._noise(self.velocity_noise, 12),
        )


def _quaternion(rotation):
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    import mujoco

    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(rotation, dtype=float).ravel())
    return quat


class _MeasuredObserver(InertialObserver):
    """The training observer fed the measured force and attitude."""

    def __init__(self, initial_height, errors):
        super().__init__(initial_height + errors.height_error)
        self.errors = errors

    def update(self, specific_force_body, rotation):
        super().update(
            self.errors.accel(np.asarray(specific_force_body)),
            self.errors.rotation(rotation),
        )


class RandomizedEnvironment(CaptureTrainingEnvironment):
    """Capture task under one drawn condition of sensor and actuator errors."""

    def __init__(self, model, trajectory, seed=0, condition=None, horizon=110):
        super().__init__(model, trajectory, seed, condition, horizon)
        self.errors = SensorErrors(self.condition, seed)
        self.joint_zero_offset = self.errors.zero_offset
        damping = float(self.condition.get("joint_damping", 0.0))
        if damping:
            self.m.dof_damping[self.vids] += damping
        nominal_height = self.capture_kinematics.observer.height
        observer = _MeasuredObserver(nominal_height, self.errors)
        self.capture_kinematics.observer = observer
        self.imu_observer = observer

    def observation(self):
        if not hasattr(self, "errors") or not self.errors.active:
            return super().observation()
        obs = Environment.observation(self).astype(np.float64)
        rotation = self.errors.rotation(self.d.xmat[self.base].reshape(3, 3))
        q, dq = self.errors.joints(self.d.qpos[self.qids], self.d.qvel[self.vids])
        obs[:3] = rotation.T @ [0, 0, -1]
        obs[3:6] = self.errors.gyro(self.d.qvel[3:6])
        obs[6:18] = q - self.trajectory["nominal"]
        obs[18:30] = dq / self.speed
        obs = obs.astype(np.float32)
        estimate = self.capture_kinematics.features(obs, _quaternion(rotation))
        return np.r_[obs, estimate[55:59]].astype(np.float32)
