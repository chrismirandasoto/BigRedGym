"""Phase and bounded joint targets for the handstand; NumPy only.

The capture policy is indexed by how many 20 ms commands have been issued
since the maneuver began, and it was trained behind a target filter:

    desired = clip(nominal + 2 * action, joint limits)
    target += clip(desired - target, +/- 0.7 * speed limit * 20 ms)

Both live here, so the robot sends exactly the targets training applied and
the policy's "previous target" input is the filtered target, not its request.

Stages after the operator enters custom control:

    PREPARE  move slowly to the training start posture and stand still while
             the IMU biases are measured
    READY    still standing; waiting for the operator's trigger
    RUN      the policy drives the filtered target; phase counts commands

Training begins with the joint target equal to the start posture, which
leaves the motors unloaded: the robot cannot wait there. While standing, the
target is instead `standing_target`, which holds the start posture against
gravity; at the trigger it switches to the training target.
"""

from enum import Enum

import numpy as np

from go2_deploy.handstand.kinematics import quat_to_matrix


class Stage(Enum):
    IDLE = 0
    PREPARE = 1
    READY = 2
    RUN = 3


class ManeuverConfig:
    control_dt = 0.02
    imu_dt = 0.002
    action_scale = 2.0  # rad per policy unit
    slew_fraction = 0.7  # of the joint speed limit
    phase_period_s = 8.0  # the policy's time input is elapsed / 8
    prepare_speed = 0.5  # rad/s toward the start posture
    prepare_seconds = 5.0  # least time standing before READY
    calibration_seconds = 2.0  # IMU window averaged for the biases
    posture_tolerance = 0.05  # rad between joints and the training start
    still_joint_speed = 0.2  # rad/s
    still_gyro = 0.1  # rad/s
    still_gyro_spread = 0.02  # rad/s, standard deviation over the window
    # m/s^2 between the two halves of the window. In simulation a bias error
    # of 0.012 m/s^2 from calibrating while still settling lost the handstand.
    still_accel_drift = 0.005
    level_tolerance = 0.09  # rad of tilt, about 5 degrees
    stale_seconds = 0.05  # longest IMU gap tolerated while RUN
    # Drift correction from planted feet; None leaves the estimator purely
    # inertial, as in training. Seconds for the estimate to close ~63% of its
    # gap to what the planted feet imply.
    foot_correction_seconds = None
    foot_force_threshold = 25.0  # N on a foot before it counts as planted
    foot_planted_steps = 3  # consecutive control periods above the threshold
    # Feet give a usable velocity only once the handstand is held: from the
    # step at which the policy hands over to its balance feedback (2.2 s).
    foot_velocity_from_step = 110


class Maneuver:
    def __init__(self, assets, estimator, config=None):
        self.cfg = config or ManeuverConfig()
        self.estimator = estimator
        self.nominal = np.asarray(assets["nominal"], dtype=float)
        self.initial_target = np.asarray(assets["initial_target"], dtype=float)
        self.initial_joints = np.asarray(assets["initial_joints"], dtype=float)
        self.standing_target = np.asarray(assets["standing_target"], dtype=float)
        self.lower, self.upper = np.asarray(assets["joint_bounds"], dtype=float).T
        self.speed = np.asarray(assets["speed"], dtype=float)
        self.stage = Stage.IDLE
        self.step_count = 0
        self.target = self.nominal.copy()
        self.triggered = False
        self.fault = None
        self.status = "idle"
        self._prepare_time = 0.0
        self._imu_gap = 0.0
        self._planted_steps = np.zeros(4, dtype=int)
        correction = assets.get("foot_correction_seconds")
        if correction is not None:
            self.cfg.foot_correction_seconds = float(correction)

    # ── Operator events ──────────────────────────────────────────────────

    def activate(self, q):
        """Custom control entered: hold where the joints are, then prepare."""
        self.stage = Stage.PREPARE
        self.target = np.asarray(q, dtype=float).copy()
        self.step_count = 0
        self.triggered = False
        self.fault = None
        self._prepare_time = 0.0
        self._imu_gap = 0.0
        self.estimator.clear_calibration(
            round(self.cfg.calibration_seconds / self.cfg.imu_dt)
        )

    def deactivate(self):
        self.stage = Stage.IDLE
        self.triggered = False

    def trigger(self):
        """Operator asks for the handstand; honoured only when READY."""
        if self.stage == Stage.READY:
            self.triggered = True
            return True
        return False

    def start_now(self, q, quat, height=None, use_biases=False):
        """Skip PREPARE: begin the maneuver from the current standing state."""
        self.stage = Stage.RUN
        self.step_count = 0
        self.target = self.initial_target.copy()
        self.fault = None
        self._imu_gap = 0.0
        self.estimator.start(q, quat, use_biases=use_biases, height=height)

    # ── IMU rate ─────────────────────────────────────────────────────────

    def on_imu(self, accel, gyro, quat, dt):
        values = np.r_[accel, gyro, quat]
        if not np.isfinite(values).all():
            self.fault = "non-finite IMU sample"
            return
        if self.stage == Stage.RUN:
            if dt > self.cfg.stale_seconds:
                self.fault = f"IMU gap of {dt:.3f} s during the maneuver"
                return
            self._imu_gap = 0.0
            self.estimator.update(accel, gyro, quat, dt)
        elif self.stage in (Stage.PREPARE, Stage.READY):
            self.estimator.add_calibration_sample(accel, gyro, quat)

    # ── Control rate ─────────────────────────────────────────────────────

    def elapsed(self):
        """The policy's time input: commands issued, as seconds / 8."""
        return self.step_count * self.cfg.control_dt / self.cfg.phase_period_s

    def _standing_still(self, q, dq, gyro, quat):
        est = self.estimator
        if self._prepare_time < self.cfg.prepare_seconds:
            return "settling"
        if abs(q - self.initial_joints).max() > self.cfg.posture_tolerance:
            return "posture differs from the training start"
        if abs(dq).max() > self.cfg.still_joint_speed:
            return "joints moving"
        if not est.calibration_ready():
            return "collecting IMU calibration"
        if np.linalg.norm(gyro) > self.cfg.still_gyro:
            return "body rotating"
        if est.calibration_motion()[0] > self.cfg.still_gyro_spread:
            return "body not still"
        if est.calibration_drift() > self.cfg.still_accel_drift:
            return "accelerometer average still drifting"
        tilt = np.arccos(np.clip(quat_to_matrix(quat)[2, 2], -1, 1))
        if tilt > self.cfg.level_tolerance:
            return "body not level"
        return None

    def correct_drift(self, q, dq, gyro, quat, foot_force):
        """Before the observation is built: use planted feet, if enabled.

        `foot_force` is the vertical load on FL, FR, RL, RR in newtons.
        """
        tau = self.cfg.foot_correction_seconds
        if tau is None or foot_force is None:
            return
        loaded = np.asarray(foot_force, dtype=float) > self.cfg.foot_force_threshold
        # A foot that has only just touched down may still be sliding.
        self._planted_steps = np.where(loaded, self._planted_steps + 1, 0)
        if self.stage == Stage.RUN:
            planted = self._planted_steps >= self.cfg.foot_planted_steps
            gain = 1.0 - np.exp(-self.cfg.control_dt / tau)
            held = self.step_count >= self.cfg.foot_velocity_from_step
            self.estimator.correct(
                q,
                dq,
                gyro,
                quat,
                planted,
                height_gain=gain,
                velocity_gain=gain if held else 0.0,
            )

    def step(self, offset, q, dq, gyro, quat):
        """One control period: return the applied target minus nominal."""
        q, dq = np.asarray(q, dtype=float), np.asarray(dq, dtype=float)
        dt = self.cfg.control_dt
        if self.stage == Stage.RUN:
            self._imu_gap += dt
            if self._imu_gap > self.cfg.stale_seconds + dt:
                self.fault = "no IMU samples during the maneuver"
            action = np.clip(
                np.asarray(offset, dtype=float) / self.cfg.action_scale, -1, 1
            )
            desired = np.clip(
                self.nominal + self.cfg.action_scale * action, self.lower, self.upper
            )
            limit = self.cfg.slew_fraction * self.speed * dt
            self.target = self.target + np.clip(desired - self.target, -limit, limit)
            self.step_count += 1
            self.status = f"running, step {self.step_count}"
        elif self.stage in (Stage.PREPARE, Stage.READY):
            limit = self.cfg.prepare_speed * dt
            self.target = self.target + np.clip(
                self.standing_target - self.target, -limit, limit
            )
            self._prepare_time += dt
            reached = abs(self.target - self.standing_target).max() < 1e-9
            waiting = self._standing_still(q, dq, gyro, quat) if reached else "moving"
            self.stage = Stage.PREPARE if waiting else Stage.READY
            self.status = waiting or "ready: trigger the handstand"
            if self.stage == Stage.READY and self.triggered:
                self.triggered = False
                self.stage = Stage.RUN
                self.step_count = 0
                self._imu_gap = 0.0
                # From here the target is the training one; see module docstring.
                self.target = self.initial_target.copy()
                self.estimator.start(q, quat, use_biases=True)
                self.status = "running, step 0"
        return self.target - self.nominal
