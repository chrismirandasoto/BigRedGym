"""Centre-of-mass height and velocity from the IMU and joint encoders only.

Matches the estimator the capture policy was trained with
(experiments/handstand_robustness/capture.py, InertialBalance):

    root velocity   integral of  R f + g   at the IMU rate, from rest
    root height     integral of the vertical root velocity, from standing
    COM             root estimate + forward kinematics of the joints

where f is the specific force at the base origin in the body frame and R the
body-to-world rotation. Left alone, the integrals turn accelerometer bias and
attitude error into velocity and height error that grows for as long as the
maneuver runs. `correct` is the optional remedy: while feet are planted (foot
force sensors), leg kinematics give the root velocity and height directly,
and the integrated estimate is pulled toward them. See
docs/go2_handstand_deploy.md for the measured effect of both.
"""

from collections import deque

import numpy as np

from go2_deploy.handstand.kinematics import quat_to_matrix

GRAVITY = 9.81
HEIGHT_SCALE = 0.6  # the policy sees COM height / 0.6
FOOT_RADIUS = 0.022


class HandstandEstimator:
    def __init__(self, model, imu_offset=(0.0, 0.0, 0.0), height_offset=0.0):
        self.model = model
        # IMU position in the base frame; zero means the sensor already
        # reports the specific force of the base origin.
        self.imu_offset = np.asarray(imu_offset, dtype=float)
        # Added to the root height the feet imply (e.g. foot compression).
        self.height_offset = float(height_offset)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.velocity = np.zeros(3)
        self.height = 0.0
        self._previous_rate = None
        self._window = deque()

    # ── Stationary calibration ───────────────────────────────────────────

    def clear_calibration(self, samples):
        """Start collecting the most recent `samples` stationary IMU readings."""
        self._window = deque(maxlen=int(samples))

    def add_calibration_sample(self, accel, gyro, quat):
        rotation = quat_to_matrix(quat)
        # What a perfect accelerometer reads at rest in this orientation.
        expected = rotation.T @ [0.0, 0.0, GRAVITY]
        self._window.append(np.r_[accel, gyro, expected])

    def calibration_ready(self):
        return len(self._window) == self._window.maxlen

    def calibration_motion(self):
        """Largest gyro and accelerometer spread over the window."""
        rows = np.array(self._window)
        return float(rows[:, 3:6].std(axis=0).max()), float(
            rows[:, :3].std(axis=0).max()
        )

    def calibration_drift(self):
        """Change in the apparent accelerometer bias across the window, m/s^2.

        A body that is still settling shows up as a bias that moves; using it
        would put that motion into every later velocity estimate.
        """
        rows = np.array(self._window)
        residual = rows[:, :3] - rows[:, 6:9]
        half = len(rows) // 2
        return float(np.linalg.norm(residual[:half].mean(0) - residual[half:].mean(0)))

    def standing_height(self, q, quat):
        """Root height with the lowest foot resting on flat ground."""
        feet = self.model.feet(q) @ quat_to_matrix(quat).T
        return float(FOOT_RADIUS - feet[:, 2].min() + self.height_offset)

    def start(self, q, quat, use_biases=True, height=None):
        """Begin integrating from rest; the robot must be standing still."""
        if use_biases and len(self._window):
            rows = np.array(self._window).mean(axis=0)
            self.gyro_bias = rows[3:6].copy()
            self.accel_bias = rows[:3] - rows[6:9]
        else:
            self.gyro_bias = np.zeros(3)
            self.accel_bias = np.zeros(3)
        self.velocity = np.zeros(3)
        self.height = self.standing_height(q, quat) if height is None else float(height)
        self._previous_rate = None

    # ── IMU-rate integration ─────────────────────────────────────────────

    def update(self, accel, gyro, quat, dt):
        """Consume one IMU sample taken `dt` seconds after the previous one."""
        rate = np.asarray(gyro, dtype=float) - self.gyro_bias
        force = np.asarray(accel, dtype=float) - self.accel_bias
        if self.imu_offset.any():
            # Move the measurement from the IMU to the base origin:
            # a_origin = a_imu - alpha x r - w x (w x r).
            alpha = (
                (rate - self._previous_rate) / dt
                if self._previous_rate is not None and dt > 0
                else np.zeros(3)
            )
            r = self.imu_offset
            force = force - np.cross(alpha, r) - np.cross(rate, np.cross(rate, r))
        self._previous_rate = rate
        acceleration = quat_to_matrix(quat) @ force + [0.0, 0.0, -GRAVITY]
        self.velocity = self.velocity + dt * acceleration
        self.height = self.height + dt * self.velocity[2]

    # ── Drift correction from planted feet ───────────────────────────────

    def root_from_feet(self, q, dq, gyro, quat, planted):
        """Root velocity and height implied by feet that rest on the ground.

        A planted foot's contact point does not move, so the base moves
        opposite to how the leg and the body rotation carry that point. The
        foot is a sphere: its contact point is the lowest point of the sphere
        and travels with the calf, which is what lets the foot roll.
        """
        rotation = quat_to_matrix(quat)
        rate = self.rate(gyro)
        dq = np.asarray(dq, dtype=float)
        centres, linear, angular = self.model.feet_motion(q)
        down = rotation.T @ [0.0, 0.0, -FOOT_RADIUS]  # centre to contact, base frame
        velocities, heights = [], []
        for foot in np.flatnonzero(planted):
            contact = centres[foot] + down
            spin = angular[foot] @ dq
            moving = np.cross(rate, contact) + linear[foot] @ dq + np.cross(spin, down)
            velocities.append(-rotation @ moving)
            heights.append(
                FOOT_RADIUS - (rotation @ centres[foot])[2] + self.height_offset
            )
        return np.mean(velocities, axis=0), float(np.mean(heights))

    def correct(self, q, dq, gyro, quat, planted, height_gain, velocity_gain):
        """Move the integrated root estimate part of the way toward the feet's.

        Gains are fractions per call. Along the nominal handstand the feet
        give the height to about 1 mm at every stage, but the velocity only
        once the handstand is held (2 mm/s); during the kick the feet slide
        and land, and their velocity is wrong by up to 0.9 m/s.
        """
        if not np.any(planted):
            return
        velocity, height = self.root_from_feet(q, dq, gyro, quat, planted)
        self.velocity = self.velocity + velocity_gain * (velocity - self.velocity)
        self.height = self.height + height_gain * (height - self.height)

    # ── Policy inputs ────────────────────────────────────────────────────

    def rate(self, gyro):
        """Body angular velocity with the calibrated bias removed."""
        return np.asarray(gyro, dtype=float) - self.gyro_bias

    def com(self, q, dq, gyro, quat):
        """[COM height / 0.6, COM velocity xyz in the world frame]."""
        rotation = quat_to_matrix(quat)
        centre, jacobian = self.model.com(q)
        relative = np.cross(self.rate(gyro), centre) + jacobian @ np.asarray(dq, float)
        height = self.height + (rotation @ centre)[2]
        velocity = self.velocity + rotation @ relative
        return np.r_[height / HEIGHT_SCALE, velocity]
