"""Continuous policy evaluation using the unchanged Assignment 6 strict gate.

The actor contract excludes simulator COM, contact forces and future states.
Reference states are used only for imitation reward and offline demonstrations.
"""

import hashlib
from collections import deque

import mujoco
import numpy as np
import torch
from torch import nn

from gym.envs.go2.handstand_replay import (
    ACTION_SCALE,
    CONTROL_DT,
    KD,
    KP,
    PHYSICS_DT,
    STATE_SPEC,
    Replay,
)


class Policy(nn.Module):
    observation_dim = 43  # gravity3, body gyro3, q12, dq12, previous target12, phase1

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.zeros(self.observation_dim))
        self.register_buffer("std", torch.ones(self.observation_dim))
        self.actor = nn.Sequential(
            nn.Linear(43, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 12),
        )
        self.critic = nn.Sequential(nn.Linear(43, 128), nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((12,), -3.0))

    def features(self, obs):
        return ((obs - self.mean) / self.std).clamp(-10, 10)

    def forward(self, obs):
        return self.actor(self.features(obs)).tanh()

    def distribution(self, obs):
        return torch.distributions.Normal(
            self.actor(self.features(obs)), self.log_std.exp()
        )

    def value(self, obs):
        return self.critic(self.features(obs)).squeeze(-1)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Environment(Replay):
    """No mid-episode resets. Physics scoring runs at 500 Hz; policy at 50 Hz."""

    def __init__(self, model, trajectory, seed=0, condition=None):
        super().__init__(model, trajectory)
        self.rng = np.random.default_rng(seed)
        self.condition = condition or {}
        self.index = 0
        self.takeoff_time = self.capture_time = self.hold_start_time = None
        self.samples = []
        self.best_samples = []
        self.current_samples = []
        self.previous_action = np.zeros(12)
        self.action_variation = 0.0
        self.latency = int(self.condition.get("latency", 0))
        self.queue = deque(
            [
                ((self.target - trajectory["nominal"]) / 2).copy()
                for _ in range(self.latency)
            ]
        )
        # Each environment owns its model. Alter parameters only for explicit tests.
        friction = self.condition.get("friction", 1.0)
        self.m.geom_friction[:, 0] *= friction
        mass = self.condition.get("mass", 1.0)
        self.m.body_mass[:] *= mass
        self.m.body_inertia[:] *= mass
        if mass != 1.0 or self.condition.get("recompute_constants", False):
            mujoco.mj_setConst(self.m, self.d)
            mujoco.mj_setState(self.m, self.d, trajectory["initial_state"], STATE_SPEC)
        self.weight = self.m.body_mass.sum() * 9.81
        qnoise = self.condition.get("joint_position", 0.0)
        vnoise = self.condition.get("joint_velocity", 0.0)
        self.d.qpos[self.qids] += self.rng.uniform(-qnoise, qnoise, 12)
        self.d.qvel[self.vids] += self.rng.uniform(-vnoise, vnoise, 12)
        angle = self.condition.get("orientation", 0.0)
        if angle:
            delta = np.zeros(self.m.nv)
            delta[3:6] = self.rng.uniform(-angle, angle, 3)
            mujoco.mj_integratePos(self.m, self.d.qpos, delta, 1.0)
        self.d.qpos[:2] += self.rng.uniform(
            -self.condition.get("position", 0), self.condition.get("position", 0), 2
        )
        mujoco.mj_forward(self.m, self.d)
        # Accept small contact penetration inherent in nominal compliant contacts;
        # reject any non-foot contact or excess penetration instead of repairing state.
        if np.any(self.d.qpos[self.qids] < self.bounds[:, 0]) or np.any(
            self.d.qpos[self.qids] > self.bounds[:, 1]
        ):
            raise ValueError("Invalid reset: joint limit")
        for c in self.d.contact:
            bodies = self.m.geom_bodyid[[c.geom1, c.geom2]]
            if c.dist < -0.003 or (
                c.dist < 0 and any(b and b not in self.feet for b in bodies)
            ):
                raise ValueError("Invalid reset: penetration or prohibited contact")
        self.anchor = self.d.geom_xpos[self.foot_geoms[:2], :2].copy()

    def observation(self):
        r = self.d.xmat[self.base].reshape(3, 3)
        obs = np.r_[
            r.T @ [0, 0, -1],
            self.d.qvel[3:6],
            self.d.qpos[self.qids] - self.trajectory["nominal"],
            self.d.qvel[self.vids] / self.speed,
            (self.target - self.trajectory["nominal"]) / ACTION_SCALE,
            self.index * CONTROL_DT / 8,
        ].astype(np.float32)
        return obs + self.rng.normal(
            0, self.condition.get("observation_noise", 0), obs.shape
        ).astype(np.float32)

    def balance_features(self):
        """Legacy teacher only: preserve original qvel and COM channels."""
        obs = np.zeros(93, np.float32)
        obs[:30] = self.observation()[:30]
        obs[3:6] = self.d.qvel[3:6]
        mujoco.mj_subtreeVel(self.m, self.d)
        obs[55] = self.d.subtree_com[self.base, 2] / 0.6
        obs[56:59] = self.d.subtree_linvel[self.base]
        return obs

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Expected finite 12-action vector")
        action = action.clip(-1, 1)
        self.action_variation += float(np.mean((action - self.previous_action) ** 2))
        self.previous_action = action.copy()
        self.queue.append(action.copy())
        action = self.queue.popleft()
        desired = np.clip(
            self.trajectory["nominal"] + ACTION_SCALE * action, *self.bounds.T
        )
        previous_target = self.target.copy()
        self.target += np.clip(
            desired - self.target,
            -0.7 * self.speed * CONTROL_DT,
            0.7 * self.speed * CONTROL_DT,
        )
        # Actuator variations for robustness studies; every default reproduces
        # the nominal motor model exactly.
        kp = KP * self.condition.get("kp_scale", 1.0)
        kd = KD * self.condition.get("kd_scale", 1.0)
        delay = int(self.condition.get("command_delay_substeps", 0))
        zero_offset = getattr(self, "joint_zero_offset", 0.0)
        for substep in range(round(CONTROL_DT / PHYSICS_DT)):
            velocity = self.d.qvel[self.vids].copy()
            effort = self.effort * self.condition.get("strength", 1.0)
            # The new target reaches the motors `delay` physics steps late, and
            # the motor's own position reading may have a zero offset.
            target = previous_target if substep < delay else self.target
            torque = np.clip(
                kp * (target - self.d.qpos[self.qids] - zero_offset) - kd * velocity,
                -effort,
                effort,
            )
            torque[(abs(velocity) >= self.speed) & (torque * velocity > 0)] = 0
            self.d.qfrc_applied[:] = 0
            self.d.qfrc_applied[self.vids] = torque
            self.d.xfrc_applied[:] = 0
            push = self.condition.get("push")
            if push and push["start"] <= self.d.time < push["start"] + push["duration"]:
                # MuJoCo xfrc is world force/torque at body's COM.
                self.d.xfrc_applied[self.base] = push["wrench"]
            mujoco.mj_step(self.m, self.d)
            if hasattr(self, "after_physics_step"):
                self.after_physics_step()
            mujoco.mj_forward(self.m, self.d)
            self.measure(velocity)
            if self.takeoff and self.takeoff_time is None:
                self.takeoff_time = float(self.d.time)
            if self.captured and self.capture_time is None:
                self.capture_time = float(self.d.time)
            if self.hold > 0 and self.hold_start_time is None:
                self.hold_start_time = float(self.d.time - PHYSICS_DT)
            self.record_metrics()
        self.index += 1
        ref = mujoco.MjData(self.m)
        mujoco.mj_setState(
            self.m,
            ref,
            self.trajectory["reference_states"][
                min(self.index, len(self.trajectory["actions"]))
            ],
            STATE_SPEC,
        )
        # Bounded imitation shaping, with strict hold and safety rewards separated.
        tracking = np.mean((self.d.qpos[self.qids] - ref.qpos[self.qids]) ** 2)
        orientation = 1 - abs(float(self.d.qpos[3:7] @ ref.qpos[3:7]))
        reward = float(
            np.exp(-5 * tracking - 10 * orientation)
            + 2 * (self.hold > 0)
            - 5 * (not self.safe)
        )
        done = not self.safe or self.index >= len(self.trajectory["actions"])
        return self.observation(), reward, done

    def record_metrics(self):
        r = self.d.xmat[self.base].reshape(3, 3)
        # Positive shoulder-to-hip longitudinal direction is -body x.
        orientation = np.arccos(np.clip(-r[2, 0], -1, 1))
        axis = self.trajectory["pitch_axis"]
        horizontal = r[:, 0] @ [axis[1], -axis[0], 0]
        pitch_error = np.arctan2(horizontal, -r[2, 0])
        jac = np.zeros((3, self.m.nv))
        slip = 0.0
        for g in self.foot_geoms[:2]:
            mujoco.mj_jacGeom(self.m, self.d, jac, None, int(g))
            slip += 0.5 * np.linalg.norm((jac @ self.d.qvel)[:2]) * PHYSICS_DT
        kinetic = 0.0
        for b in range(1, self.m.nbody):
            name = self.m.body(b).name
            if name.startswith(("FL_", "FR_")):
                linear_jac = np.zeros((3, self.m.nv))
                angular_jac = np.zeros((3, self.m.nv))
                mujoco.mj_jacBodyCom(self.m, self.d, linear_jac, angular_jac, b)
                linear = linear_jac @ self.d.qvel
                rot = self.d.ximat[b].reshape(3, 3)
                omega = rot.T @ (angular_jac @ self.d.qvel)
                kinetic += 0.5 * (
                    self.m.body_mass[b] * (linear @ linear)
                    + np.sum(self.m.body_inertia[b] * omega**2)
                )
        row = (
            float(self.d.time),
            float(orientation),
            float(pitch_error),
            float(slip),
            float(kinetic),
            float(self.hold),
        )
        self.samples.append(row)
        if self.hold > 0:
            self.current_samples.append(row)
            if len(self.current_samples) > len(self.best_samples):
                self.best_samples = self.current_samples.copy()
        else:
            self.current_samples = []

    def result(self):
        # First five seconds of longest strict interval, or available failed hold.
        rows = np.asarray(self.best_samples[:2500])
        metrics = None
        if len(rows):
            signs = np.sign(rows[:, 2][abs(rows[:, 2]) > np.deg2rad(0.1)])
            changes = int(np.sum(signs[1:] != signs[:-1]))
            duration = len(rows) * PHYSICS_DT
            metrics = dict(
                duration_seconds=duration,
                orientation_mean_deg=float(np.rad2deg(rows[:, 1].mean())),
                orientation_rms_deg=float(
                    np.rad2deg(np.sqrt(np.mean(rows[:, 1] ** 2)))
                ),
                orientation_max_deg=float(np.rad2deg(rows[:, 1].max())),
                pitch_rms_deg=float(np.rad2deg(np.sqrt(np.mean(rows[:, 2] ** 2)))),
                pitch_sign_changes=changes,
                pitch_sign_changes_hz=changes / duration,
                front_slip_m=float(rows[:, 3].sum()),
                front_leg_kinetic_J=float(rows[:, 4].mean()),
            )
        recovery = None
        push = self.condition.get("push")
        if push:
            end = push["start"] + push["duration"]
            after = np.asarray([r for r in self.samples if r[0] >= push["start"]])
            restored = [r for r in self.samples if r[0] >= end + 0.2 and r[5] >= 0.2]
            was_stable = any(r[5] >= 0.2 for r in self.samples if r[0] <= push["start"])
            recovered = bool(restored and self.safe)
            seconds = max(0.0, restored[0][0] - 0.2 - end) if recovered else None
            recovery = dict(
                recovered=recovered,
                previously_stable=was_stable,
                interpretation="recovery" if was_stable else "post_push_acquisition",
                seconds_after_push_end=seconds,
                control_steps=None
                if seconds is None
                else int(np.ceil(seconds / CONTROL_DT)),
                max_vertical_axis_deviation_deg=float(np.rad2deg(after[:, 1].max()))
                if len(after)
                else None,
                criterion=(
                    "0.2s continuous original strict gate after pulse; "
                    "legal continuation"
                ),
            )
        success = bool(
            self.best_hold >= 5 and self.takeoff and self.captured and self.safe
        )
        return dict(
            complete_success=success,
            disturbance_recovery=recovery,
            strict_hold_seconds=self.best_hold,
            certified_takeoff=self.takeoff,
            first_takeoff_time=self.takeoff_time,
            first_capture_time=self.capture_time,
            first_strict_hold_start=self.hold_start_time,
            front_capture=self.captured,
            prefix_safe=self.safe,
            illegal_contacts=self.illegal_contacts,
            max_pitch_deg=float(np.rad2deg(self.peak_pitch)),
            hold_metrics=metrics,
            action_change_mean_square=self.action_variation / max(self.index, 1),
            failure=None
            if success
            else ("safety_gate" if not self.safe else "insufficient_strict_hold"),
        )
