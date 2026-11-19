"""Hold the handstand the released kick reaches, under sensor and actuator error.

Episodes start from robot states recorded at the released controller's 2.2 s
hand-over (resources/handstand/handover_states.npz). The policy then commands
joint targets through the same path the robot uses: offset from the held
posture, joint-limit clip, slew limit, 50 Hz targets into a 500 Hz PD loop.
It observes only what go2_deploy provides (handstand_hold_features.py).

Per episode the task draws gains, torque limit, command delay, joint zero
offsets, gyro bias and noise, IMU attitude error and encoder noise; the
framework draws friction and link masses per environment and pushes the body.
The critic also sees simulator truth.
"""

import numpy as np
import torch

from gym import GYM_ROOT_DIR
from gym.envs.go2.go2 import Go2
from gym.envs.go2.handstand_hold_features import (
    FRAME,
    features,
    frame_center_and_scale,
    joint_offset,
)
from gym.utils import masked_update
from gym.utils.torch_quat import quat_apply

WXYZ_TO_XYZW = [1, 2, 3, 0]


def _rotation_from_vector(vector):
    """Rodrigues: rotation vectors (N, 3) -> matrices (N, 3, 3)."""
    angle = vector.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    x, y, z = (vector / angle).unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).view(-1, 3, 3)
    eye = torch.eye(3, device=vector.device).expand_as(k)
    angle = angle[..., None]
    return eye + torch.sin(angle) * k + (1 - torch.cos(angle)) * (k @ k)


class Go2HandstandHold(Go2):
    # ── Set-up ───────────────────────────────────────────────────────────

    def _init_buffers(self):
        super()._init_buffers()
        h = self.cfg.handstand_hold
        path = h.dataset.format(GYM_ROOT_DIR=GYM_ROOT_DIR)
        with np.load(path, allow_pickle=False) as data:
            d = {key: data[key] for key in data.files}
        names = [str(n) for n in d["joint_names"]]
        if names != list(self.actuated_dof_names):
            raise ValueError(
                f"hand-over joint order {names} != {self.actuated_dof_names}"
            )
        if h.history > d["history"].shape[1]:
            raise ValueError("hand-over dataset holds fewer history frames than asked")

        def tensor(value, dtype=torch.float):
            return torch.as_tensor(np.asarray(value), dtype=dtype, device=self.device)

        # Drop recorded states that would end the episode at once.
        qpos = tensor(d["qpos"])
        rotation = tensor(d["target_rotation"])
        up = quat_apply(
            qpos[:, 3:7][:, WXYZ_TO_XYZW],
            (rotation.T @ tensor([0.0, 0.0, 1.0])).expand(len(qpos), 3),
        )
        usable = (torch.acos(up[:, 2].clamp(-1, 1)) < h.max_tilt) & (
            qpos[:, 2] > h.min_base_height
        )
        if not usable.any():
            raise ValueError("no usable hand-over state in the dataset")
        self._starts = dict(
            qpos=qpos[usable],
            qvel=tensor(d["qvel"])[usable],
            target=tensor(d["target"])[usable],
            history=tensor(d["history"][:, : h.history])[usable],
        )
        self.num_starts = int(usable.sum())
        self._nominal = tensor(d["nominal"])
        self._hold_target = tensor(d["hold_target"])
        bounds = tensor(d["joint_bounds"])
        self._lower, self._upper = bounds[:, 0], bounds[:, 1]
        self._speed = tensor(d["speed"])
        self._effort = tensor(d["effort"])
        self._target_rotation = tensor(d["target_rotation"])
        self._target_com_height = float(d["target_com_height"])
        # The body axis that points up in the handstand.
        self._up_in_body = self._target_rotation.T @ tensor([0.0, 0.0, 1.0])
        center, scale = frame_center_and_scale(
            d["nominal"], d["hold_target"], d["speed"], d["target_rotation"]
        )
        self._frame_center, self._frame_scale = (
            center.to(self.device),
            scale.to(self.device),
        )

        n, j = self.num_envs, self.num_actuators
        zeros = lambda *shape: torch.zeros(*shape, device=self.device)  # noqa: E731
        self.hold_action = zeros(n, j)
        self._sample = torch.zeros(n, dtype=torch.long, device=self.device)
        self.hold_target = self._hold_target.repeat(n, 1)
        self._previous_target = self.hold_target.clone()
        self._frame = zeros(n, FRAME)
        self._history = zeros(n, h.history, FRAME)
        self._substep = 0
        # Per-episode errors.
        self._delay = torch.zeros(n, dtype=torch.long, device=self.device)
        self._strength = torch.ones(n, device=self.device)
        self._zero_offset = zeros(n, j)
        self._gyro_bias = zeros(n, 3)
        self._attitude = torch.eye(3, device=self.device).repeat(n, 1, 1)
        self._noise_scale = zeros(n)  # ramp factor at the episode's start
        # Physics summaries, refreshed every control step.
        self._com = zeros(n, 3)
        self._com_velocity = zeros(n, 3)
        self._loads = zeros(n, 4)
        self._tilt = zeros(n)
        self._rotation_error = zeros(n)
        self._support_error = zeros(n)
        self._front_slip = zeros(n)
        self._link_com_offset = self._backend.link_com_offset
        self._diverged = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.diverged_resets = 0

    # ── Resets ───────────────────────────────────────────────────────────

    def reset_to_handover(self, reset_mask):
        """Place each reset robot in a recorded hand-over state."""
        sample = torch.randint(self.num_starts, (self.num_envs,), device=self.device)
        self._sample = torch.where(reset_mask, sample, self._sample)
        qpos = self._starts["qpos"][sample]
        qvel = self._starts["qvel"][sample]
        quat = qpos[:, 3:7][:, WXYZ_TO_XYZW]
        root = torch.cat(
            [qpos[:, :3], quat, qvel[:, :3], quat_apply(quat, qvel[:, 3:6])], dim=1
        )
        masked_update(self.root_states, root, reset_mask)
        masked_update(self.dof_pos, qpos[:, 7:], reset_mask)
        masked_update(self.dof_vel, qvel[:, 6:], reset_mask)

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        if not reset_mask.any():
            return
        h = self.cfg.handstand_hold
        target = self._starts["target"][self._sample]
        masked_update(self.hold_target, target, reset_mask)
        masked_update(self._previous_target, target, reset_mask)
        self._history[reset_mask] = self._starts["history"][self._sample][reset_mask]
        self.hold_action[reset_mask] = 0.0

        # Errors grow from zero to full size over the first ramp_steps.
        ramp = min(1.0, self.common_step_counter / max(1, h.ramp_steps))
        n = self.num_envs

        def uniform(low, high, *shape):
            return low + (high - low) * torch.rand(n, *shape, device=self.device)

        delay = torch.floor(uniform(0.0, ramp * h.command_delay_substeps + 0.999))
        masked_update(self._delay, delay.long(), reset_mask)
        low, high = h.strength
        strength = 1.0 - ramp * (1.0 - uniform(low, high))
        masked_update(self._strength, strength, reset_mask)
        masked_update(
            self._zero_offset, ramp * uniform(-1, 1, 12) * h.zero_offset, reset_mask
        )
        masked_update(
            self._gyro_bias, ramp * uniform(-1, 1, 3) * h.gyro_bias, reset_mask
        )
        tilt = _rotation_from_vector(ramp * uniform(-1, 1, 3) * h.attitude_bias)
        self._attitude[reset_mask] = tilt[reset_mask]
        masked_update(
            self._noise_scale, torch.full((n,), ramp, device=self.device), reset_mask
        )
        self._update_physics_summaries()
        self._measure_frame(reset_mask)

    # ── Control path ─────────────────────────────────────────────────────

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        # The observation this action came from becomes the newest past frame.
        self._history = torch.cat([self._frame[:, None], self._history[:, :-1]], 1)
        offset = joint_offset(self.hold_action, self._nominal, self._hold_target)
        units = (offset / 2).clamp(-1, 1)
        desired = torch.maximum(
            torch.minimum(self._nominal + 2 * units, self._upper), self._lower
        )
        limit = self.cfg.handstand_hold.slew_fraction * self._speed * self.dt
        self._previous_target = self.hold_target
        self.hold_target = self.hold_target + torch.maximum(
            torch.minimum(desired - self.hold_target, limit), -limit
        )
        self._substep = 0

    def _compute_torques(self):
        q = self.dof_pos.index_select(1, self.actuated_dof_indices)
        dq = self.dof_vel.index_select(1, self.actuated_dof_indices)
        # A new target reaches the motor `delay` physics steps late.
        late = (self._substep < self._delay)[:, None]
        target = torch.where(late, self._previous_target, self.hold_target)
        torque = self.p_gains * (target - q - self._zero_offset) - self.d_gains * dq
        limit = self._effort * self._strength[:, None]
        torque = torch.maximum(torch.minimum(torque, limit), -limit)
        # Motors give no torque that speeds a joint past its limit.
        overspeed = (dq.abs() >= self._speed) & (torque * dq > 0)
        torque = torch.where(overspeed, torch.zeros_like(torque), torque)
        self._substep += 1
        return torque.view(self.torques.shape)

    def _post_decimation_step(self):
        super()._post_decimation_step()
        self._guard_diverged()
        self._update_physics_summaries()
        self._measure_frame(None)

    def _guard_diverged(self):
        """End and neutralise any world whose simulation has blown up.

        One non-finite world would otherwise reach the observations, the
        critic's running normaliser and the network weights of every world.
        Its state is zeroed only until the reset that follows this step.
        """
        h = self.cfg.handstand_hold
        state = [
            self.root_states,
            self.dof_pos,
            self.dof_vel,
            self._rigid_body_state.view(self.num_envs, -1),
            self.contact_forces.view(self.num_envs, -1),
            self.torques,
        ]
        bad = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for tensor in state:
            bad |= ~torch.isfinite(tensor).all(dim=1)
        bad |= self.dof_vel.abs().amax(dim=1) > h.max_joint_speed
        bad |= self.root_states[:, 7:13].abs().amax(dim=1) > h.max_body_speed
        self._diverged = bad
        if not bad.any():
            return
        for tensor in state:
            torch.nan_to_num_(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        self.root_states[bad, 3:7] = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=self.device
        )
        self.base_quat[:] = self.root_states[:, 3:7]
        for name in ("base_lin_vel", "base_ang_vel", "projected_gravity"):
            torch.nan_to_num_(getattr(self, name), nan=0.0, posinf=0.0, neginf=0.0)
        self.diverged_resets += int(bad.sum())
        if self.diverged_resets == int(bad.sum()) or self.diverged_resets % 100 < int(
            bad.sum()
        ):
            print(
                f"[go2handstandhold] {self.diverged_resets} diverged simulations "
                "reset so far",
                flush=True,
            )

    # ── Sensors ──────────────────────────────────────────────────────────

    def _measure_frame(self, mask):
        """The 42 values go2_deploy would report, with this episode's errors."""
        h = self.cfg.handstand_hold
        noise = self._noise_scale[:, None]

        def randn(*shape):
            return torch.randn(self.num_envs, *shape, device=self.device)

        gravity = torch.einsum("nji,nj->ni", self._attitude, self.projected_gravity)
        gyro = self.base_ang_vel + self._gyro_bias + noise * h.gyro_noise * randn(3)
        q = self.dof_pos.index_select(1, self.actuated_dof_indices)
        dq = self.dof_vel.index_select(1, self.actuated_dof_indices)
        q = q + self._zero_offset + noise * h.encoder_noise * randn(12)
        dq = dq + noise * h.encoder_velocity_noise * randn(12)
        frame = torch.cat(
            [
                gravity,
                gyro,
                q - self._nominal,
                dq / self._speed,
                (self.hold_target - self._nominal) / 2,
            ],
            dim=1,
        )
        if mask is None:
            self._frame = frame
        else:
            self._frame[mask] = frame[mask]

    @property
    def hold_obs(self):
        frames = torch.cat([self._frame[:, None], self._history], 1)
        out = features(frames, self._frame_center, self._frame_scale)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    @property
    def hold_privileged(self):
        """Simulator truth and this episode's errors, for the critic only."""
        feet = self._rigid_body_pos[:, self.feet_indices]
        return torch.cat(
            [
                self.projected_gravity,
                self.base_ang_vel,
                self.base_lin_vel,
                self.root_states[:, 2:3],
                self._com - feet[:, :2].mean(dim=1),
                self._com_velocity,
                self._loads,
                feet[:, 2:, 2] - 0.022,
                self.p_gains[:, :1] / 60.0 - 1.0,
                self.d_gains[:, :1] / 3.0 - 1.0,
                self._strength[:, None] - 1.0,
                self._delay[:, None].float() / 3.0,
                100.0 * self._zero_offset,
                50.0 * self._gyro_bias,
                10.0
                * (
                    self._attitude[:, :, 2]
                    - torch.tensor([0.0, 0.0, 1.0], device=self.device)
                ),
            ],
            dim=1,
        ).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    # ── Physics summaries ────────────────────────────────────────────────

    def _update_physics_summaries(self):
        mass = self._backend.link_mass
        weight = (mass / mass.sum(dim=1, keepdim=True)).unsqueeze(-1)
        offset = quat_apply(
            self._rigid_body_quat,
            self._link_com_offset.expand(self.num_envs, -1, -1),
        )
        self._com = (weight * (self._rigid_body_pos + offset)).sum(dim=1)
        velocity = self._rigid_body_lin_vel + torch.cross(
            self._rigid_body_ang_vel, offset, dim=-1
        )
        self._com_velocity = (weight * velocity).sum(dim=1)
        body_weight = 9.81 * mass.sum(dim=1, keepdim=True)
        self._loads = (
            self.contact_forces[:, self.feet_indices, 2].clamp(min=0) / body_weight
        )

        quat = self.base_quat
        up = quat_apply(quat, self._up_in_body.expand(self.num_envs, 3))
        self._tilt = torch.acos(up[:, 2].clamp(-1, 1))
        # Angle of R_target^T R, from its trace.
        eye = torch.eye(3, device=self.device)
        trace = sum(
            (
                quat_apply(quat, eye[i].expand(self.num_envs, 3))
                * self._target_rotation[:, i]
            ).sum(-1)
            for i in range(3)
        )
        self._rotation_error = torch.acos(((trace - 1) / 2).clamp(-1, 1))

        feet = self._rigid_body_pos[:, self.feet_indices]
        a, b = feet[:, 0, :2], feet[:, 1, :2]
        segment = b - a
        u = ((self._com[:, :2] - a) * segment).sum(-1) / segment.square().sum(-1).clamp(
            min=1e-10
        )
        nearest = a + u.clamp(0, 1)[:, None] * segment
        self._support_error = (self._com[:, :2] - nearest).norm(dim=-1)
        self._front_slip = (
            self._rigid_body_lin_vel[:, self.feet_indices[:2], :2]
            .norm(dim=-1)
            .amax(dim=1)
        )

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        h = self.cfg.handstand_hold
        self.terminated |= self._tilt > h.max_tilt
        self.terminated |= self.root_states[:, 2] < h.min_base_height
        self.terminated |= self._diverged

    # ── Rewards (all in [0, 1] unless a penalty) ─────────────────────────

    def _strict_hold(self):
        """The strict handstand gate, approximately (no clearance geometry)."""
        s = self.cfg.reward_settings
        feet = self._rigid_body_pos[:, self.feet_indices]
        com_speed = self._com_velocity.norm(dim=-1)
        front_force = self.contact_forces[:, self.feet_indices[:2], 2].sum(1)
        weight = 9.81 * self._backend.link_mass.sum(dim=1)
        return (
            ((feet[:, 2:, 2] - 0.022).amin(dim=1) > 0.10)
            & (self._loads[:, 2:] < 0.02).all(dim=1)
            & (self._loads[:, :2] > s.min_front_load).all(dim=1)
            & (self._rotation_error < np.deg2rad(10))
            & ((self.root_states[:, 2] - 0.57).abs() < 0.06)
            & ((self._com[:, 2] - self._target_com_height).abs() < 0.06)
            & (com_speed < 0.25)
            & (self.base_ang_vel.norm(dim=-1) < 0.5)
            & (self._support_error < 0.04)
            & (self._front_slip < 0.05)
            & (front_force / weight < 4.0)
        )

    def _reward_strict_hold(self):
        return self._strict_hold().float()

    def _reward_upright(self):
        return torch.exp(-(self._tilt / self.cfg.reward_settings.tilt_sigma).square())

    def _reward_heading(self):
        sigma = self.cfg.reward_settings.heading_sigma
        return torch.exp(-(self._rotation_error / sigma).square())

    def _reward_com_over_support(self):
        sigma = self.cfg.reward_settings.support_sigma
        return torch.exp(-(self._support_error / sigma).square())

    def _reward_stillness(self):
        s = self.cfg.reward_settings
        com = (self._com_velocity.norm(dim=-1) / s.com_speed_sigma).square()
        spin = (self.base_ang_vel.norm(dim=-1) / s.angular_speed_sigma).square()
        return torch.exp(-com - spin)

    def _reward_heights(self):
        sigma = self.cfg.reward_settings.height_sigma
        base = ((self.root_states[:, 2] - 0.57) / sigma).square()
        com = ((self._com[:, 2] - self._target_com_height) / sigma).square()
        return torch.exp(-base - com)

    def _reward_rear_clearance(self):
        feet = self._rigid_body_pos[:, self.feet_indices]
        clearance = (feet[:, 2:, 2] - 0.022).amin(dim=1)
        return (clearance / self.cfg.reward_settings.rear_clearance).clamp(0, 1)

    def _reward_front_support(self):
        need = self.cfg.reward_settings.min_front_load
        return (self._loads[:, :2].amin(dim=1) / need).clamp(0, 1)

    def _reward_front_slip(self):
        return -self._front_slip.clamp(max=1.0)

    def _reward_action_rate(self):
        change = (self.hold_target - self._previous_target) / (self._speed * self.dt)
        return -change.square().mean(dim=1)

    def _reward_dof_vel_limits(self):
        dq = self.dof_vel.index_select(1, self.actuated_dof_indices)
        return -((dq.abs() / self._speed - 0.8).clamp(min=0)).sum(dim=1)

    def _reward_termination(self):
        # The runner scales every reward by dt; a fall is one event.
        return -self.terminated.float() / self.dt
