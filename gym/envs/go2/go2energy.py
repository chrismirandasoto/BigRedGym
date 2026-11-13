"""Kick up and balance by regulating the energy of a front-foot pendulum.

Reduced-order model: the whole robot is a point mass at its centre of mass,
swinging about the midpoint of the front feet. With height h of the mass
above that pivot and speed v, its energy is

    E = m g h + m v^2 / 2

and the only resting state with E = m g l is the upright one, where l is the
pivot-to-mass distance in the handstand. Rewards use the mass-independent
ratio e = E / (m g l) = (h + v^2 / 2g) / l.

Every state on the swing-up path also has e = 1, so the energy reward alone
cannot tell a moving robot from a balanced one. A second reward multiplies it
by how close the mass is to directly above the pivot.

Nothing happens until the handstand command turns on: a random delay in
training, a key press in playback (see teleop_bindings.py).
"""

import math

import torch

from gym.envs.go2.go2handstand import Go2Handstand
from gym.utils.torch_quat import quat_apply

GRAVITY = 9.81
# A trigger step no episode reaches: the command stays off until set by hand.
NEVER = torch.iinfo(torch.long).max


class Go2Energy(Go2Handstand):
    @property
    def handstand_command(self):
        """[num_envs, 1] observation: 1 once the handstand is requested."""
        return self._handstand_command.unsqueeze(1)

    @property
    def pendulum_obs(self):
        """[num_envs, 2] privileged critic input: energy ratio, cos(angle)."""
        energy, angle = self._pendulum()
        return torch.stack((energy, angle.cos()), dim=1)

    def _init_buffers(self):
        super()._init_buffers()
        p, t = self.cfg.pendulum, self.cfg.handstand_trigger
        values = (p.length, p.energy_target, p.energy_sigma, p.angle_sigma)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError("pendulum length, target, and sigmas must be positive")
        low, high = t.delay_s
        if not 0 <= low <= high or not math.isfinite(high):
            raise ValueError("handstand_trigger.delay_s must be a finite [min, max]")
        if (
            high + self.cfg.handstand_curriculum.hold_seconds
            >= self.max_episode_length_s
        ):
            raise ValueError("episode too short to hold after the latest trigger")
        if not 0 <= t.never_fraction < 1:
            raise ValueError("handstand_trigger.never_fraction must be in [0, 1)")
        try:
            offset = self._backend.link_com_offset
        except NotImplementedError as error:
            raise NotImplementedError(
                "go2energy needs per-link centres of mass; this backend does not "
                "expose link_com_offset (the MuJoCo backends do)"
            ) from error
        self._link_com_offset = offset.expand(self.num_envs, -1, -1)
        self._handstand_command = torch.zeros(self.num_envs, device=self.device)
        self._trigger_step = torch.full(
            (self.num_envs,), NEVER, dtype=torch.long, device=self.device
        )
        # Set by the teleop interface: the key, not a timer, gives the command.
        self.manual_handstand_trigger = False
        knee = self.cfg.reward_settings.kick_knee_target
        front = [0.0, -math.pi / 2 - knee / 2, knee]
        rear = [0.0, math.pi / 2 - knee / 2, knee]
        self._handstand_posture = self.dof_pos.new_tensor(front * 2 + rear * 2)

    def _standing_posture(self):
        """Zero-action PD reference: standing, turning into the handstand pose.

        Once commanded, the reference moves linearly from the standing posture
        to the handstand posture as the body pitches from level to vertical.
        These are the poses the reset curriculum starts from (to within 0.03
        rad), so a zero action holds the robot's shape at every stage of the
        swing instead of folding the legs back to standing.
        """
        standing = super()._standing_posture()
        gravity = self.projected_gravity
        # Positive pitch raises the rear above the front.
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        progress = self._handstand_command * (pitch / (math.pi / 2)).clamp(0, 1)
        return standing + progress[:, None] * (self._handstand_posture - standing)

    # ── Handstand command ────────────────────────────────────────────────

    def _reset_system(self, reset_mask):
        # The new command must be in place before the reset matches the PD
        # targets to the sampled pose, because the reference depends on it.
        super()._reset_system(reset_mask)
        t = self.cfg.handstand_trigger
        low, high = t.delay_s
        delay = low + (high - low) * torch.rand(self.num_envs, device=self.device)
        steps = (delay / self.dt).ceil().long()
        standing = self._reset_stage == 0
        never = standing & (
            torch.rand(self.num_envs, device=self.device) < t.never_fraction
        )
        if self.manual_handstand_trigger:
            never = standing
        steps[never] = NEVER
        # A reset that starts off the ground is already part of the handstand.
        steps[~standing] = 0
        self._trigger_step[reset_mask] = steps[reset_mask]
        self._handstand_command[reset_mask] = (steps[reset_mask] == 0).float()
        # An episode that is never asked to kick up says nothing about holds.
        self._primary_reset[reset_mask] &= ~never[reset_mask]

    def _post_decimation_step(self):
        super()._post_decimation_step()
        due = self.episode_length_buf >= self._trigger_step
        self._handstand_command[due] = 1.0

    @torch.inference_mode()
    def trigger_handstand(self):
        """Request the handstand in every environment; it holds until reset."""
        self._handstand_command[:] = 1.0

    @torch.inference_mode()
    def await_handstand_trigger(self):
        """Stand until trigger_handstand() instead of after a random delay."""
        self.manual_handstand_trigger = True
        self._trigger_step[self._handstand_command == 0] = NEVER

    # ── Reduced-order model ──────────────────────────────────────────────

    def _pendulum(self):
        """Energy ratio e and angle from vertical [rad] of the point mass."""
        mass = self._backend.link_mass
        weight = (mass / mass.sum(dim=1, keepdim=True)).unsqueeze(-1)
        # Body states describe link origins; the mass sits at each link's
        # centre of mass, which also moves as the link rotates about its origin.
        offset = quat_apply(self._rigid_body_quat, self._link_com_offset)
        position = (weight * (self._rigid_body_pos + offset)).sum(dim=1)
        velocity = self._rigid_body_lin_vel + torch.cross(
            self._rigid_body_ang_vel, offset, dim=-1
        )
        velocity = (weight * velocity).sum(dim=1)
        pivot = self._rigid_body_pos[:, self.feet_indices[:2]].mean(dim=1)
        arm = position - pivot
        height = arm[:, 2]
        kinetic = velocity.square().sum(dim=1) / (2 * GRAVITY)
        energy = (height + kinetic) / self.cfg.pendulum.length
        angle = torch.atan2(arm[:, :2].norm(dim=1), height)
        return energy, angle

    def _energy_score(self, energy):
        p = self.cfg.pendulum
        score = torch.exp(-((energy - p.energy_target) / p.energy_sigma).square())
        if p.require_front_support:
            # The pivot exists only while both front feet carry load; without
            # this, hopping in place would also reach the target energy.
            load = self._handstand_loads()[:, :2].amin(dim=1)
            score = score * (load / self.cfg.reward_settings.min_front_load).clamp(0, 1)
        return score

    def _upright_score(self):
        """In [0, 1]; 1 only at the target energy with the mass over the pivot."""
        energy, angle = self._pendulum()
        position = torch.exp(-(angle / self.cfg.pendulum.angle_sigma).square())
        return self._energy_score(energy) * position

    def _update_handstand_hold(self):
        # Curriculum holds are judged on the pendulum, not the staged score.
        held = (
            (self._upright_score() >= self.cfg.handstand_curriculum.hold_score)
            & (self._handstand_command > 0)
            & ~self.terminated
            & ~self.timed_out
        )
        self._hold_time[:] = torch.where(held, self._hold_time + self.dt, 0.0)
        self._best_hold[:] = torch.maximum(self._best_hold, self._hold_time)

    # ── Rewards ──────────────────────────────────────────────────────────

    def _reward_pendulum_energy(self):
        energy, _ = self._pendulum()
        return self._handstand_command * self._energy_score(energy)

    def _reward_pendulum_upright(self):
        return self._handstand_command * self._upright_score()

    def _reward_stand_ready(self):
        """Before the command: level body at the requested height."""
        level = torch.exp(
            -self.projected_gravity[:, :2].square().sum(dim=1)
            / self.cfg.reward_settings.tracking_sigma
        )
        return (
            (1 - self._handstand_command) * level * self._reward_tracking_base_height()
        )

    def _reward_termination(self):
        # The runner multiplies all rewards by dt. A fall is one event, so keep
        # its configured cost independent of the control frequency.
        return -self.terminated.float() / self.dt
