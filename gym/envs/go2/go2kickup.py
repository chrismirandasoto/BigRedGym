"""Kick from standing, then give the existing balance policy full control."""

import math

import torch

from gym.envs.go2.go2balance import Go2Balance
from gym.envs.go2.kickup_cpg import KickupCPG


class Go2Kickup(Go2Balance):
    def _init_buffers(self):
        super()._init_buffers()
        self.kickup = KickupCPG(self.num_envs, self.device, self.cfg.kickup)
        c = self.cfg.kickup_curriculum
        if (
            len(c.pitch_degrees) != len(c.amplitudes)
            or not c.amplitudes
            or not 0 <= c.initial_level < len(c.amplitudes)
            or any(not math.isfinite(x) or not 0 < x <= 1 for x in c.amplitudes)
            or any(not math.isfinite(x) or not 0 < x < 90 for x in c.pitch_degrees)
            or c.min_episodes < 1
            or c.required_batches < 1
            or not 0 < c.success_fraction <= 1
            or not math.isfinite(c.final_hold_s)
            or c.final_hold_s <= 0
            or not math.isfinite(c.residual_limit_rad)
            or c.residual_limit_rad < 0
        ):
            raise ValueError("invalid kickup curriculum")
        self.kickup_level = c.initial_level
        self._kick_episodes = self._kick_successes = self._kick_passes = 0
        self.kickup_success_rate = 0.0
        self._kick_level = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._kick_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._kick_started = torch.zeros_like(self._kick_success)

    def reset_to_handstand(self, reset_mask):
        if self._tilt_test is not None:
            raise ValueError("the tilt test runs on go2balance, not go2kickup")
        self.reset_to_basic(reset_mask)
        self._reset_stage[reset_mask] = 2
        self._reset_level[reset_mask] = 0
        self._primary_reset[reset_mask] = False

    def _reset_idx(self, reset_mask):
        c = self.cfg.kickup_curriculum
        if c.enabled:
            eligible = (
                reset_mask
                & self._kick_started
                & (self.terminated | self.timed_out)
                & (self._kick_level == self.kickup_level)
            )
            final = self._kick_level == len(c.amplitudes) - 1
            success = torch.where(
                final,
                self.kickup.handed_off & (self._best_hold >= c.final_hold_s),
                self._kick_success,
            )
            self._kick_episodes += int(eligible.sum())
            self._kick_successes += int((eligible & success).sum())
            if self._kick_episodes >= c.min_episodes:
                self.kickup_success_rate = self._kick_successes / self._kick_episodes
                self._kick_passes = (
                    self._kick_passes + 1
                    if self.kickup_success_rate >= c.success_fraction
                    else 0
                )
                if (
                    self._kick_passes >= c.required_batches
                    and self.kickup_level < len(c.amplitudes) - 1
                ):
                    self.kickup_level += 1
                    self._kick_passes = 0
                print(
                    f"[kickup curriculum] level={self.kickup_level} "
                    f"success={self.kickup_success_rate:.1%}",
                    flush=True,
                )
                self._kick_episodes = self._kick_successes = 0
        super()._reset_idx(reset_mask)
        self.kickup.reset(reset_mask)
        self._kick_level[reset_mask] = self.kickup_level
        self._kick_success[reset_mask] = False
        self._kick_started[reset_mask] = True
        self.kickup.amplitude[reset_mask] = (
            c.amplitudes[self.kickup_level] if c.enabled else 1.0
        )

    def kickup_in_region(self):
        """Override with the learned/verified region membership predicate.

        Default is a provisional geometric gate, NOT a certified ROA test.
        Positive pitch rotates the rear above the front in this task.
        """
        c = self.cfg.kickup
        gravity = self.projected_gravity
        pitch = torch.atan2(gravity[:, 0], -gravity[:, 2])
        supported = (self.contact_forces[:, self.feet_indices[:2], 2] > 1).all(dim=1)
        return (
            (pitch >= c.handoff_pitch_rad)
            & (pitch <= c.handoff_max_pitch_rad)
            & (self.base_ang_vel.norm(dim=1) <= c.handoff_max_speed)
            & supported
        )

    def _compute_torques(self):
        # Preserve the actor's target tensor, including its observation/history
        # contract. Replace targets only for the duration of this PD evaluation.
        self.kickup.advance(
            self.dt / self.cfg.control.decimation, self.kickup_in_region()
        )
        actor_target = self.dof_pos_target
        active = ~(self.kickup.handed_off | self.kickup.failed)
        target = self.kickup.targets() - self.default_dof_pos
        c = self.cfg.kickup_curriculum
        if c.enabled:
            # Bounded residual lets PPO learn corrections during the kick.
            target = target + c.residual_limit_rad * actor_target.tanh()
        self.dof_pos_target = torch.where(active[:, None], target, actor_target)
        try:
            return super()._compute_torques()
        finally:
            self.dof_pos_target = actor_target

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        self.terminated |= self.kickup.failed
        c = self.cfg.kickup_curriculum
        if c.enabled:
            final = self._kick_level == len(c.amplitudes) - 1
            reached = self._reward_kickup_progress() >= 1
            self._kick_success |= reached & ~self.terminated & ~final
            # End an intermediate kick at its target, without claiming ROA entry.
            self.timed_out |= self._kick_success & ~final

    def _reward_kickup_progress(self):
        c = self.cfg.kickup_curriculum
        if not c.enabled:
            return torch.zeros(self.num_envs, device=self.device)
        pitch = torch.atan2(self.projected_gravity[:, 0], -self.projected_gravity[:, 2])
        goals = pitch.new_tensor(c.pitch_degrees).deg2rad()[self._kick_level]
        support = (self.contact_forces[:, self.feet_indices[:2], 2] > 1).all(dim=1)
        return (pitch / goals).clamp(0, 1) * support * ~self.terminated

    def training_state_dict(self):
        state = super().training_state_dict()
        state["kickup_curriculum"] = dict(
            level=self.kickup_level,
            episodes=self._kick_episodes,
            successes=self._kick_successes,
            passes=self._kick_passes,
            success_rate=self.kickup_success_rate,
        )
        return state

    @torch.inference_mode()
    def load_training_state_dict(self, state):
        super().load_training_state_dict(state)
        saved = state.get("kickup_curriculum")
        if saved is None:
            return
        level = int(saved["level"])
        if not 0 <= level < len(self.cfg.kickup_curriculum.amplitudes):
            raise ValueError("checkpoint kickup level is out of range")
        self.kickup_level = level
        self._kick_episodes = int(saved["episodes"])
        self._kick_successes = int(saved["successes"])
        self._kick_passes = int(saved["passes"])
        self.kickup_success_rate = float(saved["success_rate"])
        self._kick_started.zero_()
        self.reset()
