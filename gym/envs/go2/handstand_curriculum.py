"""Normal-gravity reset curriculum, advanced by completed-episode hold success."""

import math

import torch

from gym.envs.go2.go2_config import (
    FOOT_RADIUS,
    LEG_LENGTH,
    STANDING_THIGH,
    GO2_HIP_X,
    GO2_FOOT_Y,
)
from gym.utils.torch_quat import quat_apply, quat_from_euler_xyz


def foot_positions(joints):
    """Body-frame foot centres for (n, 4, 3) joint angles, FL FR RL RR.

    FK for Go2's equal-link legs, with zero hip abduction.
    """
    thigh, calf = joints[..., 1], joints[..., 2]
    feet = torch.zeros_like(joints)
    feet[..., 0] = joints.new_tensor(GO2_HIP_X) - LEG_LENGTH * (
        thigh.sin() + (thigh + calf).sin()
    )
    feet[..., 1] = joints.new_tensor(GO2_FOOT_Y)
    feet[..., 2] = -LEG_LENGTH * (thigh.cos() + (thigh + calf).cos())
    return feet


class HandstandCurriculum:
    def _init_handstand_curriculum(self):
        c = self.cfg.handstand_curriculum
        n = len(c.pitch_degrees)
        if not (n == len(c.tilt_noise_degrees) == len(c.angular_velocity_noise)):
            raise ValueError("handstand curriculum level arrays must have equal length")
        if not 0 <= c.initial_level < n:
            raise ValueError("handstand initial_level is outside the curriculum")
        if not (
            0 <= c.replay_fraction < 1
            and 0 <= c.standing_fraction < 1
            and c.replay_fraction + c.standing_fraction < 1
        ):
            raise ValueError(
                "handstand reset mixture fractions must sum to less than 1"
            )
        if c.min_episodes < 1 or c.required_batches < 1 or c.hold_seconds <= 0:
            raise ValueError(
                "handstand curriculum requires positive episode/hold settings"
            )
        if not 0 < c.success_fraction <= 1 or not 0 < c.hold_score <= 1:
            raise ValueError("handstand success and hold thresholds must be in (0, 1]")
        self.curriculum_level = c.initial_level
        self.curriculum_success_rate = 0.0
        self._curriculum_episodes = 0
        self._curriculum_successes = 0
        self._curriculum_passes = 0
        self._hold_time = torch.zeros(self.num_envs, device=self.device)
        self._best_hold = torch.zeros_like(self._hold_time)
        self._reset_stage = torch.zeros_like(self._hold_time)
        self._reset_level = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._primary_reset = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._episode_active = torch.zeros_like(self._primary_reset)
        self._curriculum_pitch = torch.tensor(c.pitch_degrees, device=self.device)
        self._curriculum_tilt = torch.tensor(c.tilt_noise_degrees, device=self.device)
        self._curriculum_velocity = torch.tensor(
            c.angular_velocity_noise, device=self.device
        )

    def _record_handstand_episodes(self, reset_mask):
        c = self.cfg.handstand_curriculum
        if not c.enabled:
            return
        # Manual resets and episodes drawn from easier replay cannot promote a level.
        eligible = (
            reset_mask
            & self._episode_active
            & self._primary_reset
            & (self._reset_level == self.curriculum_level)
            & (self.terminated | self.timed_out)
        )
        self._curriculum_episodes += int(eligible.sum().item())
        self._curriculum_successes += int(
            (eligible & (self._best_hold >= c.hold_seconds)).sum().item()
        )
        if self._curriculum_episodes < c.min_episodes:
            return
        self.curriculum_success_rate = (
            self._curriculum_successes / self._curriculum_episodes
        )
        passed = self.curriculum_success_rate >= c.success_fraction
        self._curriculum_passes = self._curriculum_passes + 1 if passed else 0
        print(
            f"[handstand curriculum] level={self.curriculum_level} "
            f"hold_success={self.curriculum_success_rate:.1%} "
            f"episodes={self._curriculum_episodes}",
            flush=True,
        )
        self._curriculum_episodes = self._curriculum_successes = 0
        if (
            self._curriculum_passes >= c.required_batches
            and self.curriculum_level < len(c.pitch_degrees) - 1
        ):
            self.curriculum_level += 1
            self._curriculum_passes = 0
            print(
                f"[handstand curriculum] advancing to level {self.curriculum_level}",
                flush=True,
            )

    def _update_handstand_hold(self):
        c = self.cfg.handstand_curriculum
        # Called after termination checks; a body collision cannot count as a hold.
        held = (
            (self._reward_handstand_balance() >= c.hold_score)
            & ~self.terminated
            & ~self.timed_out
        )
        self._hold_time[:] = torch.where(held, self._hold_time + self.dt, 0.0)
        self._best_hold[:] = torch.maximum(self._best_hold, self._hold_time)

    def reset_to_handstand(self, reset_mask):
        """Sample a pose, rotate its leg geometry, and place front feet at terrain."""
        c = self.cfg.handstand_curriculum
        ids = reset_mask.nonzero(as_tuple=False).flatten()
        count = ids.numel()
        if count == 0:
            return
        level = self.curriculum_level
        levels = torch.full((count,), level, device=self.device, dtype=torch.long)
        primary = torch.ones(count, device=self.device, dtype=torch.bool)
        if c.enabled and level > 0:
            draw = torch.rand(count, device=self.device)
            replay = draw < c.replay_fraction
            levels[replay] = torch.randint(level, (count,), device=self.device)[replay]
            standing = (draw >= c.replay_fraction) & (
                draw < c.replay_fraction + c.standing_fraction
            )
            # -1 is a complete standing reset, never used as a promotion sample.
            levels[standing] = -1
            primary = ~(replay | standing)
        if not c.enabled:
            levels.fill_(-1)
            primary.fill_(False)
        table_ids = levels.clamp(min=0)
        pitch = torch.deg2rad(self._curriculum_pitch[table_ids])
        tilt = torch.deg2rad(self._curriculum_tilt[table_ids])
        standing = levels == -1
        pitch[standing] = 0
        tilt[standing] = 0
        # A zero-pitch final level is a true four-foot standing reset.
        standing |= pitch == 0
        nominal_pitch = pitch.clone()
        pitch += (2 * torch.rand(count, device=self.device) - 1) * tilt
        roll = (2 * torch.rand(count, device=self.device) - 1) * tilt
        yaw = (2 * torch.rand(count, device=self.device) - 1) * math.pi
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        fraction = (nominal_pitch / (math.pi / 2)).clamp(0, 1)
        knee = (
            -2 * STANDING_THIGH * (1 - fraction)
            + self.cfg.reward_settings.kick_knee_target * fraction
        )
        joints = torch.zeros(count, 4, 3, device=self.device)
        joints[:, :, 2] = knee[:, None]
        # Front feet stay below the front hips in world coordinates.
        joints[:, :2, 1] = (-nominal_pitch - knee / 2)[:, None]
        rear_thigh = (
            STANDING_THIGH * (1 - fraction) + (math.pi / 2 - knee / 2) * fraction
        )
        joints[:, 2:, 1] = rear_thigh[:, None]
        feet = foot_positions(joints)
        world_feet = quat_apply(quat[:, None, :].expand(-1, 4, -1), feet)
        root = torch.zeros(count, 13, device=self.device)
        root[:, 2] = FOOT_RADIUS + c.ground_clearance - world_feet[:, :, 2].amin(dim=1)
        root[:, 3:7] = quat
        velocity_noise = self._curriculum_velocity[table_ids].clone()
        velocity_noise[standing] = 0
        root[:, 10:13] = (
            2 * torch.rand(count, 3, device=self.device) - 1
        ) * velocity_noise[:, None]
        self.root_states[ids] = root
        self.dof_pos[ids[:, None], self.actuated_dof_indices] = joints.flatten(1)
        self.dof_vel[ids] = 0
        stages = torch.where(nominal_pitch >= math.radians(85), 2.0, 1.0)
        stages[standing] = 0
        self._reset_stage[ids] = stages
        self._reset_level[ids] = levels
        self._primary_reset[ids] = primary

    def training_state_dict(self):
        """Persist curriculum progress, not transient robot states."""
        return {
            "handstand_curriculum": {
                "level": self.curriculum_level,
                "episodes": self._curriculum_episodes,
                "successes": self._curriculum_successes,
                "passes": self._curriculum_passes,
                "success_rate": self.curriculum_success_rate,
            }
        }

    @torch.inference_mode()
    def load_training_state_dict(self, state):
        saved = state.get("handstand_curriculum")
        if saved is None:
            return
        level = int(saved["level"])
        if not 0 <= level < len(self.cfg.handstand_curriculum.pitch_degrees):
            raise ValueError("checkpoint handstand curriculum level is out of range")
        self.curriculum_level = level
        self._curriculum_episodes = int(saved["episodes"])
        self._curriculum_successes = int(saved["successes"])
        self._curriculum_passes = int(saved["passes"])
        self.curriculum_success_rate = float(saved["success_rate"])
        self._episode_active.zero_()
        self.reset()
