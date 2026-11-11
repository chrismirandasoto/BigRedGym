"""Empirical backward reachable-set curriculum around front-foot balance."""

import mujoco
import numpy as np
import torch

from gym.envs.go2.go2balance import BalanceDiagnostics, Go2Balance
from gym.envs.go2.reachability_physics import ContactSampler


class Go2Reachability(Go2Balance):
    def _init_buffers(self):
        super()._init_buffers()
        if self._tilt_test is not None:
            # Rear-joint sampling below rewrites the root velocity.
            raise ValueError("the tilt test runs on go2balance, not go2reachability")
        self.reset_accepted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.reset_attempts = torch.zeros(self.num_envs, device=self.device)
        self.initial_joints = torch.zeros_like(self.dof_pos)
        self.initial_velocity = torch.zeros_like(self.dof_vel)
        self.velocity_envelope = torch.zeros(self.num_envs, device=self.device)
        self._sampler = ContactSampler(self)
        self._contact_loss_time = torch.zeros(self.num_envs, device=self.device)
        self._contact_loss_started = torch.zeros_like(self.reset_accepted)
        self._contact_recovered = torch.zeros_like(self.reset_accepted)
        self.training_diagnostics = ReachabilityDiagnostics(self.device)

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        self._contact_loss_time[reset_mask] = 0
        self._contact_loss_started[reset_mask] = False
        self._contact_recovered[reset_mask] = False

    def _post_decimation_step(self):
        super()._post_decimation_step()
        c = self.cfg.balance_failure
        lost = (
            self.contact_forces[:, self.feet_indices[:2], 2] < c.min_normal_force
        ).any(dim=1)
        active = (
            self.episode_length_buf * self.dt >= c.contact_grace_seconds
        ) & ~self.terminated
        previous = self._contact_loss_time.clone()
        self._contact_loss_started[:] = active & lost & (previous == 0)
        self._contact_recovered[:] = (
            active & ~lost & (previous > 0) & (previous < c.both_front_airborne_seconds)
        )
        self._contact_loss_time[:] = torch.where(active & lost, previous + self.dt, 0.0)

    def update_training_diagnostics(self):
        super().update_training_diagnostics()
        d = self.training_diagnostics
        d.losses += self._contact_loss_started.sum()
        d.recoveries += (self._contact_recovered & ~self.terminated).sum()
        d.total_losses += self._contact_loss_started.sum()
        d.total_recoveries += (self._contact_recovered & ~self.terminated).sum()
        d.best_hold_sum += self._best_hold.mean()

    def record_training_rewards(self, rewards):
        self.training_diagnostics.record_rewards(
            rewards, self.terminated, self.timed_out
        )

    def reset_to_handstand(self, reset_mask):
        super().reset_to_handstand(reset_mask)
        s = self._sampler
        c = self.cfg.reachability
        for i in reset_mask.nonzero().flatten().tolist():
            root = self.root_states[i].cpu().numpy().copy()
            reference = self.dof_pos[i].cpu().numpy().copy()
            level = int(self._reset_level[i].clamp(min=0))
            self.reset_accepted[i] = False
            self.velocity_envelope[i] = 0
            self.reset_attempts[i] = 0
            for attempt in range(c.attempts):
                self.reset_attempts[i] = attempt + 1
                draw = torch.rand(6).numpy()
                target = s.limits[6:, 0] + draw * np.diff(s.limits[6:], axis=1).ravel()
                joints = reference.copy()
                joints[6:] += c.position_fraction[level] * (target - joints[6:])
                s.load(root, joints)
                direction = 2 * torch.rand(6).numpy() - 1
                direction /= max(np.linalg.norm(direction), 1e-10)
                velocity, inertia = s.direction(direction)
                bound = s.bound(joints, direction, inertia)
                amplitude = c.velocity_fraction[level] * float(torch.rand(())) * bound
                velocity *= amplitude
                # Include compensating planted-leg motors in the speed check.
                ratio = np.max(np.abs(velocity[s.v]) / s.speed)
                velocity /= max(1.0, ratio)
                if not s.probe(velocity, joints):
                    continue
                self.dof_pos[i] = torch.as_tensor(joints, device=self.device)
                self.dof_vel[i] = torch.as_tensor(velocity[s.v], device=self.device)
                self.root_states[i, 7:10] = torch.as_tensor(
                    velocity[:3], device=self.device
                )
                # MuJoCo base angular velocity is local; task root uses world frame.
                rotation = np.empty(9)

                mujoco.mju_quat2Mat(rotation, root[[6, 3, 4, 5]])
                self.root_states[i, 10:13] = torch.as_tensor(
                    rotation.reshape(3, 3) @ velocity[3:6], device=self.device
                )
                self.velocity_envelope[i] = bound
                self.reset_accepted[i] = True
                break
            # Rejected proposals fall back to goal replay and cannot promote curriculum.
            self._primary_reset[i] &= self.reset_accepted[i]
            self.initial_joints[i] = self.dof_pos[i]
            self.initial_velocity[i] = self.dof_vel[i]

    def _reward_front_contact_loss(self):
        """Smooth per-foot support deficit; brief losses can recover."""
        loads = self._handstand_loads()[:, :2]
        deficit = (1 - loads / self.cfg.reward_settings.front_support_fraction).clamp(
            0, 1
        )
        return -deficit.square().mean(dim=1)

    def _reward_reachability_stability(self):
        """Bounded angular-speed cost, including while support is lost."""
        speed_squared = self.base_ang_vel.square().sum(dim=1)
        return torch.expm1(
            -speed_squared / self.cfg.reward_settings.recovery_angular_sigma**2
        )


class ReachabilityDiagnostics(BalanceDiagnostics):
    """Rollout recovery events and completed episode returns, before resets."""

    def __init__(self, device):
        self.device = device
        self.episode_returns = None
        self.total_losses = torch.zeros((), device=device)
        self.total_recoveries = torch.zeros_like(self.total_losses)
        super().__init__(device)
        self.names += (
            "contact_loss_events",
            "contact_recoveries",
            "cumulative_recovery_fraction",
            "best_hold_seconds",
            "fall_episode_return",
            "timeout_episode_return",
            "fall_episodes",
            "timeout_episodes",
        )

    def reset(self):
        super().reset()
        self.losses = torch.zeros((), device=self.device)
        self.recoveries = torch.zeros_like(self.losses)
        self.best_hold_sum = torch.zeros_like(self.losses)
        self.return_sums = torch.zeros(2, device=self.device)
        self.episode_counts = torch.zeros(2, device=self.device)

    def update(self, values):
        # Base diagnostic storage contains only the original balance terms.
        self.sums += torch.stack(
            [values[name].float().mean() for name in self.names[: len(self.sums)]]
        )
        self.steps += 1

    def record_rewards(self, rewards, terminated, timed_out):
        if self.episode_returns is None:
            self.episode_returns = torch.zeros_like(rewards)
        self.episode_returns += rewards
        for i, mask in enumerate((terminated, timed_out & ~terminated)):
            self.return_sums[i] += self.episode_returns[mask].sum()
            self.episode_counts[i] += mask.sum()
        self.episode_returns[terminated | timed_out] = 0

    def finish(self):
        super().finish()
        self.contact_loss_events = self.losses.clone()
        self.contact_recoveries = self.recoveries.clone()
        self.cumulative_recovery_fraction = (
            self.total_recoveries / self.total_losses.clamp(min=1)
        )
        self.best_hold_seconds = self.best_hold_sum / max(self.steps, 1)
        self.fall_episode_return, self.timeout_episode_return = (
            self.return_sums / self.episode_counts.clamp(min=1)
        )
        self.fall_episodes, self.timeout_episodes = self.episode_counts.clone()
