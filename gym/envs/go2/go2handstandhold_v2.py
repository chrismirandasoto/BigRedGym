"""go2handstandhold, scored the way the 67/100 evaluation scores it.

v1 (go2handstandhold.py) learned to stay up under sensor and actuator error,
but scored 0/100: inside the strict gate 91% of the time, it broke it about
20 times per run, on front-foot slip and load lasting a few milliseconds.
Two differences from the evaluator (gym/envs/go2/handstand_replay.py) let that
through, and v2 removes both:

- The gate is checked after every 2 ms physics step, not once per 20 ms
  control step. A control step counts as held only if all its physics steps
  pass, and front slip and load are scored by their worst physics step.
- The evaluator needs 5 unbroken seconds. `unbroken_hold` pays for the length
  of the current unbroken hold, so a break costs the hold built up so far.

The evaluator's safety flag is also matched: a joint past its speed limit or
range ends the episode, as it ends the evaluated trial.

The actor's observations and actions are v1's, so the deploy module and
exporter are unchanged and v1's actor is the starting point. The critic also
sees the hold's progress.
"""

import torch

from gym.envs.go2.go2handstandhold import Go2HandstandHold


class Go2HandstandHoldV2(Go2HandstandHold):
    def _init_buffers(self):
        super()._init_buffers()
        n = self.num_envs
        self.hold_seconds = torch.zeros(n, device=self.device)
        self._gate_all = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._unsafe = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._slip_peak = torch.zeros(n, device=self.device)
        self._front_load_low = torch.zeros(n, device=self.device)
        self._start_substeps()

    def _start_substeps(self):
        self._substep_gate = torch.ones_like(self._gate_all)
        self._substep_unsafe = torch.zeros_like(self._unsafe)
        self._substep_slip = torch.zeros_like(self._slip_peak)
        self._substep_load = torch.full_like(self._front_load_low, float("inf"))

    # ── Every physics step ───────────────────────────────────────────────

    def _pre_decimation_step(self):
        super()._pre_decimation_step()
        self._start_substeps()

    def _post_physics_step(self):
        super()._post_physics_step()
        self._update_physics_summaries()
        unsafe = self._joint_unsafe()
        self._substep_unsafe |= unsafe
        self._substep_gate &= self._strict_hold() & ~unsafe
        self._substep_slip = torch.maximum(self._substep_slip, self._front_slip)
        self._substep_load = torch.minimum(
            self._substep_load, self._loads[:, :2].amin(dim=1)
        )

    def _joint_unsafe(self):
        """The evaluator's joint part of `safe`: speed and range, 1e-3 slack."""
        q = self.dof_pos.index_select(1, self.actuated_dof_indices)
        dq = self.dof_vel.index_select(1, self.actuated_dof_indices)
        return (
            (dq.abs() / self._speed > 1.001)
            | (q < self._lower - 0.001)
            | (q > self._upper + 0.001)
        ).any(dim=1)

    # ── Every control step ───────────────────────────────────────────────

    def _post_decimation_step(self):
        super()._post_decimation_step()
        finite = ~self._diverged
        self._gate_all = self._substep_gate & finite
        self._unsafe = self._substep_unsafe & finite
        self._slip_peak = torch.where(finite, self._substep_slip, 0.0)
        self._front_load_low = torch.where(finite, self._substep_load, 0.0)
        self.hold_seconds = torch.where(
            self._gate_all, self.hold_seconds + self.dt, torch.zeros_like(self.hold_seconds)
        )

    def _check_terminations_and_timeouts(self):
        super()._check_terminations_and_timeouts()
        self.terminated |= self._unsafe

    def _reset_idx(self, reset_mask):
        super()._reset_idx(reset_mask)
        if reset_mask.any():
            self.hold_seconds[reset_mask] = 0.0
            self._gate_all[reset_mask] = False
            self._unsafe[reset_mask] = False

    @property
    def hold_progress(self):
        """For the critic: the unbroken hold so far, and whether it is held."""
        horizon = self.cfg.reward_settings.unbroken_horizon
        return torch.stack(
            [(self.hold_seconds / horizon).clamp(max=1.5), self._gate_all.float()], 1
        )

    # ── Rewards ──────────────────────────────────────────────────────────

    def _reward_strict_hold(self):
        return self._gate_all.float()

    def _reward_unbroken_hold(self):
        horizon = self.cfg.reward_settings.unbroken_horizon
        return self._gate_all * (self.hold_seconds / horizon).clamp(max=1.0)

    def _reward_front_slip(self):
        # 1 at the gate's limit, growing quadratically, capped at 9.
        limit = self.cfg.reward_settings.slip_limit
        return -(self._slip_peak / limit).clamp(max=3.0).square()

    def _reward_front_support(self):
        need = self.cfg.reward_settings.min_front_load
        return (self._front_load_low / need).clamp(0, 1)
