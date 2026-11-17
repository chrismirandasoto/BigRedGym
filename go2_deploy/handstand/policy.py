"""The capture controller as one 47 -> 12 module that TorchScript can trace.

Same arithmetic as CapturePolicy (experiments/handstand_robustness/capture.py)
for a single observation, with the phase-dependent tables kept as buffers:

    k = round(obs[42] * 400)                 commands issued so far
    k <  active_steps   prior action + bound * tanh(gains[k] @ scaled error)
    k >= active_steps   prior action + clip(balance_weight @ error, +/- bound)

"error" is the observation minus the reference observation at step k; the
phase channel only selects k. After the last recorded step the final prior
action and reference are held and the balance feedback keeps running.
"""

import torch
from torch import nn


class CaptureDeployPolicy(nn.Module):
    num_obs = 47
    num_actions = 12

    def __init__(self, state):
        super().__init__()
        for name in ("actions", "centers", "scale", "gains", "balance_weight"):
            self.register_buffer(name, state[name].detach().clone().float())
        self.bound = float(state["bound"])
        self.active_steps = int(state["active_steps"])
        self.steps = len(self.actions)
        if (
            self.centers.shape != (self.steps, 47)
            or self.gains.shape != (self.steps, 12, 46)
            or self.balance_weight.shape != (12, 34)
            or self.scale.shape != (46,)
            or not 0 < self.active_steps <= self.steps
        ):
            raise ValueError("not a capture_residual_v1 policy state")

    def forward(self, obs):
        step = (obs[42] * 400).round()
        early = step.clamp(0, self.active_steps - 1).long()
        late = step.clamp(0, self.steps - 1).long()

        centre = self.centers[early]
        error = torch.cat([obs[:42] - centre[:42], obs[43:47] - centre[43:47]])
        error = (error / self.scale).clamp(-20, 20)
        prefix = self.actions[early] + self.bound * torch.tanh(
            self.gains[early] @ error
        )

        centre = self.centers[late]
        error = torch.cat([obs[:30] - centre[:30], obs[43:47] - centre[43:47]])
        correction = (self.balance_weight @ error).clamp(-self.bound, self.bound)
        balance = self.actions[late] + correction

        return torch.where(step >= self.active_steps, balance, prefix).clamp(-1, 1)
