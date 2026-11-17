"""Released capture controller plus a learned, history-aware correction.

    action = released_controller(obs[:47]) + bound * tanh(network(features))

The released controller is kept inside, frozen. The network's last layer
starts at zero, so an untrained policy is the released controller exactly.
The network also sees the previous `history` control periods of the 42
sensor/command channels, which is what lets it tell a delayed or weaker
actuator from a nominal one; the released controller sees one period only.

Observation layout (47 + 42 * history values):

    0:47              the capture observation (go2_deploy handstand_capture.py)
    47 + 42*j : ...   the first 42 channels as they were j + 1 periods ago

History frames from before the maneuver began are ignored.

The critic is used in training only and may see simulator truth.
"""

import torch
from torch import nn

from experiments.handstand_robustness.capture import CapturePolicy

KIND = "capture_history_v1"
FRAME = 42  # gravity 3, gyro 3, joint angle 12, joint velocity 12, target 12
FEATURE_CLAMP = 20.0
FEATURE_SCALE = 5.0
LOG_STD_START, LOG_STD_MIN, LOG_STD_MAX = -5.8, -7.0, -1.0


def _network(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden),
        nn.ELU(),
        nn.Linear(hidden, hidden),
        nn.ELU(),
        nn.Linear(hidden, outputs),
    )


class WideBalanceCapture(CapturePolicy):
    """The released controller with a wider limit on its balance feedback.

    After the hand-over at 2.2 s the released controller adds
    clip(balance_weight @ error, +/- 0.05) to its prior action. This class
    clips at `late_bound` instead. Whenever the feedback stays inside
    +/- 0.05, as on the nominal maneuver, the two are identical.
    """

    def __init__(self, actions, centers, balance, active_steps, bound, late_bound):
        super().__init__(actions, centers, balance, active_steps, bound)
        self.register_buffer("late_bound", torch.tensor(float(late_bound)))

    def forward(self, obs, feedback=True):
        prefix = super(CapturePolicy, self).forward(obs, feedback)
        k = (obs[..., 42] * 400).round().long().clamp(0, len(self.actions) - 1)
        features = torch.cat([obs[..., :30], obs[..., 43:47]], -1)
        centers = torch.cat([self.centers[k, :30], self.centers[k, 43:47]], -1)
        correction = ((features - centers) @ self.balance_weight.T).clamp(
            -self.late_bound, self.late_bound
        )
        balance = (self.actions[k] + correction).clamp(-1, 1)
        late = (obs[..., 42] * 400).round() >= self.active_steps
        return torch.where(late[..., None], balance, prefix)


class HistoryResidualPolicy(nn.Module):
    checkpoint_kind = KIND

    def __init__(self, base, history=4, bound=0.2, hidden=256, privileged=0):
        super().__init__()
        if not isinstance(base, CapturePolicy):
            raise ValueError("the base must be a capture policy")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.history = int(history)
        self.bound = float(bound)
        self.hidden = int(hidden)
        self.privileged = int(privileged)
        self.num_obs = 47 + FRAME * self.history
        self.num_actions = 12
        features = 46 + 2 + FRAME * self.history
        self.actor = _network(features, hidden, 12)
        self.critic = _network(features + self.privileged, hidden, 1)
        # exp(-5.8) = 0.003: about 1 mrad of joint-target noise. The released
        # controller loses most handstands with any more than that.
        self.log_std = nn.Parameter(torch.full((12,), LOG_STD_START))
        # Untrained, the correction is exactly zero.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def features(self, obs):
        base = self.base
        steps = len(base.actions)
        current = obs[..., :47]
        step = (current[..., 42] * 400).round()
        centre = base.centers[step.clamp(0, steps - 1).long()]
        error = torch.cat(
            [
                current[..., :42] - centre[..., :42],
                current[..., 43:47] - centre[..., 43:47],
            ],
            -1,
        )
        parts = [
            (error / base.scale).clamp(-FEATURE_CLAMP, FEATURE_CLAMP) / FEATURE_SCALE,
            (step / steps).clamp(0, 1.5).unsqueeze(-1),
            (step >= base.active_steps).to(obs.dtype).unsqueeze(-1),
        ]
        for j in range(self.history):
            frame = obs[..., 47 + FRAME * j : 47 + FRAME * (j + 1)]
            then = step - (j + 1)
            centre = base.centers[then.clamp(0, steps - 1).long()]
            error = (frame - centre[..., :FRAME]) / base.scale[:FRAME]
            error = error.clamp(-FEATURE_CLAMP, FEATURE_CLAMP) / FEATURE_SCALE
            # Frames from before the maneuver carry no information about it.
            parts.append(error * (then >= 0).to(obs.dtype).unsqueeze(-1))
        return torch.cat(parts, -1)

    def released(self, obs):
        """What the released controller alone would command."""
        return self.base(obs[..., :47])

    def forward(self, obs):
        correction = self.bound * torch.tanh(self.actor(self.features(obs)))
        return (self.released(obs) + correction).clamp(-1, 1)

    def distribution(self, obs):
        return torch.distributions.Normal(
            self.actor(self.features(obs)),
            self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp(),
        )

    def action(self, obs, latent):
        return (self.released(obs) + self.bound * torch.tanh(latent)).clamp(-1, 1)

    def value(self, obs, privileged):
        inputs = torch.cat([self.features(obs), privileged], -1)
        return self.critic(inputs).squeeze(-1)

    def settings(self):
        late = getattr(self.base, "late_bound", None)
        return dict(
            history=self.history,
            bound=self.bound,
            hidden=self.hidden,
            privileged=self.privileged,
            late_bound=None if late is None else float(late),
        )


def from_base_state(state, late_bound=None, **settings):
    """A fresh history policy around a capture policy's state dict.

    `late_bound` widens the released controller's balance-feedback limit
    (WideBalanceCapture); None keeps the released 0.05.
    """
    state = {k: v for k, v in state.items() if k != "late_bound"}
    balance = {
        key: state["balance_" + key].detach().numpy()
        for key in ["weight", "bias", "lower", "upper"]
    }
    arguments = (
        state["actions"],
        state["centers"],
        balance,
        int(state["active_steps"]),
        float(state["bound"]),
    )
    if late_bound is None:
        base = CapturePolicy(*arguments)
    else:
        base = WideBalanceCapture(*arguments, late_bound)
    base.load_state_dict(state, strict=False)
    return HistoryResidualPolicy(base.eval(), **settings)


def load_history_policy(path):
    """Returns (policy, saved dict) from a capture_history_v1 checkpoint."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("kind") != KIND:
        raise ValueError(f"{path} is not a {KIND} checkpoint")
    state = saved["policy"]
    base_state = {
        key[len("base.") :]: value
        for key, value in state.items()
        if key.startswith("base.")
    }
    policy = from_base_state(base_state, **saved["settings"])
    policy.load_state_dict(state)
    return policy.eval(), saved
