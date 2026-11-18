"""Released kick, then a trained hold policy (go2handstandhold), as one module.

    step <  handover   the released capture controller (CaptureDeployPolicy)
    step >= handover   the hold policy trained in gym/envs/go2/go2handstandhold.py

Input: the 47-value capture observation followed by `history` past frames of
42 (go2_deploy's hs_history). Output: 12 values in capture action units
(radians / 2 from the nominal posture), like the released controller, so the
deploy stack's clip and slew limit apply unchanged.

Built on the PC by scripts/export_handstand_bundle.py --hold-checkpoint and
saved as TorchScript; the robot needs neither this file nor gym.
"""

import torch
from torch import nn

from go2_deploy.handstand.policy import CaptureDeployPolicy
from gym.envs.go2.handstand_hold_features import (
    ACTION_CLIP,
    ACTION_SCALE,
    FEATURE_CLAMP,
    FRAME,
)


def mlp_from_state(state, prefix="NN."):
    """The actor network of a QGym checkpoint (Linear + ELU layers)."""
    indices = sorted(
        {int(k[len(prefix) :].split(".")[0]) for k in state if k.startswith(prefix)}
    )
    layers = []
    for n, index in enumerate(indices):
        weight = state[f"{prefix}{index}.weight"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(state[f"{prefix}{index}.bias"])
        layers.append(linear)
        if n < len(indices) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class CaptureHoldDeployPolicy(nn.Module):
    num_actions = 12

    def __init__(
        self, capture_state, hold_network, center, scale, nominal, hold_target
    ):
        super().__init__()
        self.capture = CaptureDeployPolicy(capture_state)
        self.hold = hold_network
        inputs = hold_network[0].in_features
        if inputs % FRAME or inputs < 2 * FRAME:
            raise ValueError(f"hold policy input {inputs} is not 42 x (history + 1)")
        self.history = inputs // FRAME - 1
        self.num_obs = 47 + FRAME * self.history
        self.handover = self.capture.active_steps
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        offset = torch.as_tensor(hold_target, dtype=torch.float32) - torch.as_tensor(
            nominal, dtype=torch.float32
        )
        self.register_buffer("hold_offset", offset)

    def hold_action(self, obs):
        """The hold policy's raw output for one observation."""
        frames = torch.cat(
            [obs[:FRAME].unsqueeze(0), obs[47 : self.num_obs].view(self.history, FRAME)]
        )
        x = ((frames - self.center) * self.scale).clamp(-FEATURE_CLAMP, FEATURE_CLAMP)
        return self.hold(x.flatten())

    def forward(self, obs):
        kick = self.capture(obs[:47])
        action = self.hold_action(obs).clamp(-ACTION_CLIP, ACTION_CLIP)
        hold = ((self.hold_offset + ACTION_SCALE * action) / 2).clamp(-1, 1)
        step = (obs[42] * 400).round()
        return torch.where(step >= self.handover, hold, kick)
