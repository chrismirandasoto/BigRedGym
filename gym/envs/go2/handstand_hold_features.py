"""What the handstand hold policy sees; shared by training and deployment.

One observation *frame* is the first 42 values of the capture controller's
observation, exactly as go2_deploy's handstand implementation builds it:

    0:3    gravity direction in the body frame
    3:6    body angular velocity, rad/s
    6:18   joint angle - nominal posture, rad
    18:30  joint velocity / joint speed limit
    30:42  (applied joint target - nominal posture) / 2

The policy input is the current frame followed by `history` earlier frames,
most recent first (go2_deploy's `hs_history`), each centred on the held
handstand and scaled to order one. No time, phase or COM estimate is used.

The policy output is in action units around the held handstand:

    joint offset from nominal = (hold_target - nominal) + ACTION_SCALE * clip(a)

which the deploy stack then divides by 2 and passes through the same joint
limit clip and slew limit as training.
"""

import torch

FRAME = 42
ACTION_SCALE = 0.5  # rad per policy unit
ACTION_CLIP = 1.5  # policy units, so at most 0.75 rad from the held posture
FEATURE_CLAMP = 10.0


def frame_center_and_scale(nominal, hold_target, speed, target_rotation):
    """Per-channel centre and scale of one frame, as float32 tensors (42,)."""
    nominal = torch.as_tensor(nominal, dtype=torch.float64)
    hold = torch.as_tensor(hold_target, dtype=torch.float64)
    speed = torch.as_tensor(speed, dtype=torch.float64)
    rotation = torch.as_tensor(target_rotation, dtype=torch.float64)
    gravity = rotation.T @ torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64)
    offset = hold - nominal
    center = torch.cat(
        [gravity, torch.zeros(3), offset, torch.zeros(12), offset / 2]
    ).float()
    scale = torch.cat(
        [
            torch.full((3,), 2.0),  # gravity: 0.5 per 15 degrees of tilt
            torch.full((3,), 0.5),  # rad/s
            torch.full((12,), 2.0),  # rad
            speed / 5.0,  # back to rad/s, then / 5
            torch.full((12,), 4.0),  # target offsets were halved
        ]
    ).float()
    return center, scale


def features(frames, center, scale):
    """frames (..., K + 1, 42), current first -> policy input (..., 42 (K + 1))."""
    out = ((frames - center) * scale).clamp(-FEATURE_CLAMP, FEATURE_CLAMP)
    return out.flatten(-2)


def joint_offset(actions, nominal, hold_target):
    """Policy output -> joint offset from the nominal posture, rad."""
    hold = torch.as_tensor(hold_target, dtype=actions.dtype, device=actions.device)
    nominal = torch.as_tensor(nominal, dtype=actions.dtype, device=actions.device)
    return (hold - nominal) + ACTION_SCALE * actions.clamp(-ACTION_CLIP, ACTION_CLIP)
