"""Signed tilt and tilt-rate initial conditions about the front-foot support line.

The handstand is treated as an inverted pendulum pivoting on the line through
both front-foot centres. The whole robot is rotated rigidly about that line and
given the matching rigid-body velocity, so the planted feet neither move nor
start with any speed.

Sign: positive is over-rotation, the direction a kick-up travels. The raised
rear end moves toward the robot's back; negative falls back toward standing.
"""

import math

import torch

from gym.utils.torch_quat import (
    normalize,
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
)

SAMPLING_MODES = ("grid", "uniform")


def validate_tilt_test(cfg):
    ranges = (cfg.angle_degrees, cfg.angular_velocity)
    if any(
        len(r) != 2 or not all(math.isfinite(x) for x in r) or r[0] > r[1]
        for r in ranges
    ):
        raise ValueError("tilt test ranges must be finite [low, high] pairs")
    if cfg.sampling not in SAMPLING_MODES:
        raise ValueError(f"tilt test sampling must be one of {SAMPLING_MODES}")
    if len(cfg.grid) != 2 or any(int(n) != n or n < 1 for n in cfg.grid):
        raise ValueError("tilt test grid must be two positive integers")


def sample_tilt(ids, cfg, device):
    """Signed (angle [rad], rate [rad/s]) for each environment id.

    "grid" ties a cell to the environment id, so a robot always restarts from
    its own cell; angle varies along rows and rate along columns. "uniform"
    draws independently on every reset.
    """
    low = torch.tensor(
        [math.radians(cfg.angle_degrees[0]), cfg.angular_velocity[0]], device=device
    )
    high = torch.tensor(
        [math.radians(cfg.angle_degrees[1]), cfg.angular_velocity[1]], device=device
    )
    if cfg.sampling == "uniform":
        fraction = torch.rand(len(ids), 2, device=device)
    else:
        angles, rates = cfg.grid
        cell = ids % (angles * rates)
        index = torch.stack((cell // rates, cell % rates), dim=1)
        steps = torch.tensor([angles - 1, rates - 1], device=device).clamp(min=1)
        # A single-point axis sits at the middle of its range.
        fraction = torch.where(
            torch.tensor([angles, rates], device=device) > 1, index / steps, 0.5
        )
    state = low + fraction * (high - low)
    return state[:, 0], state[:, 1]


def tilt_about_front_feet(root, feet, angle, rate):
    """Rotate root states rigidly about the front-foot line.

    root: (n, 13) position, xyzw quaternion, world linear and angular velocity.
    feet: (n, 4, 3) body-frame foot centres, front feet first (FL, FR).
    """
    quat = root[:, 3:7]
    front = quat_apply(quat[:, None, :].expand(-1, 2, -1), feet[:, :2])
    # FL - FR is the body's left axis; a right-handed turn about it pitches over.
    axis = normalize(front[:, 0] - front[:, 1])
    pivot = front.mean(dim=1)
    turn = quat_from_angle_axis(angle, axis)
    arm = quat_apply(turn, -pivot)
    omega = rate[:, None] * axis
    tilted = root.clone()
    tilted[:, :3] = root[:, :3] + pivot + arm
    tilted[:, 3:7] = quat_mul(turn, quat)
    # The support line is the instantaneous centre of rotation.
    tilted[:, 7:10] = root[:, 7:10] + torch.cross(omega, arm, dim=1)
    tilted[:, 10:13] = root[:, 10:13] + omega
    return tilted
