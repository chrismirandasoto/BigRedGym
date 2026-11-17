"""The numbers the handstand needs besides the policy network.

handstand_assets.json sits next to policy.pt in a bundle; deploy_meta.json
records its SHA256 so the two cannot drift apart. Fields:

    nominal          12  posture that joint observations/actions are relative to
    initial_target   12  joint target at the start of the maneuver
    initial_joints   12  joint angles of the training start state
    standing_target  12  joint target that holds initial_joints at rest
    joint_bounds  12x2  [lower, upper], rad
    speed            12  joint speed limits, rad/s
    kinematics           rigid-body tree (go2_deploy/handstand/kinematics.py)
    height_offset        correction to the root height the feet imply, m
    imu_offset        3  IMU position in the base frame, m
"""

import hashlib
import json
import os

ASSETS_FILE = "handstand_assets.json"
ASSETS_HASH_KEY = "handstand_assets_sha256"
REQUIRED = (
    "nominal",
    "initial_target",
    "initial_joints",
    "standing_target",
    "joint_bounds",
    "speed",
    "kinematics",
    "height_offset",
    "imu_offset",
)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_assets(cfg):
    """Assets from the deployed bundle, or from GO2_HANDSTAND_ASSETS."""
    bundle_dir = getattr(cfg, "bundle_dir", None)
    if bundle_dir:
        path = os.path.join(bundle_dir, ASSETS_FILE)
        expected = (cfg.bundle_meta or {}).get(ASSETS_HASH_KEY)
        if expected is None or sha256(path) != expected:
            raise ValueError(f"{path} does not match the hash in deploy_meta.json")
    elif os.environ.get("GO2_HANDSTAND_ASSETS"):
        path = os.environ["GO2_HANDSTAND_ASSETS"]
    else:
        raise ValueError(
            "the handstand needs a bundle from scripts/export_handstand_bundle.py "
            "(deploy with --bundle)"
        )
    with open(path) as f:
        assets = json.load(f)
    missing = [key for key in REQUIRED if key not in assets]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    return assets
