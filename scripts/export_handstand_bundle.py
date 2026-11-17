"""Export a handstand capture checkpoint as a go2_deploy bundle.

Run on the PC (needs MuJoCo and the training code; the robot needs neither):

    python scripts/export_handstand_bundle.py --name handstand_capture_v1

Writes logs/policies/<name>/:
    policy.pt               TorchScript 47 -> 12 controller, tables included
    deploy_meta.json        observation layout, scaling, gains, 50 Hz, hashes
    handstand_assets.json   posture, joint limits, kinematic tree, IMU offset

The export fails unless the exported network reproduces the checkpoint's
actions at every phase step, including past the end of the reference.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CUSTOM_IMPL = "handstand_capture.py"
if os.environ.setdefault("GO2_CUSTOM_IMPL", CUSTOM_IMPL) != CUSTOM_IMPL:
    sys.exit(f"GO2_CUSTOM_IMPL must be {CUSTOM_IMPL} to export a handstand bundle")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from go2_deploy.deploy_config import DeployConfig  # noqa: E402
from go2_deploy.handstand.assets import ASSETS_FILE, ASSETS_HASH_KEY, sha256  # noqa: E402
from go2_deploy.handstand.kinematics import KinematicModel, extract_tree  # noqa: E402
from go2_deploy.handstand.policy import CaptureDeployPolicy  # noqa: E402
from go2_deploy.policy_bundle import export_bundle, load_bundle  # noqa: E402

TRAJECTORY_SHA256 = "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92"
PARITY_TOLERANCE = 1e-6


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def parity_observations(reference, seed=0):
    """Reference observations at every phase step, nominal and perturbed.

    Steps run to 400, past the 320 recorded, where the last one is held.
    """
    rng = np.random.default_rng(seed)
    centers = reference.centers.numpy()
    rows = []
    for step in range(401):
        base = centers[min(step, len(centers) - 1)].copy()
        base[42] = step * 0.02 / 8
        for size in (0.0, 0.01, 0.3):
            row = base + size * rng.normal(size=47)
            row[42] = base[42]
            rows.append(row)
    return torch.tensor(np.array(rows), dtype=torch.float32)


def history_observations(reference, seed=0):
    """As parity_observations, with random past frames appended."""
    current = parity_observations(reference.base, seed)
    generator = torch.Generator().manual_seed(seed)
    frames = 0.2 * torch.randn(
        len(current), 42 * reference.history, generator=generator
    )
    return torch.cat([current, frames], -1)


def largest_difference(reference, policy, observations):
    with torch.no_grad():
        expected = reference(observations)
        actual = torch.stack([policy(row) for row in observations])
    return float((expected - actual).abs().max())


def standing_target(robot, trajectory):
    """Joint target that holds the training start posture at rest.

    Training starts with the joints at the start posture and the target equal
    to it, so the motors carry no load at t = 0 and the body begins to sag.
    A robot cannot wait in that state. This finds the target whose settled
    posture, under the same PD gains, is the training start posture; the
    robot stands on it and switches to the training target at the trigger.
    """
    import mujoco
    from gym.envs.go2.handstand_replay import KD, KP, STATE_SPEC

    model, data = robot.m, robot.d
    wanted = data.qpos[robot.qids].copy()

    def settle(target, steps):
        for _ in range(steps):
            error = target - data.qpos[robot.qids]
            torque = np.clip(
                KP * error - KD * data.qvel[robot.vids], -robot.effort, robot.effort
            )
            data.qfrc_applied[:] = 0
            data.qfrc_applied[robot.vids] = torque
            mujoco.mj_step(model, data)

    target = trajectory["initial_target"].copy()
    for _ in range(40):
        settle(target, 750)
        target += 0.6 * (wanted - data.qpos[robot.qids])
    # Check from the recorded start state, not the state the search ended in.
    mujoco.mj_setState(model, data, trajectory["initial_state"], STATE_SPEC)
    settle(target, 2500)
    error = float(abs(data.qpos[robot.qids] - wanted).max())
    speed = float(abs(data.qvel[robot.vids]).max())
    if error > 1e-3 or speed > 1e-3:
        raise ValueError(
            f"standing target does not hold the start posture ({error:.4f} rad)"
        )
    mujoco.mj_setState(model, data, trajectory["initial_state"], STATE_SPEC)
    mujoco.mj_forward(model, data)
    return target


def build_assets(trajectory_path, foot_correction_seconds=None):
    from gym.envs.go2.handstand_replay import Replay, build_model, load_trajectory

    model = build_model()
    trajectory = load_trajectory(trajectory_path, TRAJECTORY_SHA256, model)
    robot = Replay(model, trajectory)
    tree = extract_tree(model, robot.names, robot.foot_geoms)
    joints = robot.d.qpos[robot.qids].copy()
    KinematicModel(tree)  # fails here if the tree is not a valid Go2
    return dict(
        schema=1,
        joint_order=robot.names,
        nominal=trajectory["nominal"].tolist(),
        initial_target=trajectory["initial_target"].tolist(),
        initial_joints=joints.tolist(),
        standing_target=standing_target(robot, trajectory).tolist(),
        joint_bounds=robot.bounds.tolist(),
        speed=robot.speed.tolist(),
        effort=robot.effort.tolist(),
        kinematics=tree,
        height_offset=0.0,
        foot_correction_seconds=foot_correction_seconds,
        imu_offset=model.body_pos[model.body("imu").id].tolist(),
        trajectory_sha256=TRAJECTORY_SHA256,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="Bundle name")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/capture_ppo_latest.pt")
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    parser.add_argument("--out_root", default=os.path.join("logs", "policies"))
    parser.add_argument(
        "--foot-correction",
        type=float,
        help="seconds; pull the root estimate toward planted feet (default: off, "
        "the purely inertial estimator of training)",
    )
    args = parser.parse_args()

    from experiments.handstand_robustness.history_policy import KIND as HISTORY_KIND
    from experiments.handstand_robustness.history_policy import load_history_policy
    from experiments.handstand_robustness.randomized_training import load_policy

    kind = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get(
        "kind"
    )
    cfg = DeployConfig()
    if kind == HISTORY_KIND:
        # The network takes past observation frames as one more piece.
        reference, saved = load_history_policy(args.checkpoint)
        policy = reference
        cfg.obs_vector = list(cfg.obs_vector) + ["hs_history"]
        cfg.obs_sizes = dict(cfg.obs_sizes) | {"hs_history": 42 * reference.history}
        observations = history_observations(reference)
    else:
        # capture_residual_v1 (released) or capture_robust_v1 (fine-tuned)
        reference, saved = load_policy(args.checkpoint)
        reference.eval()
        policy = CaptureDeployPolicy(reference.state_dict()).eval()
        observations = parity_observations(reference)
    difference = largest_difference(reference, policy, observations)
    if difference > PARITY_TOLERANCE:
        sys.exit(f"deploy policy differs from the checkpoint by {difference:.2e}")

    out_dir = os.path.join(args.out_root, args.name)
    provenance = dict(
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=sha256(args.checkpoint),
        checkpoint_update=saved.get("update"),
        checkpoint_kind=saved.get("kind"),
        trajectory_sha256=TRAJECTORY_SHA256,
        git_commit=git_commit(),
        validation="simulation only; not validated on hardware",
    )
    meta_path = export_bundle(policy, cfg, out_dir, provenance)

    assets_path = os.path.join(out_dir, ASSETS_FILE)
    with open(assets_path, "w") as f:
        json.dump(build_assets(args.trajectory, args.foot_correction), f)
    with open(meta_path) as f:
        meta = json.load(f)
    meta[ASSETS_HASH_KEY] = sha256(assets_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # The traced file, loaded the way the robot loads it, must still agree.
    traced, loaded_cfg = load_bundle(out_dir)
    traced_difference = largest_difference(reference, traced, observations)
    if traced_difference > PARITY_TOLERANCE:
        sys.exit(f"exported policy.pt differs by {traced_difference:.2e}")
    print(f"Wrote {out_dir}")
    print(
        f"  {loaded_cfg.obs_dim()} observations, {loaded_cfg.ctrl_freq} Hz, "
        f"kp {loaded_cfg.kp}, kd {loaded_cfg.kd}, "
        f"smoothing {'on' if loaded_cfg.exp_moving_avg else 'off'}"
    )
    print(
        f"  largest action difference from the checkpoint over "
        f"{len(observations)} observations: {traced_difference:.2e}"
    )


if __name__ == "__main__":
    main()
