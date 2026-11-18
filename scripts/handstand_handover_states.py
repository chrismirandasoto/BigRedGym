"""Record robot states at the 2.2 s hand-over, for go2handstandhold training.

Runs the released capture controller (CPU, MuJoCo) through the kick from the
released start scatter, with and without sensor/actuator errors, and keeps
the state at the control step where its balance feedback would take over.
The GPU task resets its episodes from these states.

    python scripts/handstand_handover_states.py

Writes resources/handstand/handover_states.npz (about 2 MB). It is tracked
in git, so MoLab has it; after regenerating, commit it with `git add -f`.

Needs artifacts/ from the handstand release (see docs/handstand_reproduce.md).
"""

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

TRAJECTORY_SHA256 = "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92"
HISTORY = 8  # frames stored; the task uses the most recent ones
_cache = {}


def record(job):
    """One kick; the state at the hand-over, or None if it failed before."""
    from experiments.handstand_robustness.core import Environment
    from experiments.handstand_robustness.randomization import (
        PROFILES,
        RandomizedEnvironment,
        draw,
        scaled,
    )
    from experiments.handstand_robustness.randomized_training import load_policy
    from gym.envs.go2.handstand_replay import build_model, load_trajectory

    torch.set_num_threads(1)
    if "policy" not in _cache:
        _cache["policy"] = load_policy(job["checkpoint"])[0].eval()
        _cache["trajectory"] = load_trajectory(
            Path(job["trajectory"]), TRAJECTORY_SHA256, build_model()
        )
    policy, trajectory = _cache["policy"], _cache["trajectory"]
    ranges = PROFILES["release"] | scaled(PROFILES["deploy"], job["scale"])
    if job["scale"] == 0:
        ranges = PROFILES["release"]
    try:
        env = RandomizedEnvironment(
            build_model(), trajectory, job["seed"], draw(ranges, job["seed"]), 400
        )
    except ValueError:
        return None  # start state rejected
    frames = []
    for _ in range(job["handover"]):
        frames.append(Environment.observation(env)[:42].copy())
        with torch.no_grad():
            action = policy(torch.tensor(env.observation())[None])[0].numpy()
        _, _, done = env.step(action)
        if done or not env.safe:
            return None
    state = np.empty(mujoco.mj_stateSize(env.m, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.m, env.d, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    history = np.zeros((HISTORY, 42), dtype=np.float32)
    recent = frames[::-1][:HISTORY]  # most recent first, as go2_deploy keeps them
    history[: len(recent)] = recent
    return dict(
        qpos=env.d.qpos.copy(),
        qvel=env.d.qvel.copy(),
        target=env.target.copy(),
        history=history,
        scale=job["scale"],
        seed=job["seed"],
        holding=float(env.hold),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/capture_ppo_latest.pt")
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("resources/handstand/handover_states.npz")
    )
    parser.add_argument("--per-scale", type=int, default=1024)
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.0, 0.03, 0.1, 0.3],
        help="sizes of the 'deploy' error profile; 0 = released start scatter only",
    )
    parser.add_argument("--handover", type=int, default=110)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    from experiments.handstand_robustness.core import sha256
    from gym.envs.go2.handstand_replay import (
        TARGET_HEIGHT,
        TARGET_ROTATION,
        Replay,
        build_model,
        load_trajectory,
    )

    jobs = [
        dict(
            checkpoint=str(args.checkpoint),
            trajectory=str(args.trajectory),
            seed=args.seed + 10000 * i + j,
            scale=scale,
            handover=args.handover,
        )
        for i, scale in enumerate(args.scales)
        for j in range(args.per_scale)
    ]
    started = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        rows = pool.map(record, jobs, chunksize=4)
    kept = [r for r in rows if r is not None]
    for scale in args.scales:
        tried = sum(j["scale"] == scale for j in jobs)
        got = sum(r["scale"] == scale for r in kept)
        print(f"scale {scale:g}: {got}/{tried} reached the hand-over safely")

    model = build_model()
    trajectory = load_trajectory(args.trajectory, TRAJECTORY_SHA256, model)
    robot = Replay(model, trajectory)
    stack = {key: np.array([r[key] for r in kept]) for key in kept[0]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema_version=np.array(1),
        mujoco_version=np.array(mujoco.__version__),
        handover_step=np.array(args.handover),
        qpos=stack["qpos"].astype(np.float32),
        qvel=stack["qvel"].astype(np.float32),
        target=stack["target"].astype(np.float32),
        history=stack["history"],
        scale=stack["scale"].astype(np.float32),
        seed=stack["seed"].astype(np.int64),
        holding=stack["holding"].astype(np.float32),
        # Everything the GPU task needs about the controller and the gate.
        joint_names=np.array(robot.names),
        nominal=trajectory["nominal"],
        hold_target=trajectory["nominal"] + 2 * trajectory["actions"][-1],
        joint_bounds=robot.bounds,
        speed=robot.speed,
        effort=robot.effort,
        target_rotation=TARGET_ROTATION,
        target_com_height=np.array(TARGET_HEIGHT),
        checkpoint_sha256=np.array(sha256(args.checkpoint)),
        trajectory_sha256=np.array(TRAJECTORY_SHA256),
    )
    print(
        f"Wrote {len(kept)} states to {args.output} "
        f"in {time.perf_counter() - started:.0f} s"
    )


if __name__ == "__main__":
    main()
