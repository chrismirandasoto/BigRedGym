"""Sensor/actuator robustness of the handstand capture controller; simulation only.

    compare   deploy stack vs training policy on the released start states
    sweep     one error at a time, at several sizes
    evaluate  a checkpoint (or its deploy bundle) under a named error profile
    train     PPO fine-tuning under randomized errors, into a new directory

Every mode writes JSON under --output and never modifies its input
checkpoint. Trials run in parallel worker processes.
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.handstand_robustness.randomization import (  # noqa: E402
    PROFILES,
    RandomizedEnvironment,
    draw,
    scaled,
)
from gym.envs.go2.handstand_replay import build_model, load_trajectory  # noqa: E402

TRAJECTORY_SHA256 = "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92"
RELEASE_SEED = 301000  # the released evaluation used seeds 301001-301100
# Largest size tried for each single error in `sweep`; see randomization.py.
SWEEP = {
    "gyro_bias": [0.002, 0.005, 0.01, 0.02],
    "gyro_noise": [0.002, 0.005, 0.01, 0.02],
    "accel_bias": [0.01, 0.03, 0.1, 0.3],
    "accel_noise": [0.02, 0.05, 0.1, 0.3],
    "orientation_bias": [0.002, 0.005, 0.01, 0.02],
    "encoder_noise": [0.0002, 0.0005, 0.001, 0.002],
    "encoder_velocity_noise": [0.02, 0.05, 0.1, 0.2],
    "height_error": [0.001, 0.003, 0.005, 0.01],
    "zero_offset": [0.001, 0.003, 0.005, 0.01],
    "kp_scale": [0.02, 0.05, 0.1, 0.2],
    "kd_scale": [0.02, 0.05, 0.1, 0.2],
    "strength": [[0.98, 1.0], [0.95, 1.0], [0.9, 1.0], [0.8, 1.0]],
    "command_delay_substeps": [[1, 1], [2, 2], [3, 3], [5, 5]],
    "joint_damping": [0.01, 0.02, 0.05, 0.1],
    "friction": [0.05, 0.1, 0.2, 0.3],
}
_cache = {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def trajectory(path):
    if "trajectory" not in _cache:
        _cache["trajectory"] = load_trajectory(
            Path(path), TRAJECTORY_SHA256, build_model()
        )
    return _cache["trajectory"]


def policy(path):
    from experiments.handstand_robustness.randomized_training import load_policy

    if path not in _cache:
        _cache[path] = load_policy(Path(path))[0].eval()
    return _cache[path]


def trial(job):
    """One full standing-start maneuver; returns a small result row."""
    torch.set_num_threads(1)
    t = trajectory(job["trajectory"])
    seed = job["seed"]
    condition = draw(job["ranges"], seed)
    row = dict(seed=seed, label=job.get("label"), condition=condition)
    try:
        if job.get("bundle"):
            from experiments.handstand_robustness.deploy_sil import (
                DeployedEnvironment,
                make_controller,
                run_deployed,
            )

            env = DeployedEnvironment(
                build_model(),
                t,
                make_controller(job["bundle"]),
                seed,
                condition,
                horizon=400,
                accelerometer=job.get("accelerometer", "origin"),
            )
            result = run_deployed(
                env, start=job.get("start", "direct"), height=job.get("height", "feet")
            )
            row.update(
                {
                    key: result[key]
                    for key in (
                        "emergency_stop",
                        "started",
                        "prepare_seconds",
                        "observation_error",
                        "target_error",
                        "gyro_bias_estimate",
                        "accel_bias_estimate",
                    )
                }
            )
        else:
            from experiments.handstand_robustness.deploy_sil import run_native

            env = RandomizedEnvironment(build_model(), t, seed, condition, horizon=400)
            result = run_native(env, policy(job["checkpoint"]))
    except ValueError as error:
        # A drawn start state can be rejected (penetration, joint limit).
        return row | dict(invalid=str(error), complete_success=False)
    return row | dict(
        complete_success=bool(result["complete_success"]),
        strict_hold_seconds=float(result["strict_hold_seconds"]),
        capture_success=bool(result["capture_success"]),
        takeoff=bool(result["certified_takeoff"]),
        failure=result["failure"],
        control_steps=int(result["control_steps"]),
        observer_velocity_error=float(env.max_observer_velocity_error),
        observer_height_error=float(env.max_observer_height_error),
    )


def run_trials(jobs, workers):
    started = time.perf_counter()
    if workers <= 1:
        rows = [trial(job) for job in jobs]
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            rows = pool.map(trial, jobs, chunksize=1)
    return rows, time.perf_counter() - started


def wilson(successes, trials):
    """95% Wilson interval for a success fraction."""
    if not trials:
        return [0.0, 1.0]
    z, p = 1.959964, successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (
        z
        * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2))
        / (1 + z * z / trials)
    )
    return [float(centre - half), float(centre + half)]


def summary(rows):
    valid = [r for r in rows if "invalid" not in r]
    successes = sum(r["complete_success"] for r in valid)
    return dict(
        successes=successes,
        trials=len(valid),
        invalid_starts=len(rows) - len(valid),
        wilson95=wilson(successes, len(valid)),
        captures=sum(r["capture_success"] for r in valid),
        failures={
            kind: sum(r["failure"] == kind for r in valid)
            for kind in sorted({r["failure"] for r in valid if r["failure"]})
        },
    )


def base_job(args, **extra):
    return (
        dict(trajectory=str(args.trajectory), checkpoint=str(args.checkpoint)) | extra
    )


def seeds(args):
    return range(args.evaluation_seed + 1, args.evaluation_seed + 1 + args.trials)


def compare(args):
    """Same start states through the training policy and the deploy stack."""
    ranges = PROFILES["release"]
    variants = {
        "native": {},
        "deploy_exact": dict(bundle=str(args.bundle), height="training"),
        "deploy": dict(bundle=str(args.bundle)),
        "deploy_imu_mount": dict(bundle=str(args.bundle), accelerometer="imu"),
        "deploy_operator_sequence": dict(bundle=str(args.bundle), start="prepare"),
        "deploy_operator_sequence_imu_mount": dict(
            bundle=str(args.bundle), start="prepare", accelerometer="imu"
        ),
    }
    result = {}
    for name, extra in variants.items():
        jobs = [
            base_job(args, seed=s, ranges=ranges, label=name, **extra)
            for s in seeds(args)
        ]
        rows, seconds = run_trials(jobs, args.workers)
        result[name] = summary(rows) | dict(seconds=seconds, rows=rows)
        print(name, result[name]["successes"], "/", result[name]["trials"], flush=True)
    native = {r["seed"]: r["complete_success"] for r in result["native"]["rows"]}
    for name in variants:
        if name != "native":
            result[name]["same_outcome_as_native"] = sum(
                native[r["seed"]] == r["complete_success"] for r in result[name]["rows"]
            )
    return result


def deployed_options(args):
    """Extra job fields that route a trial through the deploy bundle."""
    if not args.deployed:
        return {}
    return dict(
        bundle=str(args.bundle), start=args.start, accelerometer=args.accelerometer
    )


def sweep(args):
    """One error at a time, on top of the released start-state scatter."""
    reset = PROFILES["release"]
    names = args.factors or list(SWEEP)
    extra = deployed_options(args)
    jobs = [
        base_job(args, seed=s, ranges=reset, label="none", **extra) for s in seeds(args)
    ]
    for name in names:
        for level, value in enumerate(SWEEP[name]):
            jobs += [
                base_job(
                    args,
                    seed=s,
                    ranges=reset | {name: value},
                    label=f"{name}:{level}",
                    **extra,
                )
                for s in seeds(args)
            ]
    rows, seconds = run_trials(jobs, args.workers)
    table = {"none": summary([r for r in rows if r["label"] == "none"])}
    for name in names:
        table[name] = [
            dict(size=value)
            | summary([r for r in rows if r["label"] == f"{name}:{level}"])
            for level, value in enumerate(SWEEP[name])
        ]
        print(
            name,
            [
                f"{row['size']}: {row['successes']}/{row['trials']}"
                for row in table[name]
            ],
            flush=True,
        )
    return dict(table=table, seconds=seconds, rows=rows)


def evaluate(args):
    result = {}
    for name in args.profiles:
        for factor in args.scales:
            ranges = scaled(PROFILES[name], factor)
            extra = deployed_options(args)
            label = f"{name} x{factor:g}"
            jobs = [
                base_job(args, seed=s, ranges=ranges, label=label, **extra)
                for s in seeds(args)
            ]
            rows, seconds = run_trials(jobs, args.workers)
            result[label] = summary(rows) | dict(
                ranges=ranges, seconds=seconds, rows=rows
            )
            print(
                label,
                result[label]["successes"],
                "/",
                result[label]["trials"],
                flush=True,
            )
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("mode", choices=["compare", "sweep", "evaluate", "train"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/capture_ppo_latest.pt")
    )
    parser.add_argument(
        "--bundle", type=Path, default=Path("logs/policies/handstand_capture_v1")
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--evaluation-seed", type=int, default=RELEASE_SEED)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--factors", nargs="*", choices=list(SWEEP))
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["sensor", "actuator", "both"],
        choices=list(PROFILES),
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--deployed", action="store_true", help="evaluate through the deploy bundle"
    )
    parser.add_argument("--start", choices=["direct", "prepare"], default="prepare")
    parser.add_argument("--accelerometer", choices=["origin", "imu"], default="imu")
    # train
    parser.add_argument("--profile", default="both", choices=list(PROFILES))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument(
        "--episodes", type=int, default=64, help="rollout episodes per update"
    )
    parser.add_argument("--validation-trials", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7110)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    from experiments.handstand_robustness.core import sha256

    if args.mode == "train":
        from experiments.handstand_robustness.randomized_training import train

        result = train(args)
    else:
        result = dict(compare=compare, sweep=sweep, evaluate=evaluate)[args.mode](args)
    write(
        args.output / f"{args.mode}.json",
        dict(
            result=result,
            configuration={
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            checkpoint_sha256=sha256(args.checkpoint),
            reference_sha256=sha256(args.trajectory),
            scope="simulation only; error sizes are assumed, not measured on a robot",
        ),
    )
    print(args.output / f"{args.mode}.json")


if __name__ == "__main__":
    main()
