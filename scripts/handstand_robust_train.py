"""Train a robust handstand correction on a multi-core machine; simulation only.

    python scripts/handstand_robust_train.py --output output/robust_history_v1

Trains a history-aware correction on top of the released capture controller
(which is only read), through the go2_deploy stack, under sensor and actuator
errors that grow as the policy keeps succeeding. Needs CPU cores, not a GPU:
one worker process per core. Stop with Ctrl+C at any time and continue with
--resume. See docs/go2_handstand_deploy.md, "Server training".

Outputs under --output:
    latest.pt, best.pt, update_NNNNN.pt   checkpoints (capture_history_v1)
    curve.jsonl                           one line per update
    bundle/                               deploy bundle fixing the observation layout
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OMP_NUM_THREADS", "1")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/capture_ppo_latest.pt"),
        help="released capture checkpoint to build on (read only)",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("artifacts/handstand_reference/trajectory.npz"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument(
        "--episodes",
        type=int,
        help="episodes per update (default: 4 per worker, at least 64)",
    )
    parser.add_argument("--seed", type=int, default=1)
    # Policy
    parser.add_argument(
        "--history", type=int, default=4, help="past control periods seen"
    )
    parser.add_argument(
        "--bound",
        type=float,
        default=0.2,
        help="largest correction, in action units (2 rad)",
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument(
        "--late-bound",
        type=float,
        default=0.2,
        help="limit on the released controller's balance feedback (released: 0.05); "
        "identical on the nominal maneuver, slightly more robust off it",
    )
    parser.add_argument(
        "--foot-correction",
        type=float,
        default=0.3,
        help="seconds; planted-foot drift correction in the estimator",
    )
    # PPO
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="starting step size; adapts every update to hold --target-kl",
    )
    parser.add_argument("--entropy", type=float, default=0.002)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatch", type=int, default=2048)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    # Curriculum over the size of the errors (1.0 = the full "deploy" profile)
    parser.add_argument("--initial-scale", type=float, default=0.005)
    parser.add_argument("--scale-step", type=float, default=1.15)
    # Rates are of training episodes, which carry exploration noise; the
    # untrained policy manages about 0.3 there with no errors at all.
    parser.add_argument(
        "--raise-at", type=float, default=0.4, help="success rate to grow the errors"
    )
    parser.add_argument(
        "--lower-at", type=float, default=0.15, help="success rate to shrink them"
    )
    parser.add_argument(
        "--nominal-fraction",
        type=float,
        default=0.125,
        help="episodes kept free of errors",
    )
    # Validation (deterministic policy, fixed seeds)
    parser.add_argument("--validate-every", type=int, default=10)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument(
        "--validation-scales", type=float, nargs="+", default=[0.0, 0.03, 0.1, 0.3, 1.0]
    )
    args = parser.parse_args()
    if args.episodes is None:
        args.episodes = max(64, 4 * args.workers)

    from experiments.handstand_robustness.history_training import train

    result = train(args)
    summary = args.output / "train.json"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
