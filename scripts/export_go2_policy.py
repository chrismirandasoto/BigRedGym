"""Export a trained Go2 run as a policy bundle for onboard deployment.

Run on the PC (it builds the training env to load the checkpoint):

    uv run --frozen python scripts/export_go2_policy.py --name trot_v1
    uv run --frozen python scripts/export_go2_policy.py --name trot_v1 \
        --run_name <run>

Writes logs/policies/<name>/ (policy.pt, deploy_meta.json).
The scaling recorded is DeployConfig's at export time, which inherits the
training config: export right after training, before editing scales.
"""

import argparse
import os
import subprocess

from go2_deploy.deploy_config import DeployConfig
from go2_deploy.policy_bundle import export_bundle
from go2_deploy.rl_controller import setup_actor

DEFAULT_OUT_ROOT = os.path.join("logs", "policies")


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="Bundle name")
    parser.add_argument("--run_name", help="Training run in logs/ (default: latest)")
    parser.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    cfg = DeployConfig()
    actor = setup_actor(args.run_name, cfg)
    out_dir = os.path.join(args.out_root, args.name)
    provenance = {"run_name": args.run_name or "latest", "git_commit": git_commit()}
    meta_path = export_bundle(actor, cfg, out_dir, provenance)
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
