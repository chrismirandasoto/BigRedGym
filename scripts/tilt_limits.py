"""Map handstand recovery over initial tilt and tilt rate.

Each robot starts in its own grid cell: the top pose rotated rigidly about the
front-foot line, with the matching angular velocity about that line.

Run with python -m scripts.tilt_limits.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from gym.envs.go2.go2balance import Go2Balance
from gym.envs.go2.go2balance_config import Go2BalanceCfg, Go2BalanceRunnerCfg
from gym.utils.helpers import set_seed
from gym.utils.original_cfg import checkpoint_configs
from gym.utils.robot_metrics import configure_contact_cpg
from gym.utils.task_registry import select_backend, task_registry

OUTCOMES = ("recovered", "no hold", "fell")
# Categorical slots that stay distinguishable in every pairing.
COLORS = ("#2a78d6", "#1baf7a", "#eb6834")
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def evaluation_configs(args):
    cfg, train_cfg = Go2BalanceCfg(), Go2BalanceRunnerCfg()
    if args.checkpoint is not None and not args.current_cfg:
        cfg, train_cfg = checkpoint_configs(args.checkpoint, args.config_task)
        if not hasattr(cfg, "tilt_test"):
            cfg.tilt_test = Go2BalanceCfg().tilt_test
    configure_contact_cpg(args, cfg, train_cfg)
    angles, rates = args.grid
    t = cfg.tilt_test
    t.enabled = True
    t.sampling = "grid"
    t.grid = [angles, rates]
    t.angle_degrees = list(args.angle_deg)
    t.angular_velocity = list(args.rate)
    c = cfg.handstand_curriculum
    levels = len(c.pitch_degrees)
    # The tilt grid is the only disturbance: no reset noise, replay or pushes.
    c.enabled = True
    c.replay_fraction = 0
    c.standing_fraction = 0
    c.min_episodes = 10**12
    c.tilt_noise_degrees = [0.0] * levels
    c.angular_velocity_noise = [0.0] * levels
    c.hold_seconds = args.hold_seconds
    cfg.balance_perturbations.enabled = False
    cfg.push_robots.toggle = False
    cfg.init_state.reset_mode = "reset_to_handstand"
    cfg.env.num_envs = angles * rates
    cfg.env.episode_length_s = max(cfg.env.episode_length_s, args.seconds + 1)
    cfg.seed = args.seed
    train_cfg.runner.device = args.device
    return cfg, train_cfg


def coasting_line(env):
    """Tilt rate from which a rigid, uncontrolled robot coasts up to balance.

    Locked joints and planted feet make the robot a physical pendulum about the
    front-foot line. Energy conservation gives the rate that just reaches the
    upright equilibrium: rate = -2 sqrt(m g l / I) sin(phi / 2), where phi is
    the centre-of-mass angle from vertical. MuJoCo CPU only.
    """
    backend = env._backend
    if not hasattr(backend, "_datas"):
        return None
    m, d = backend._model_for_env(0), backend._datas[0]
    feet = backend._canonical_to_native_body_np[env.feet_indices[:2].cpu().numpy()]
    base = int(m.jnt_bodyid[0])
    axis = d.xpos[feet[0]] - d.xpos[feet[1]]
    axis /= np.linalg.norm(axis)
    pivot = d.xpos[feet].mean(axis=0)
    arm = d.subtree_com[base] - pivot
    arm -= axis * (arm @ axis)
    length = np.linalg.norm(arm)
    phi = math.asin(np.clip(np.cross([0.0, 0.0, 1.0], arm / length) @ axis, -1, 1))
    # Free-joint velocity: world linear, then body-frame angular.
    velocity = np.zeros(m.nv)
    velocity[:3] = np.cross(axis, d.xpos[base] - pivot)
    velocity[3:6] = d.xmat[base].reshape(3, 3).T @ axis
    mass = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, mass)
    inertia = velocity @ mass @ velocity
    gravity = float(np.linalg.norm(m.opt.gravity))
    growth = math.sqrt(m.body_subtreemass[base] * gravity * length / inertia)
    # Environment 0's tilt is known, so its CoM angle fixes the nominal offset.
    offset = phi - float(env.initial_tilt[0])
    return dict(
        com_distance_m=float(length),
        inertia_kg_m2=float(inertia),
        growth_rate_per_s=growth,
        com_offset_deg=math.degrees(offset),
    )


def cell_edges(values):
    if len(values) == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    half = (values[1] - values[0]) / 2
    return np.r_[values - half, values[-1] + half]


def plot(angles_deg, rates, outcome, line, label, hold, seconds, path):
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": "#c3c2b7",
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    x, y = cell_edges(angles_deg), cell_edges(rates)
    ax.pcolormesh(
        x,
        y,
        outcome.T,
        cmap=ListedColormap(COLORS),
        vmin=-0.5,
        vmax=2.5,
        edgecolors=SURFACE,
        linewidth=1.5,
    )
    handles = [
        Patch(color=color, label=f"{name} ({int((outcome == i).sum())})")
        for i, (name, color) in enumerate(zip(OUTCOMES, COLORS))
    ]
    if line is not None:
        fine = np.linspace(x[0], x[-1], 200)
        offset = math.radians(line["com_offset_deg"])
        coast = -2 * line["growth_rate_per_s"] * np.sin((np.radians(fine) + offset) / 2)
        inside = (coast >= y[0]) & (coast <= y[-1])
        handles += ax.plot(
            fine[inside],
            coast[inside],
            color=INK,
            linewidth=2,
            label="rigid coast to balance (no control)",
        )
    ax.set(
        xlim=(x[0], x[-1]),
        ylim=(y[0], y[-1]),
        xlabel="Initial tilt from vertical [deg]   (− toward standing, + over-rotated)",
        ylabel="Initial tilt rate [rad/s]",
    )
    ax.set_title(
        f"Handstand recovery by initial tilt and tilt rate\n{label}",
        loc="left",
        color=INK,
    )
    ax.legend(
        handles=handles,
        title=f"{hold:g} s hold within {seconds:g} s, no fall",
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        frameon=False,
        labelcolor=MUTED,
        alignment="left",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def recovered_span(values, recovered):
    """Extent of recovered cells along one grid line through the origin."""
    hit = values[recovered]
    return dict(
        recovered=int(recovered.sum()),
        cells=len(values),
        low=float(hit.min()) if hit.size else None,
        high=float(hit.max()) if hit.size else None,
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, help="Go2 balance or reachability checkpoint"
    )
    parser.add_argument("--config_task", choices=["go2balance", "go2reachability"])
    parser.add_argument(
        "--current_cfg",
        action="store_true",
        help="Use current configs instead of the saved run",
    )
    parser.add_argument("--cpg", action="store_true")
    parser.add_argument(
        "--contact_modes", nargs=4, choices=["phase", "stance", "swing", "ignore"]
    )
    parser.add_argument(
        "--angle_deg",
        nargs=2,
        type=float,
        default=[-30.0, 30.0],
        metavar=("LOW", "HIGH"),
        help="Tilt range; positive is over-rotation",
    )
    parser.add_argument(
        "--rate",
        nargs=2,
        type=float,
        default=[-3.0, 3.0],
        metavar=("LOW", "HIGH"),
        help="Tilt-rate range, rad/s",
    )
    parser.add_argument(
        "--grid",
        nargs=2,
        type=int,
        default=[21, 21],
        metavar=("ANGLES", "RATES"),
        help="Grid cells; one robot per cell",
    )
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--hold_seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("output_tilt_limits"))
    args = parser.parse_args()
    if min(args.grid) < 1:
        parser.error("--grid needs positive cell counts")
    if args.angle_deg[0] > args.angle_deg[1] or args.rate[0] > args.rate[1]:
        parser.error("ranges must be LOW HIGH")
    if not 0 < args.hold_seconds < args.seconds or not math.isfinite(args.seconds):
        parser.error("--seconds must exceed a positive --hold_seconds")
    torch.set_num_threads(1)
    set_seed(args.seed)
    cfg, train_cfg = evaluation_configs(args)
    task_registry.convert_frequencies_to_params(cfg, train_cfg)
    backend = select_backend(cfg, args.device)
    try:
        env = Go2Balance(cfg, args.device, True, backend)
        runner = None
        label = "Fixed top-posture PD baseline (no trained policy)"
        if args.checkpoint:
            train_cfg.logging.enable_local_saving = False
            train_cfg.runner.resume = False
            task_registry.set_log_dir_name(train_cfg)
            runner = task_registry.make_alg_runner(env, train_cfg)
            runner.load(str(args.checkpoint), load_optimizer=False)
            runner.switch_to_eval()
            label = f"Policy: {args.checkpoint.parent.name}/{args.checkpoint.name}"
        n = env.num_envs
        env._reset_idx(torch.ones(n, dtype=torch.bool, device=env.device))
        tilt = env.initial_tilt.cpu().numpy().copy()
        rate = env.initial_tilt_rate.cpu().numpy().copy()
        angles, rates = args.grid
        angle_axis = tilt.reshape(angles, rates)[:, 0]
        rate_axis = rate.reshape(angles, rates)[0]
        line = coasting_line(env)
        failed = torch.zeros(n, dtype=torch.bool)
        best = torch.zeros(n)
        first_failure = torch.full((n,), float("nan"))
        first_hold = torch.full((n,), float("nan"))
        for step in range(math.ceil(args.seconds / env.dt)):
            if runner:
                runner.set_actions(
                    runner.actor_cfg["actions"],
                    runner.get_inference_actions(),
                    runner.actor_cfg["disable_actions"],
                )
            else:
                env.dof_pos_target.zero_()
            env.step()
            elapsed = (step + 1) * env.dt
            fell = env.terminated.cpu() & ~failed
            first_failure[fell] = elapsed
            failed |= fell
            best = torch.maximum(best, torch.where(failed, 0.0, env._best_hold.cpu()))
            held = (best >= args.hold_seconds) & torch.isnan(first_hold)
            first_hold[held] = elapsed
        # A hold followed by a fall inside the horizon is still a fall.
        recovered = (best >= args.hold_seconds) & ~failed
        outcome = torch.where(recovered, 0, torch.where(failed, 2, 1)).numpy()
        args.output.mkdir(parents=True, exist_ok=True)
        records = [
            dict(
                env_id=i,
                tilt_deg=math.degrees(tilt[i]),
                tilt_rate_rad_s=float(rate[i]),
                outcome=OUTCOMES[outcome[i]],
                recovered=int(recovered[i]),
                fell=int(failed[i]),
                best_hold_s=float(best[i]),
                first_hold_s=float(first_hold[i]),
                first_failure_s=float(first_failure[i]),
            )
            for i in range(n)
        ]
        with (args.output / "samples.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0])
            writer.writeheader()
            writer.writerows(records)
        grid = outcome.reshape(angles, rates)
        angles_deg = np.degrees(angle_axis)
        plot(
            angles_deg,
            rate_axis,
            grid,
            line,
            label,
            args.hold_seconds,
            args.seconds,
            args.output / "tilt_limits.png",
        )
        still = int(np.abs(rate_axis).argmin())
        upright = int(np.abs(angle_axis).argmin())
        report = dict(
            controller=label,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            if args.checkpoint
            else None,
            seed=args.seed,
            horizon_seconds=args.seconds,
            hold_seconds=args.hold_seconds,
            angle_degrees=list(args.angle_deg),
            angular_velocity_rad_s=list(args.rate),
            grid=list(args.grid),
            counts={name: int((outcome == i).sum()) for i, name in enumerate(OUTCOMES)},
            recovered_fraction=float(recovered.float().mean()),
            tilt_deg_at_rate=dict(
                rate_rad_s=float(rate_axis[still]),
                **recovered_span(angles_deg, grid[:, still] == 0),
            ),
            rate_rad_s_at_tilt=dict(
                tilt_deg=float(angles_deg[upright]),
                **recovered_span(rate_axis, grid[upright] == 0),
            ),
            rigid_pendulum=line,
            interpretation=(
                "One deterministic rollout per cell from a rigid tilt about the "
                "front-foot line; positive tilt and rate are over-rotation. "
                "Recovered means a strict hold with no termination in the "
                "horizon. Spans are the extremes of recovered cells on one grid "
                "line and may contain failures between them."
            ),
        )
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"Recovered {report['counts']['recovered']}/{n}, "
            f"fell {report['counts']['fell']}; results: {args.output}"
        )
    finally:
        backend.close()


if __name__ == "__main__":
    main()
