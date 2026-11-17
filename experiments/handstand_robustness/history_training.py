"""Train a history-aware correction on the released handstand controller.

What is trained is what is deployed: every episode flies the go2_deploy
stack (the framework's control loop, the robot-side estimator with its
standing calibration, the bounded target filter) against the simulator, with
the policy's network swapped for a sampling one. Sensor and actuator errors
are drawn per episode from the "deploy" profile, scaled by a curriculum that
grows while the policy keeps succeeding.

Episodes run in worker processes; the learner is ordinary PPO with a critic
that also sees simulator truth. Simulation only.
"""

import copy
import json
import multiprocessing
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from experiments.handstand_robustness.core import sha256
from experiments.handstand_robustness.history_policy import (
    KIND,
    LOG_STD_MAX,
    LOG_STD_MIN,
    from_base_state,
    load_history_policy,
)
from experiments.handstand_robustness.randomization import PROFILES, draw, scaled
from gym.envs.go2.handstand_replay import (
    KD,
    KP,
    STATE_SPEC,
    Replay,
    build_model,
    load_trajectory,
)

TRAJECTORY_SHA256 = "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92"
PROFILE = "deploy"
HORIZON = 400  # control periods, 8 s
SUCCESS_BONUS = 5.0
PREPARE_SECONDS = 2.5  # episodes start already settled; see settled_state
PREPARE_LIMIT_SECONDS = 8.0
PRIVILEGED = 17
VALIDATION_SEED = 601000  # training uses seeds below 600000
_cache = {}


# ── Worker side ──────────────────────────────────────────────────────────


def _trajectory(path):
    if "trajectory" not in _cache:
        _cache["trajectory"] = load_trajectory(
            Path(path), TRAJECTORY_SHA256, build_model()
        )
    return _cache["trajectory"]


def settled_state(trajectory, standing_target, seconds=6.0):
    """The nominal robot after standing on the bundle's standing target.

    The recorded start state is not at rest (the motors carry no load), and
    settling from it takes about 5 s of simulation. Episodes start from this
    state instead and only stand long enough to calibrate.
    """
    if "settled" not in _cache:
        robot = Replay(build_model(), trajectory)
        model, data = robot.m, robot.d
        target = np.asarray(standing_target, dtype=float)
        for _ in range(round(seconds / model.opt.timestep)):
            velocity = data.qvel[robot.vids]
            torque = np.clip(
                KP * (target - data.qpos[robot.qids]) - KD * velocity,
                -robot.effort,
                robot.effort,
            )
            data.qfrc_applied[:] = 0
            data.qfrc_applied[robot.vids] = torque
            mujoco.mj_step(model, data)
        state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
        mujoco.mj_getState(model, data, state, STATE_SPEC)
        state[0] = 0.0  # simulation time
        _cache["settled"] = state
    return _cache["settled"]


class Sampler:
    """Stands in for the bundle's network inside the deploy controller."""

    def __init__(self, policy, explore):
        self.policy, self.explore = policy, explore
        self.recording = False
        self.rows = []

    def __call__(self, obs):
        with torch.no_grad():
            if self.explore:
                distribution = self.policy.distribution(obs)
                latent = distribution.sample()
                log = float(distribution.log_prob(latent).sum())
                action = self.policy.action(obs, latent)
            else:
                latent, log = torch.zeros(12), 0.0
                action = self.policy(obs)
        if self.recording:
            self.rows.append((obs.clone(), latent, log))
        return action


def privileged(env):
    """Simulator truth for the critic: root motion and the drawn condition."""
    c = env.condition
    rotation = env.d.xmat[env.base].reshape(3, 3)
    return np.r_[
        env.d.qvel[:3],
        env.d.qpos[2],
        rotation.T @ [0, 0, -1],
        c.get("kp_scale", 1.0) - 1,
        c.get("kd_scale", 1.0) - 1,
        c.get("strength", 1.0) - 1,
        c.get("command_delay_substeps", 0) / 5,
        c.get("friction", 1.0) - 1,
        10 * c.get("joint_damping", 0.0),
        100 * np.asarray(c.get("orientation_bias", np.zeros(3))),
        100 * np.abs(c.get("zero_offset", np.zeros(12))).mean(),
    ].astype(np.float32)


def episode(job):
    """One operator sequence and maneuver through the deploy stack."""
    torch.set_num_threads(1)
    from experiments.handstand_robustness.deploy_sil import (
        DeployedEnvironment,
        make_controller,
        published_target,
    )
    from go2_deploy.handstand.maneuver import Stage
    from go2_deploy.state import State

    key = (job["weights"], job["version"])
    if _cache.get("policy_key") != key:
        _cache["policy"] = load_history_policy(job["weights"])[0]
        _cache["policy_key"] = key
    trajectory = _trajectory(job["trajectory"])
    seed = job["seed"]
    condition = draw(job["ranges"], seed)
    torch.manual_seed(seed)

    controller = make_controller(job["bundle"])
    sampler = Sampler(_cache["policy"], job["explore"])
    controller.rl_controller.policy = sampler
    import sys

    custom = sys.modules["go2_deploy.main_controller"].custom_impl
    env = DeployedEnvironment(
        build_model(), trajectory, None, seed, condition, HORIZON, accelerometer="imu"
    )
    maneuver = custom._maneuver(controller)
    mujoco.mj_setState(
        env.m, env.d, settled_state(trajectory, maneuver.standing_target), STATE_SPEC
    )
    mujoco.mj_forward(env.m, env.d)
    env.anchor = env.d.geom_xpos[env.foot_geoms[:2], :2].copy()
    env.controller = controller
    controller._on_lowstate_msg(env.message())
    maneuver.cfg.prepare_seconds = PREPARE_SECONDS
    controller._state = State.CUSTOM_CTRL

    row = dict(seed=seed, scale=job["scale"], started=False, complete_success=False)
    for _ in range(round(PREPARE_LIMIT_SECONDS / 0.02)):
        if maneuver.stage == Stage.RUN or controller._estop_flag or not env.safe:
            break
        if maneuver.stage == Stage.READY:
            custom.on_key(controller, "h")
        controller._control_loop()
        env.stand(published_target(controller))
    if maneuver.stage != Stage.RUN or not env.safe:
        return row  # never became ready: nothing to learn from

    sampler.recording = True
    nominal = trajectory["nominal"]
    truths, rewards = [], []
    for _ in range(HORIZON):
        truth = privileged(env)
        controller._control_loop()
        if controller._estop_flag:
            sampler.rows.pop()
            break
        _, reward, done = env.step((published_target(controller) - nominal) / 2)
        truths.append(truth)
        rewards.append(reward)
        if done:
            break
    result = env.result()
    row.update(
        started=bool(rewards),
        complete_success=bool(result["complete_success"]),
        capture_success=bool(result["capture_success"]),
        strict_hold_seconds=float(result["strict_hold_seconds"]),
        steps=len(rewards),
    )
    if not job["explore"] or not rewards:
        return row
    rewards[-1] += SUCCESS_BONUS * row["complete_success"]
    return row | dict(
        obs=torch.stack([r[0] for r in sampler.rows]),
        latent=torch.stack([r[1] for r in sampler.rows]),
        log=torch.tensor([r[2] for r in sampler.rows]),
        privileged=torch.tensor(np.array(truths)),
        reward=torch.tensor(rewards, dtype=torch.float32),
    )


# ── Learner side ─────────────────────────────────────────────────────────


def _save(path, policy, optimizer, state, args, source_sha):
    contents = dict(
        kind=KIND,
        schema=1,
        policy=policy.state_dict(),
        settings=policy.settings(),
        optimizer=optimizer.state_dict(),
        reference_sha256=TRAJECTORY_SHA256,
        source_checkpoint_sha256=source_sha,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        torch_rng=torch.get_rng_state(),
        **state,
    )
    temporary = Path(str(path) + ".tmp")
    torch.save(contents, temporary)
    os.replace(temporary, path)


def export_training_bundle(checkpoint, args):
    """The deploy bundle that fixes the observation layout and estimator."""
    import subprocess
    import sys

    name = "bundle"
    command = [
        sys.executable,
        "scripts/export_handstand_bundle.py",
        "--name",
        name,
        "--checkpoint",
        str(checkpoint),
        "--trajectory",
        str(args.trajectory),
        "--out_root",
        str(args.output),
        "--foot-correction",
        str(args.foot_correction),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return args.output / name


def _advantages(policy, episode_rows, gamma, lam):
    out = []
    for e in episode_rows:
        with torch.no_grad():
            values = policy.value(e["obs"], e["privileged"])
        values = torch.cat([values, torch.zeros(1)])  # every episode ends
        advantage = torch.zeros(len(e["reward"]))
        gae = 0.0
        for i in reversed(range(len(advantage))):
            delta = e["reward"][i] + gamma * values[i + 1] - values[i]
            gae = delta + gamma * lam * gae
            advantage[i] = gae
        out.append((advantage, advantage + values[:-1]))
    return torch.cat([a for a, _ in out]), torch.cat([r for _, r in out])


def _validate(pool, base, args, version):
    scores = {}
    for scale in args.validation_scales:
        ranges = scaled(PROFILES[PROFILE], scale)
        jobs = [
            base
            | dict(
                version=version,
                seed=VALIDATION_SEED + 1 + i,
                ranges=ranges,
                scale=scale,
                explore=False,
            )
            for i in range(args.validation_episodes)
        ]
        rows = pool.map(episode, jobs, chunksize=1)
        started = [r for r in rows if r["started"]]
        scores[f"{scale:g}"] = dict(
            successes=sum(r["complete_success"] for r in started),
            trials=len(rows),
            not_ready=len(rows) - len(started),
        )
    return scores


def train(args):
    """Returns the training history; writes checkpoints under args.output."""
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "latest.pt"
    weights = args.output / "weights.pt"
    log = args.output / "curve.jsonl"
    torch.manual_seed(args.seed)

    if args.resume and latest.exists():
        policy, saved = load_history_policy(latest)
        state = {k: saved[k] for k in ("update", "scale", "best", "learning_rate")}
        source_sha = saved["source_checkpoint_sha256"]
        torch.set_rng_state(saved["torch_rng"])
    else:
        if latest.exists():
            raise ValueError(f"{args.output} already holds a run; use --resume")
        from experiments.handstand_robustness.randomized_training import load_policy

        base, _ = load_policy(args.checkpoint)
        policy = from_base_state(
            base.state_dict(),
            history=args.history,
            bound=args.bound,
            hidden=args.hidden,
            privileged=PRIVILEGED,
            late_bound=args.late_bound,
        )
        source_sha = sha256(args.checkpoint)
        state = dict(
            update=0,
            scale=args.initial_scale,
            best=None,
            learning_rate=args.learning_rate,
        )
        saved = None
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=state["learning_rate"])
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer"])
    _save(weights, policy, optimizer, state, args, source_sha)
    bundle = args.output / "bundle"
    if not (bundle / "deploy_meta.json").exists():
        bundle = export_training_bundle(weights, args)
    base_job = dict(
        trajectory=str(args.trajectory), weights=str(weights), bundle=str(bundle)
    )
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    history = []
    started = time.perf_counter()

    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        if state["update"] == 0 and not log.exists():
            # The untrained policy (the released controller) on the same seeds.
            row = dict(update=0, scale=state["scale"])
            row["validation"] = _validate(pool, base_job, args, "v0")
            state["best"] = _score(row["validation"], args)
            _save(
                args.output / "update_00000.pt",
                policy,
                optimizer,
                state,
                args,
                source_sha,
            )
            with open(log, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        for update in range(state["update"] + 1, args.updates + 1):
            scale = state["scale"]
            ranges = scaled(PROFILES[PROFILE], scale)
            jobs = [
                base_job
                | dict(
                    version=update,
                    seed=args.seed * 100000 + update * 1000 + i,
                    # Some episodes stay nominal so the released skill is kept.
                    ranges={} if i < args.nominal_fraction * args.episodes else ranges,
                    scale=scale,
                    explore=True,
                )
                for i in range(args.episodes)
            ]
            rows = pool.map(episode, jobs, chunksize=1)
            episodes = [r for r in rows if "obs" in r]
            if not episodes:
                raise ValueError("no episode became ready; check the bundle and ranges")
            nominal_count = int(args.nominal_fraction * args.episodes)
            randomized = [r for r in rows[nominal_count:] if r["started"]]
            success = sum(r["complete_success"] for r in randomized) / max(
                1, len(randomized)
            )

            obs = torch.cat([e["obs"] for e in episodes])
            truth = torch.cat([e["privileged"] for e in episodes])
            latent = torch.cat([e["latent"] for e in episodes])
            old = torch.cat([e["log"] for e in episodes])
            advantage, returns = _advantages(
                policy, episodes, args.gamma, args.gae_lambda
            )
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            before = {k: v.clone() for k, v in policy.state_dict().items()}
            before_optimizer = copy.deepcopy(optimizer.state_dict())
            policy.train()
            stopped = False
            for _ in range(args.epochs):
                for ids in torch.randperm(len(obs)).split(args.minibatch):
                    distribution = policy.distribution(obs[ids])
                    log_ratio = distribution.log_prob(latent[ids]).sum(-1) - old[ids]
                    ratio = log_ratio.exp()
                    if float(((ratio - 1) - log_ratio).mean()) > 1.5 * args.target_kl:
                        stopped = True
                        break
                    actor = -torch.minimum(
                        ratio * advantage[ids], ratio.clamp(0.8, 1.2) * advantage[ids]
                    ).mean()
                    critic = (
                        (policy.value(obs[ids], truth[ids]) - returns[ids])
                        .square()
                        .mean()
                    )
                    entropy = distribution.entropy().sum(-1).mean()
                    loss = actor + 0.5 * critic - args.entropy * entropy
                    optimizer.zero_grad()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    if not bool(torch.isfinite(loss) and torch.isfinite(norm)):
                        raise ValueError("non-finite PPO loss or gradient")
                    optimizer.step()
                if stopped:
                    break
            policy.eval()
            with torch.no_grad():
                log_ratio = policy.distribution(obs).log_prob(latent).sum(-1) - old
                kl = float(((log_ratio.exp() - 1) - log_ratio).mean())
                residual = float((policy(obs) - policy.released(obs)).abs().mean())
            # Keep each update's policy change near the target. A tiny change
            # to this controller can lose the handstand, so an update that
            # moved the policy much too far is undone, not just slowed.
            rolled_back = not np.isfinite(kl) or kl > 4 * args.target_kl
            if rolled_back:
                policy.load_state_dict(before)
                optimizer.load_state_dict(before_optimizer)
                state["learning_rate"] = max(1e-9, state["learning_rate"] / 2)
            elif kl > 2 * args.target_kl:
                state["learning_rate"] = max(1e-9, state["learning_rate"] / 1.5)
            elif kl < 0.5 * args.target_kl:
                state["learning_rate"] = min(1e-3, state["learning_rate"] * 1.5)
            for group in optimizer.param_groups:
                group["lr"] = state["learning_rate"]
            # Curriculum: harder while succeeding, easier when failing.
            if success >= args.raise_at:
                state["scale"] = min(1.0, scale * args.scale_step)
            elif success < args.lower_at:
                state["scale"] = max(args.initial_scale, scale / args.scale_step)
            state["update"] = update

            row = dict(
                update=update,
                scale=scale,
                randomized_success=success,
                nominal_success=float(
                    np.mean(
                        [r["complete_success"] for r in rows[:nominal_count]] or [0]
                    )
                ),
                episodes=len(episodes),
                not_ready=sum(not r["started"] for r in rows),
                transitions=len(obs),
                mean_steps=float(np.mean([e["steps"] for e in episodes])),
                mean_reward=float(
                    np.mean([float(e["reward"].sum()) for e in episodes])
                ),
                approx_kl=kl,
                kl_early_stopped=stopped,
                rolled_back=rolled_back,
                mean_abs_correction=residual,
                noise_std=float(
                    policy.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).exp().mean()
                ),
                learning_rate=state["learning_rate"],
                seconds=time.perf_counter() - started,
            )
            _save(weights, policy, optimizer, state, args, source_sha)
            if update % args.validate_every == 0 or update == args.updates:
                row["validation"] = _validate(pool, base_job, args, f"v{update}")
                score = _score(row["validation"], args)
                _save(
                    args.output / f"update_{update:05d}.pt",
                    policy,
                    optimizer,
                    state,
                    args,
                    source_sha,
                )
                if state["best"] is None or score > state["best"]:
                    state["best"] = score
                    _save(
                        args.output / "best.pt",
                        policy,
                        optimizer,
                        state,
                        args,
                        source_sha,
                    )
            _save(latest, policy, optimizer, state, args, source_sha)
            with open(log, "a") as f:
                f.write(json.dumps(row) + "\n")
            history.append(row)
            print(json.dumps(row), flush=True)
    return dict(
        history=history, source_checkpoint_sha256=source_sha, best=state["best"]
    )


def _score(validation, args):
    """Checkpoint selection: total validation successes, then the hardest scale."""
    hardest = f"{max(args.validation_scales):g}"
    return [
        sum(v["successes"] for v in validation.values()),
        validation[hardest]["successes"],
    ]
