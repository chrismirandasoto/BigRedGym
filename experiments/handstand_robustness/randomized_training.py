"""PPO fine-tuning of the capture controller under randomized errors.

Starts from a capture_residual_v1 checkpoint and never writes to it. The
action law is unchanged, so the first update's policy is the baseline exactly;
the only structural change is that the balance feedback used after 2.2 s
becomes trainable alongside the per-step gains used before it. Episodes are
whole maneuvers (up to 8 s) from standing, each under its own drawn
sensor/actuator condition, collected in parallel worker processes.

Checkpoints are kind "capture_robust_v1" and export through
scripts/export_handstand_bundle.py like the baseline.
"""

import copy
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.handstand_robustness.capture import CapturePolicy
from experiments.handstand_robustness.core import sha256
from experiments.handstand_robustness.randomization import (
    PROFILES,
    RandomizedEnvironment,
    draw,
    scaled,
)
from gym.envs.go2.handstand_replay import build_model, load_trajectory

KIND = "capture_robust_v1"
TRAJECTORY_SHA256 = "21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92"
HORIZON = 400
SUCCESS_BONUS = 2.0
VALIDATION_SEED = 401000  # disjoint from training seeds and the 301xxx test set
_cache = {}


class RobustCapturePolicy(CapturePolicy):
    """CapturePolicy whose late balance feedback is a parameter, with a
    stochastic form covering the whole maneuver for PPO."""

    checkpoint_kind = KIND

    def __init__(self, actions, centers, balance, active_steps=110, bound=0.05):
        super().__init__(actions, centers, balance, active_steps, bound)
        weight = self.balance_weight.clone()
        del self._buffers["balance_weight"]
        self.balance_weight = nn.Parameter(weight)

    def _late(self, obs):
        k = (obs[..., 42] * 400).round().long().clamp(0, len(self.actions) - 1)
        features = torch.cat([obs[..., :30], obs[..., 43:47]], -1)
        centers = torch.cat([self.centers[k, :30], self.centers[k, 43:47]], -1)
        return k, features - centers

    def is_late(self, obs):
        return (obs[..., 42] * 400).round() >= self.active_steps

    def latent_mean(self, obs):
        """Feedback in units of the action bound, before saturation."""
        early = self.latent(obs)
        _, error = self._late(obs)
        late = (error @ self.balance_weight.T) / self.bound
        return torch.where(self.is_late(obs)[..., None], late, early)

    def distribution(self, obs):
        return torch.distributions.Normal(
            self.latent_mean(obs), self.log_std.clamp(-5, -1).exp()
        )

    def action(self, obs, latent):
        early_k, _ = self.inputs(obs)
        late_k, _ = self._late(obs)
        early = self.actions[early_k] + self.bound * latent.tanh()
        late = self.actions[late_k] + self.bound * latent.clamp(-1, 1)
        return torch.where(self.is_late(obs)[..., None], late, early).clamp(-1, 1)

    def value(self, obs):
        k, error = self._late(obs)
        _, delta = self.inputs(obs)
        late = torch.cat(
            [
                error[..., :30],
                obs[..., 30:42] - self.centers[k, 30:42],
                error[..., 30:],
            ],
            -1,
        )
        late = (late / self.scale).clamp(-20, 20)
        features = torch.where(self.is_late(obs)[..., None], late, delta)
        return self.critic(torch.cat([features, obs[..., 42:43]], -1)).squeeze(-1)


def load_policy(path):
    """A capture policy of either kind; returns (policy, saved dict)."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved["policy"]
    if saved.get("kind") not in ("capture_residual_v1", KIND):
        raise ValueError(f"{path} is not a capture checkpoint")
    balance = {
        key: state["balance_" + key].numpy()
        for key in ["weight", "bias", "lower", "upper"]
    }
    cls = RobustCapturePolicy if saved["kind"] == KIND else CapturePolicy
    policy = cls(
        state["actions"],
        state["centers"],
        balance,
        int(state["active_steps"]),
        float(state["bound"]),
    )
    policy.load_state_dict(state)
    return policy, saved


def robust_from(path):
    """The baseline at `path` as a trainable RobustCapturePolicy."""
    source, saved = load_policy(path)
    state = source.state_dict()
    balance = {
        key: state["balance_" + key].numpy()
        for key in ["weight", "bias", "lower", "upper"]
    }
    policy = RobustCapturePolicy(
        state["actions"],
        state["centers"],
        balance,
        int(state["active_steps"]),
        float(state["bound"]),
    )
    policy.load_state_dict(state)
    return policy, saved


def _trajectory(path):
    if "trajectory" not in _cache:
        _cache["trajectory"] = load_trajectory(
            Path(path), TRAJECTORY_SHA256, build_model()
        )
    return _cache["trajectory"]


def episode(job):
    """One whole maneuver; stochastic for training, deterministic to validate."""
    torch.set_num_threads(1)
    key = (job["weights"], job["update"])
    if _cache.get("key") != key:
        _cache["policy"] = load_policy(job["weights"])[0].eval()
        _cache["key"] = key
    policy = _cache["policy"]
    seed = job["seed"]
    condition = draw(job["ranges"], seed)
    try:
        env = RandomizedEnvironment(
            build_model(), _trajectory(job["trajectory"]), seed, condition, HORIZON
        )
    except ValueError:
        return None  # start state rejected
    torch.manual_seed(seed)
    trace = []
    for _ in range(HORIZON):
        obs = torch.tensor(env.observation())
        with torch.no_grad():
            if job["explore"]:
                distribution = policy.distribution(obs)
                latent = distribution.sample()
                log = distribution.log_prob(latent).sum()
                action = policy.action(obs, latent)
                value = policy.value(obs)
            else:
                action = policy(obs[None])[0]
        _, reward, done = env.step(action.numpy())
        if job["explore"]:
            trace.append((obs, latent, float(log), float(value), reward))
        if done:
            break
    result = env.result()
    row = dict(
        seed=seed,
        complete_success=bool(result["complete_success"]),
        capture_success=bool(result["capture_success"]),
        strict_hold_seconds=float(result["strict_hold_seconds"]),
        steps=int(env.index),
    )
    if not job["explore"]:
        return row
    rewards = np.array([r[4] for r in trace])
    rewards[-1] += SUCCESS_BONUS * row["complete_success"]
    values = np.array([r[3] for r in trace] + [0.0])  # every episode ends
    advantages = np.zeros(len(trace))
    gae = 0.0
    for i in reversed(range(len(trace))):
        gae = rewards[i] + 0.99 * values[i + 1] - values[i] + 0.99 * 0.95 * gae
        advantages[i] = gae
    return row | dict(
        obs=torch.stack([r[0] for r in trace]),
        latent=torch.stack([r[1] for r in trace]),
        log=torch.tensor([r[2] for r in trace]),
        advantage=torch.tensor(advantages, dtype=torch.float32),
        returns=torch.tensor(advantages + values[:-1], dtype=torch.float32),
        reward=float(rewards.sum()),
    )


def _save(path, policy, optimizer, update, args, history, source_sha):
    contents = dict(
        kind=KIND,
        schema=1,
        policy=policy.state_dict(),
        optimizer=optimizer.state_dict(),
        update=update,
        reference_sha256=TRAJECTORY_SHA256,
        source_checkpoint_sha256=source_sha,
        config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        history=history,
    )
    temporary = path.with_suffix(".tmp")
    torch.save(contents, temporary)
    os.replace(temporary, path)


def _validate(pool, base, ranges_by_name, trials):
    scores = {}
    for name, ranges in ranges_by_name.items():
        jobs = [
            base | dict(seed=VALIDATION_SEED + 1 + i, ranges=ranges, explore=False)
            for i in range(trials)
        ]
        rows = [r for r in pool.map(episode, jobs, chunksize=1) if r is not None]
        scores[name] = dict(
            successes=sum(r["complete_success"] for r in rows),
            trials=len(rows),
            mean_hold_seconds=float(np.mean([r["strict_hold_seconds"] for r in rows])),
        )
    return scores


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source_sha = sha256(args.checkpoint)
    latest = args.output / "latest.pt"
    if args.resume and latest.exists():
        policy, saved = load_policy(latest)
        history, update = saved["history"], saved["update"]
        source_sha = saved["source_checkpoint_sha256"]
    else:
        if any(args.output.glob("update_*.pt")):
            raise ValueError(
                f"{args.output} already holds a run; use --resume or a new directory"
            )
        policy, _ = robust_from(args.checkpoint)
        history, update = [], 0
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    if update:
        optimizer.load_state_dict(saved["optimizer"])
    train_ranges = scaled(PROFILES[args.profile], args.scale)
    validation = {"release": PROFILES["release"], "training": train_ranges}
    weights = args.output / "weights.pt"
    base = dict(trajectory=str(args.trajectory), weights=str(weights))
    started = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
        if not history:
            _save(weights, policy, optimizer, 0, args, history, source_sha)
            initial = _validate(
                pool, base | dict(update=0), validation, args.validation_trials
            )
            history.append(dict(update=0, validation=initial))
            _save(
                args.output / "update_0000.pt",
                policy,
                optimizer,
                0,
                args,
                history,
                source_sha,
            )
            print(dict(update=0, validation=initial), flush=True)
        best = max(_score(row) for row in history)
        for update in range(update + 1, args.updates + 1):
            _save(weights, policy, optimizer, update - 1, args, history, source_sha)
            jobs = [
                base
                | dict(
                    update=update - 1,
                    seed=args.seed * 1000 + update * 1000 + i,
                    ranges=train_ranges,
                    explore=True,
                )
                for i in range(args.episodes)
            ]
            episodes = [
                e for e in pool.map(episode, jobs, chunksize=1) if e is not None
            ]
            obs = torch.cat([e["obs"] for e in episodes])
            latent = torch.cat([e["latent"] for e in episodes])
            old = torch.cat([e["log"] for e in episodes])
            advantage = torch.cat([e["advantage"] for e in episodes])
            returns = torch.cat([e["returns"] for e in episodes])
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            stopped = False
            for _ in range(4):
                for ids in torch.randperm(len(obs)).split(1024):
                    log = policy.distribution(obs[ids]).log_prob(latent[ids]).sum(-1)
                    log_ratio = log - old[ids]
                    ratio = log_ratio.exp()
                    if float(((ratio - 1) - log_ratio).mean()) > 0.02:
                        stopped = True
                        break
                    actor = -torch.minimum(
                        ratio * advantage[ids], ratio.clamp(0.8, 1.2) * advantage[ids]
                    ).mean()
                    critic = (policy.value(obs[ids]) - returns[ids]).square().mean()
                    loss = actor + 0.5 * critic
                    optimizer.zero_grad()
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    if not bool(torch.isfinite(loss) and torch.isfinite(norm)):
                        raise ValueError("non-finite PPO loss or gradient")
                    optimizer.step()
                if stopped:
                    break
            with torch.no_grad():
                log_ratio = policy.distribution(obs).log_prob(latent).sum(-1) - old
                kl = float(((log_ratio.exp() - 1) - log_ratio).mean())
            _save(weights, policy, optimizer, update, args, history, source_sha)
            scores = _validate(
                pool, base | dict(update=update), validation, args.validation_trials
            )
            row = dict(
                update=update,
                transitions=len(obs),
                training_successes=sum(e["complete_success"] for e in episodes),
                training_episodes=len(episodes),
                mean_reward=float(np.mean([e["reward"] for e in episodes])),
                approx_kl=kl,
                kl_early_stopped=stopped,
                validation=scores,
                seconds=time.perf_counter() - started,
            )
            history.append(row)
            _save(
                args.output / f"update_{update:04d}.pt",
                policy,
                optimizer,
                update,
                args,
                history,
                source_sha,
            )
            _save(latest, policy, optimizer, update, args, history, source_sha)
            if _score(row) > best:
                best = _score(row)
                _save(
                    args.output / "best.pt",
                    copy.deepcopy(policy),
                    optimizer,
                    update,
                    args,
                    history,
                    source_sha,
                )
            print({k: v for k, v in row.items() if k != "seconds"}, flush=True)
            if not np.isfinite(kl) or kl > 0.05:
                print("STOP: policy moved too far in one update", flush=True)
                break
    weights.unlink(missing_ok=True)
    return dict(
        history=history, source_checkpoint_sha256=source_sha, best_score=list(best)
    )


def _score(row):
    """Selection: never trade away the released condition for the randomized one."""
    v = row["validation"]
    return (
        min(v["release"]["successes"], v["training"]["successes"]),
        v["training"]["successes"] + v["release"]["successes"],
        v["training"]["mean_hold_seconds"],
    )
