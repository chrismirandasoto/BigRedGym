"""PPO2 survives non-finite batches; the actor's optional noise floor."""

import torch
from tensordict import TensorDict

from learning.algorithms.ppo2 import PPO2
from learning.modules.actor import Actor
from learning.modules.critic import Critic


def make(std_min=None, init_noise_std=0.3):
    torch.manual_seed(0)
    actor = Actor(
        6,
        3,
        [16],
        activation="elu",
        normalize_obs=False,
        init_noise_std=init_noise_std,
        std_min=std_min,
    )
    critic = Critic(6, [16], activation="elu", normalize_obs=False)
    return PPO2(
        actor,
        critic,
        batch_size=64,
        max_gradient_steps=4,
        schedule="fixed",
        entropy_coef=0.0,
    ), actor


def batch(steps=8, envs=8):
    """Rollout layout: [steps, envs]."""
    return TensorDict(
        {
            "actor_obs": torch.randn(steps, envs, 6),
            "actions": torch.randn(steps, envs, 3),
            "advantages": torch.randn(steps, envs),
        },
        batch_size=(steps, envs),
    )


def test_a_nan_advantage_skips_the_step_and_keeps_the_weights():
    alg, actor = make()
    before = {k: v.clone() for k, v in actor.state_dict().items()}
    data = batch()
    data["advantages"][0, 5] = float("nan")
    alg.update_actor(data)
    assert alg.skipped_steps == 4
    for key, value in actor.state_dict().items():
        torch.testing.assert_close(value, before[key])
    # A clean batch still trains.
    alg.update_actor(batch())
    assert any(not torch.equal(v, before[k]) for k, v in actor.state_dict().items())
    assert alg.skipped_steps == 4


def test_a_collapsed_std_cannot_overflow_the_ratio():
    alg, actor = make(init_noise_std=1e-30)
    data = batch()
    alg.update_actor(data)  # log-ratios of order 1e60 without the cap
    for value in actor.state_dict().values():
        assert torch.isfinite(value).all()


def test_noise_floor_is_opt_in():
    _, plain = make(init_noise_std=1e-4)
    _, floored = make(std_min=0.05, init_noise_std=1e-4)
    obs = torch.randn(5, 6)
    plain.update_distribution(obs)
    floored.update_distribution(obs)
    assert torch.allclose(plain.action_std, torch.full((5, 3), 1e-4))
    assert torch.allclose(floored.action_std, torch.full((5, 3), 0.05))
