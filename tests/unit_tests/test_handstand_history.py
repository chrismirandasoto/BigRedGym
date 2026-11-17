"""History-aware correction: policy, observation piece, export, curriculum draws."""

import numpy as np
import pytest
import torch

from experiments.handstand_robustness.history_policy import (
    KIND,
    HistoryResidualPolicy,
    from_base_state,
    load_history_policy,
)
from experiments.handstand_robustness.randomization import PROFILES, draw, scaled
from go2_deploy.state import State
from tests.unit_tests.test_handstand_deploy import (  # noqa: F401  (fixtures)
    handstand,
    message,
    random_capture_policy,
    robot,
)

STEPS, ACTIVE = 20, 8  # of random_capture_policy


def observations(policy, count=40, seed=0, size=0.3):
    generator = torch.Generator().manual_seed(seed)
    obs = size * torch.randn(count, policy.num_obs, generator=generator)
    obs[:, 42] = torch.arange(count) % 30 * 0.02 / 8
    return obs


def make(**settings):
    defaults = dict(history=3, bound=0.2, hidden=32, privileged=5)
    return from_base_state(
        random_capture_policy().state_dict(), **(defaults | settings)
    )


def test_untrained_policy_is_the_released_controller_exactly():
    base = random_capture_policy()
    policy = make()
    obs = observations(policy)
    assert policy.num_obs == 47 + 42 * 3
    with torch.no_grad():
        torch.testing.assert_close(policy(obs), base(obs[:, :47]), atol=0, rtol=0)
        # The stochastic form at zero is the same action.
        zero = policy.action(obs, torch.zeros(len(obs), 12))
        torch.testing.assert_close(zero, base(obs[:, :47]), atol=0, rtol=0)
        assert policy.value(obs, torch.zeros(len(obs), 5)).shape == (len(obs),)
    # Only the correction, the critic and the noise level are trainable.
    names = {n.split(".")[0] for n, p in policy.named_parameters() if p.requires_grad}
    assert names == {"actor", "critic", "log_std"}


def test_trained_correction_is_bounded_and_added_to_the_released_action():
    policy = make(bound=0.1)
    with torch.no_grad():
        policy.actor[-1].bias.fill_(50.0)  # saturate the correction
        obs = observations(policy, size=0.01)
        released = policy.released(obs)
        expected = (released + 0.1).clamp(-1, 1)
        torch.testing.assert_close(policy(obs), expected, atol=1e-6, rtol=0)


def test_history_frames_from_before_the_maneuver_are_ignored():
    policy = make()
    with torch.no_grad():
        for layer in policy.actor:
            if hasattr(layer, "weight"):
                layer.weight.normal_(0, 0.3)
        obs = observations(policy, count=1)
        changed = obs.clone()
        changed[:, 47 + 42 :] += 1.0  # frames 2 and 3 periods back
        for step, same in [(0, True), (1, True), (2, False), (5, False)]:
            obs[:, 42] = changed[:, 42] = step * 0.02 / 8
            equal = torch.allclose(policy(obs), policy(changed))
            assert equal == same, step
        # The most recent frame counts from step 1 on.
        changed = obs.clone()
        changed[:, 47 : 47 + 42] += 1.0
        obs[:, 42] = changed[:, 42] = 0.0
        assert torch.allclose(policy(obs), policy(changed))
        obs[:, 42] = changed[:, 42] = 0.02 / 8
        assert not torch.allclose(policy(obs), policy(changed))


def test_wider_balance_limit_only_differs_when_the_released_feedback_saturates():
    base = random_capture_policy()
    wide = make(late_bound=0.3)
    obs = observations(wide)
    late = (obs[:, 42] * 400).round() >= ACTIVE
    with torch.no_grad():
        k = (obs[:, 42] * 400).round().long().clamp(0, STEPS - 1)
        error = torch.cat([obs[:, :30], obs[:, 43:47]], -1) - torch.cat(
            [base.centers[k, :30], base.centers[k, 43:47]], -1
        )
        feedback = error @ base.balance_weight.T
        inside = (feedback.abs() <= 0.05).all(-1)
        released, widened = base(obs[:, :47]), wide.released(obs)
    assert (late & ~inside).any() and (~late).any()
    same = torch.isclose(released, widened, atol=1e-7).all(-1)
    assert same[~late].all()  # before the hand-over nothing changes
    assert not same[late & ~inside].any()
    expected = (base.actions[k] + feedback.clamp(-0.3, 0.3)).clamp(-1, 1)
    torch.testing.assert_close(widened[late], expected[late], atol=1e-6, rtol=0)
    assert wide.settings()["late_bound"] == pytest.approx(0.3)
    assert make().settings()["late_bound"] is None


def test_checkpoint_round_trip_and_traced_export_agree(tmp_path):
    policy = make(late_bound=0.2)
    with torch.no_grad():
        policy.actor[-1].weight.normal_(0, 0.05)
    path = tmp_path / "policy.pt"
    torch.save(
        dict(kind=KIND, policy=policy.state_dict(), settings=policy.settings()), path
    )
    loaded, saved = load_history_policy(path)
    assert isinstance(loaded, HistoryResidualPolicy) and saved["kind"] == KIND
    obs = observations(policy)
    traced = torch.jit.trace(loaded, torch.zeros(loaded.num_obs))
    with torch.no_grad():
        expected = policy(obs)
        torch.testing.assert_close(loaded(obs), expected)
        for row, want in zip(obs, expected):  # the robot calls it unbatched
            torch.testing.assert_close(traced(row), want, atol=1e-6, rtol=0)
    torch.save(dict(kind="capture_residual_v1"), path)
    with pytest.raises(ValueError, match=KIND):
        load_history_policy(path)


def test_history_piece_holds_the_previous_observation_frames(handstand):  # noqa: F811
    module, controller = handstand
    cfg = controller.cfg
    cfg.obs_vector = list(cfg.obs_vector) + ["hs_history"]
    cfg.obs_sizes = dict(cfg.obs_sizes) | {"hs_history": 3 * 42}
    maneuver = module._maneuver(controller)
    q, dq = maneuver.initial_joints, np.zeros(12)
    controller._state = State.CUSTOM_CTRL
    controller.last_lowstate_msg = message(q, dq, 2)
    piece = module.get_custom_obs_pieces()["hs_history"]

    def period(frame_value):
        controller.last_obs = torch.full((47 + 126,), float(frame_value))
        module.process_action(controller, torch.zeros(12))
        return piece(controller, controller.last_lowstate_msg).reshape(3, 42)

    maneuver.activate(q)
    assert (period(9.0) == 0).all()  # standing: nothing is remembered
    maneuver.start_now(q, np.array([1.0, 0, 0, 0]), height=0.34)
    assert (piece(controller, None) == 0).all()  # step 0 sees no history
    frames = period(1.0)
    assert (frames[0] == 1).all() and (frames[1:] == 0).all()
    frames = period(2.0)
    assert (frames[0] == 2).all() and (frames[1] == 1).all() and (frames[2] == 0).all()
    period(3.0)
    frames = period(4.0)  # oldest frame drops out, most recent first
    assert [float(f[0]) for f in frames] == [4.0, 3.0, 2.0]
    # A new maneuver starts with an empty history.
    maneuver.activate(q)
    assert (period(9.0) == 0).all()


def test_history_piece_must_be_last_and_whole_frames(handstand):  # noqa: F811
    module, controller = handstand
    controller.cfg.obs_vector = ["hs_history"] + list(controller.cfg.obs_vector)
    controller.cfg.obs_sizes = dict(controller.cfg.obs_sizes) | {"hs_history": 84}
    with pytest.raises(ValueError, match="last"):
        module._history(controller)


def test_scaled_command_delay_becomes_rarer_not_absent():
    """So the curriculum meets delays gradually instead of all at once."""
    delays = [
        draw(scaled(PROFILES["deploy"], 0.1), seed)["command_delay_substeps"]
        for seed in range(2000)
    ]
    assert set(delays) == {0, 1}
    assert np.mean(delays) == pytest.approx(0.15, abs=0.03)  # mean of U(0, 0.3)
    full = [draw(PROFILES["deploy"], s)["command_delay_substeps"] for s in range(500)]
    assert set(full) == {0, 1, 2, 3}
    assert draw({"command_delay_substeps": [2, 2]}, 0)["command_delay_substeps"] == 2


def test_training_cli_defaults_are_consistent():
    """Catches a renamed option before a long server run does."""
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "scripts/handstand_robust_train.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0
    for option in (
        "--resume",
        "--workers",
        "--late-bound",
        "--initial-scale",
        "--target-kl",
    ):
        assert option in done.stdout
