"""Front-foot pendulum energy, its two rewards, and the handstand command."""

import math

import pytest
import torch

from gym.envs.base.mujoco_cpu_backend import MuJocoCPUBackend
from gym.envs.go2.go2energy import NEVER, Go2Energy
from gym.envs.go2.go2energy_config import Go2EnergyCfg, Go2EnergyRunnerCfg
from gym.utils.interfaces.teleop_bindings import BINDINGS, TeleopCommands
from gym.utils.task_registry import task_registry

N = 4
ALL = torch.ones(N, dtype=torch.bool)


def make_env(pitch_degrees):
    """Every reset at one exact pitch: 90 is the handstand, 0 is standing."""
    cfg = Go2EnergyCfg()
    cfg.seed = 7
    cfg.env.num_envs = N
    c = cfg.handstand_curriculum
    c.pitch_degrees = [pitch_degrees]
    c.tilt_noise_degrees = [0.0]
    c.angular_velocity_noise = [0.0]
    cfg.domain_randomization.startup.link_mass_scale_range = [1.0, 1.0]
    task_registry.convert_frequencies_to_params(cfg, Go2EnergyRunnerCfg())
    backend = MuJocoCPUBackend()
    return Go2Energy(cfg, "cpu", True, backend), backend


@pytest.fixture
def top():
    env, backend = make_env(90.0)
    try:
        yield env
    finally:
        backend.close()


@pytest.fixture
def standing():
    env, backend = make_env(0.0)
    try:
        yield env
    finally:
        backend.close()


# ── Reduced-order model ──────────────────────────────────────────────────


@torch.inference_mode()
def test_configured_length_is_the_handstand_reset_pose(top):
    """At rest in the handstand the energy is exactly m*g*l, straight up."""
    top._reset_idx(ALL)
    energy, angle = top._pendulum()
    torch.testing.assert_close(energy, torch.ones(N), atol=0.01, rtol=0)
    assert angle.abs().max() < 0.02


@torch.inference_mode()
def test_standing_still_is_well_below_the_target_energy(standing):
    standing._reset_idx(ALL)
    energy, angle = standing._pendulum()
    assert 0.5 < energy.min() and energy.max() < 0.7
    # The mass is behind the front feet, not above them.
    assert angle.min() > math.radians(20)


@torch.inference_mode()
def test_kinetic_term_is_the_centre_of_mass_speed(standing):
    """Rigid upward motion at v adds exactly v^2 / 2g of height."""
    standing._reset_idx(ALL)
    rest, _ = standing._pendulum()
    speed = 1.5
    standing.root_states[:, 9] = speed
    standing._backend.reset_state(ALL)
    moving, _ = standing._pendulum()
    expected = speed**2 / (2 * 9.81) / standing.cfg.pendulum.length
    torch.testing.assert_close(
        moving - rest, torch.full((N,), expected), atol=1e-4, rtol=0
    )


@torch.inference_mode()
def test_energy_uses_mujocos_own_centre_of_mass_state(standing):
    """With every joint moving, height and speed still match the simulator."""
    import mujoco

    torch.manual_seed(0)
    for _ in range(40):
        standing.dof_pos_target[:] = 0.6 * torch.randn_like(standing.dof_pos_target)
        standing.step()
    energy, _ = standing._pendulum()
    backend = standing._backend
    pivot = standing._rigid_body_pos[:, standing.feet_indices[:2]].mean(dim=1)
    for i, data in enumerate(backend._datas):
        model = backend._model_for_env(i)
        mujoco.mj_forward(model, data)
        mujoco.mj_subtreeVel(model, data)
        height = data.subtree_com[1][2] - pivot[i, 2].item()
        kinetic = (data.subtree_linvel[1] ** 2).sum() / (2 * 9.81)
        expected = (height + kinetic) / standing.cfg.pendulum.length
        assert energy[i].item() == pytest.approx(expected, abs=1e-4)


# ── Rewards ──────────────────────────────────────────────────────────────


def loaded(env, front=True):
    """Half the weight on each front foot, or none."""
    weight = 9.81 * env._backend.link_mass.sum(dim=1)
    env.contact_forces[:] = 0
    if front:
        env.contact_forces[:, env.feet_indices[:2], 2] = (weight / 2)[:, None]


@torch.inference_mode()
def test_both_rewards_peak_at_the_supported_handstand(top):
    top._reset_idx(ALL)
    loaded(top)
    assert top._handstand_command.all()
    assert top._reward_pendulum_energy().min() > 0.99
    assert top._reward_pendulum_upright().min() > 0.99


@torch.inference_mode()
def test_right_energy_in_the_wrong_place_earns_only_the_energy_reward(standing):
    """A standing robot moving fast enough has e = 1 without being upright."""
    standing._reset_idx(ALL)
    standing._handstand_command[:] = 1
    loaded(standing)
    still, _ = standing._pendulum()
    length = standing.cfg.pendulum.length
    standing.root_states[:, 9] = torch.sqrt(2 * 9.81 * length * (1 - still))
    standing._backend.reset_state(ALL)
    loaded(standing)
    energy, _ = standing._pendulum()
    torch.testing.assert_close(energy, torch.ones(N), atol=1e-3, rtol=0)
    assert standing._reward_pendulum_energy().min() > 0.99
    assert standing._reward_pendulum_upright().max() < 0.1


@torch.inference_mode()
def test_energy_needs_the_front_feet_on_the_ground(top):
    top._reset_idx(ALL)
    loaded(top, front=False)
    assert top._reward_pendulum_energy().max() == 0
    assert top._reward_pendulum_upright().max() == 0
    top.cfg.pendulum.require_front_support = False
    assert top._reward_pendulum_energy().min() > 0.99


@torch.inference_mode()
def test_command_switches_between_standing_and_pendulum_rewards(standing):
    standing._reset_idx(ALL)
    loaded(standing)
    standing._handstand_command[:] = 0
    assert standing._reward_pendulum_energy().max() == 0
    assert standing._reward_pendulum_upright().max() == 0
    assert standing._reward_stand_ready().min() > 0.9
    standing._handstand_command[:] = 1
    assert standing._reward_stand_ready().max() == 0
    assert standing._reward_pendulum_energy().min() > 0


@torch.inference_mode()
def test_hold_time_counts_only_commanded_upright_steps(top):
    top._reset_idx(ALL)
    loaded(top)
    top.terminated[:] = False
    top.timed_out[:] = False
    top._handstand_command[0] = 0
    top._update_handstand_hold()
    assert top._hold_time.tolist() == pytest.approx([0.0] + [top.dt] * (N - 1))


# ── Handstand command ────────────────────────────────────────────────────


@torch.inference_mode()
def test_handstand_resets_start_commanded(top):
    top._reset_idx(ALL)
    assert top.handstand_command.shape == (N, 1)
    assert (top.handstand_command == 1).all()


@torch.inference_mode()
def test_standing_resets_wait_for_their_sampled_delay(standing):
    t = standing.cfg.handstand_trigger
    t.never_fraction = 0.0
    standing._reset_idx(ALL)
    low, high = (math.ceil(s / standing.dt) for s in t.delay_s)
    assert ((standing._trigger_step >= low) & (standing._trigger_step <= high)).all()
    assert not standing._handstand_command.any()
    standing.episode_length_buf[:] = standing._trigger_step - 2
    standing._post_decimation_step()
    assert not standing._handstand_command.any()
    standing._post_decimation_step()
    assert standing._handstand_command.all()
    # The command holds for the rest of the episode.
    standing._post_decimation_step()
    assert standing._handstand_command.all()


@torch.inference_mode()
def test_never_commanded_episodes_cannot_promote_the_curriculum(standing):
    standing.cfg.handstand_trigger.never_fraction = 0.999
    standing._reset_idx(ALL)
    assert (standing._trigger_step == NEVER).all()
    assert not standing._primary_reset.any()


# ── Keyboard trigger ─────────────────────────────────────────────────────


@torch.inference_mode()
def test_key_replaces_the_training_delay(standing, capsys):
    standing.cfg.handstand_trigger.never_fraction = 0.0
    standing._reset_idx(ALL)
    teleop = TeleopCommands(standing)
    assert standing.manual_handstand_trigger
    assert (standing._trigger_step == NEVER).all()
    for _ in range(3):
        standing.step()
    assert not standing._handstand_command.any()

    teleop.apply("handstand")
    assert standing._handstand_command.all()
    assert "Handstand commanded" in capsys.readouterr().out
    standing.step()
    assert standing._handstand_command.all()

    teleop.apply("reset")
    assert not standing._handstand_command.any()
    assert (standing._trigger_step == NEVER).all()


def test_handstand_key_is_bound_and_harmless_on_other_tasks(capsys):
    from tests.unit_tests.test_teleop_keyboard import FakeEnv

    assert BINDINGS["handstand"] == "H"
    TeleopCommands(FakeEnv()).apply("handstand")
    assert "no handstand command" in capsys.readouterr().out


# ── PD reference ─────────────────────────────────────────────────────────


@torch.inference_mode()
@pytest.mark.parametrize("pitch", [90.0, 70.0, 45.0, 20.0])
def test_zero_action_holds_every_curriculum_reset_pose(pitch):
    """The reference follows the swing, so resets need almost no residual.

    The reset bends the rear thigh slightly off the straight-line blend, by at
    most 0.026 rad halfway up.
    """
    env, backend = make_env(pitch)
    try:
        env._reset_idx(ALL)
        joints = env.dof_pos.index_select(1, env.actuated_dof_indices)
        assert (env._standing_posture() - joints).abs().max() < 0.03
        assert env.dof_pos_target.abs().max() < 0.03
    finally:
        backend.close()


@torch.inference_mode()
def test_reference_is_plain_standing_until_commanded(top):
    from gym.envs.go2.go2 import Go2

    top._reset_idx(ALL)
    handstand = top._standing_posture().clone()
    top._handstand_command[:] = 0
    torch.testing.assert_close(top._standing_posture(), Go2._standing_posture(top))
    assert (handstand - top._standing_posture()).abs().max() > 1.0


# ── Registration and policy interface ────────────────────────────────────


def test_task_is_registered_with_command_observation(standing):
    import gym.envs  # noqa: F401

    assert "go2energy" in task_registry.task_classes
    actor, critic = Go2EnergyRunnerCfg.actor, Go2EnergyRunnerCfg.critic
    assert "handstand_command" in actor.obs and "handstand_command" in critic.obs
    assert "pendulum_obs" in critic.obs and "pendulum_obs" not in actor.obs
    assert standing.get_states(actor.obs).shape == (N, 3 + 3 + 4 + 12 + 12 + 12 + 1)
    assert standing.get_states(critic.obs).shape[0] == N
    for name in vars(critic.reward.weights):
        if not name.startswith("_"):
            assert hasattr(standing, f"_reward_{name}"), name
