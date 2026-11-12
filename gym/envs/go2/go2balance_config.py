"""Top-of-handstand balance first; success unlocks controlled disturbances."""

from gym.envs.base.domain_randomization import DomainRandomizationCfg
from gym.envs.go2.go2handstand_config import Go2HandstandCfg, Go2HandstandRunnerCfg


class Go2BalanceCfg(Go2HandstandCfg):
    balance_only = True

    class env(Go2HandstandCfg.env):
        # Existing policies can drift into a fall after 25–30 seconds.
        episode_length_s = 45

    # Keep physics fixed while learning the nominal balance controller.
    class domain_randomization(DomainRandomizationCfg):
        pass

    class handstand_curriculum(Go2HandstandCfg.handstand_curriculum):
        hold_seconds = 5.0
        success_fraction = 0.9
        pitch_degrees = [90.0] * 5
        tilt_noise_degrees = [0.0, 1.0, 3.0, 3.0, 5.0]
        angular_velocity_noise = [0.0, 0.0, 0.03, 0.03, 0.06]
        standing_fraction = 0.0
        replay_fraction = 0.2

    class balance_perturbations:
        enabled = True
        # Bounded world-XY delta velocity, m/s; these are impulses, not forces.
        max_delta_velocity = [0.0, 0.0, 0.0, 0.03, 0.06]
        interval_s = 6.0 #had been 3 seconds
        settled_hold_s = 0.5
        undisturbed_fraction = 0.25

    class tilt_test:
        # Evaluation resets: rigid tilt about the front-foot line, on top of
        # the nominal top pose. Positive is over-rotation (kick-up direction).
        enabled = False
        angle_degrees = [-30.0, 30.0]
        angular_velocity = [-3.0, 3.0]  # rad/s about the same line
        sampling = "grid"  # "grid": one cell per robot; "uniform": random
        grid = [21, 21]  # angle cells, rate cells

    class balance_failure:
        min_normal_force = 1.0  # newtons per front foot
        contact_grace_seconds = 0.15
        body_contact_seconds = 0.15
        collapse_seconds = 0.25
        body_contact_force = 1.0
        # Trunk angle from the nose-down vertical that counts as a fall once
        # it lasts tilt_seconds. None disables the rule. Without it a policy
        # can lie propped on its side: no trunk contact, base above the
        # collapse height, and so no fall for the whole episode.
        max_tilt_degrees = None
        tilt_seconds = 0.25

    class reward_settings(Go2HandstandCfg.reward_settings):
        valid_front_speed = 0.08  # m/s, maximum speed for a valid hold
        sustained_ramp_s = 2.0

    class push_robots(Go2HandstandCfg.push_robots):
        toggle = False  # use the gated balance perturbations instead

    class scaling(Go2HandstandCfg.scaling):
        dof_pos_target = 12 * [0.25]


class Go2BalanceRunnerCfg(Go2HandstandRunnerCfg):
    class actor(Go2HandstandRunnerCfg.actor):
        init_noise_std = 0.3

    class critic(Go2HandstandRunnerCfg.critic):
        class reward:
            class weights:
                # Additive shaping supplies gradients when the strict product
                # is near zero. Promotion still uses the original strict score.
                balance_orientation = 0.5
                balance_front_support = 0.5
                balance_clearance = 0.25
                balance_stability = 0.25
                handstand_balance = 10.0
                balance_sustained = 40.0
                balance_front_unloading = 2.0
                balance_body_contact = 5.0
                handstand_slip = 5.0
                torques = 5e-6
                action_rate = 0.1
                action_rate2 = 0.01
                dof_pos_limits = 1.0
                feet_contact_forces = 0.2
                collision = 1.0

            class termination_weight:
                termination = 20.0  # event cost; reward cancels runner dt scaling

    class runner(Go2HandstandRunnerCfg.runner):
        experiment_name = "go2balance"
