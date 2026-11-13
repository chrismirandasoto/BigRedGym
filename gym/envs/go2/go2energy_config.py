"""Energy-shaped kickup and balance; initial values, tune in simulation."""

from gym.envs.go2.go2_config import Go2RunnerCfg
from gym.envs.go2.go2handstand_config import Go2HandstandCfg


class Go2EnergyCfg(Go2HandstandCfg):
    class env(Go2HandstandCfg.env):
        # Latest trigger, the kick, and a full curriculum hold must all fit.
        episode_length_s = 10

    class pendulum:
        # Centre of mass above the front-foot midpoint in the handstand reset
        # pose, metres. Measured in simulation; test_go2_energy.py checks it,
        # so re-measure after changing reward_settings.kick_knee_target.
        length = 0.563
        # Energy to hold, as a fraction of m*g*length. 1 is upright at rest.
        energy_target = 1.0
        # Width of the squared exponential, in the same fraction. Standing
        # still is 0.585, so much below 0.2 leaves no gradient to start from.
        energy_sigma = 0.2
        # Width for the angle of the mass from directly above the pivot, rad.
        angle_sigma = 0.3
        # Count energy only while both front feet carry load.
        require_front_support = True

    class handstand_trigger:
        # Training: seconds after a standing reset before the command turns on.
        delay_s = [0.5, 2.0]
        # Standing episodes that never get the command, to keep standing learned.
        never_fraction = 0.1

    class scaling(Go2HandstandCfg.scaling):
        # Radians per policy unit, around a reference that already follows the
        # swing: enough for a crouch and kick, fine enough to balance.
        dof_pos_target = 12 * [0.5]


class Go2EnergyRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        obs = Go2RunnerCfg.actor.obs + ["handstand_command"]
        # Unit noise would be 0.5 rad per joint, too rough to hold a handstand.
        init_noise_std = 0.3

    class critic(Go2RunnerCfg.critic):
        obs = Go2RunnerCfg.critic.obs + ["handstand_command", "pendulum_obs"]

        class reward:
            class weights:
                pendulum_energy = 2.0
                pendulum_upright = 10.0
                stand_ready = 2.0
                handstand_slip = 0.5  # the model assumes a pivot that stays put
                torques = 5e-6
                action_rate = 0.1
                action_rate2 = 0.01
                dof_pos_limits = 1.0
                feet_contact_forces = 0.2
                collision = 1.0

            class termination_weight:
                termination = 2.0  # event cost; reward cancels runner dt scaling

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2energy"
        max_iterations = 1500
