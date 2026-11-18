"""Robust handstand hold after the released kick; see go2handstandhold.py.

Physics, gains and timing match the model the released controller was
optimised on (MuJoCo 2 ms Euler steps, 50 Hz targets, kp 60, kd 3). Changing
them makes the hand-over states and the deployed robot disagree with training.
"""

from gym.envs.go2.go2_config import Go2Cfg, Go2RunnerCfg

# Handstand nominal posture (configs/handstand_reference.json), rounded for
# the framework's default posture; the task uses the exact dataset values.
NOMINAL = {"hip": 0.0, "thigh": 0.7334159659826462, "calf": -1.4654903159477315}


class Go2HandstandHoldCfg(Go2Cfg):
    class env(Go2Cfg.env):
        num_envs = 4096
        # Hand-over at 2.2 s plus a full 5 s strict hold fits well inside 6 s
        # only if the hold is reached quickly; episodes run 6 s from hand-over.
        episode_length_s = 6.0
        env_spacing = 2.0

    class init_state(Go2Cfg.init_state):
        default_joint_angles = NOMINAL
        reset_mode = "reset_to_handover"

    class control(Go2Cfg.control):
        stiffness = {"hip": 60.0, "thigh": 60.0, "calf": 60.0}
        damping = {"hip": 3.0, "thigh": 3.0, "calf": 3.0}
        ctrl_frequency = 50
        desired_sim_frequency = 500

    class push_robots(Go2Cfg.push_robots):
        toggle = True
        interval_s = 1.5
        max_push_vel_xy = 0.1  # m/s added to the body
        push_box_dims = [0.2, 0.1, 0.1]

    class asset(Go2Cfg.asset):
        # Anything but a foot touching anything ends the episode, as in the
        # strict handstand gate.
        terminate_after_contacts_on = ["base", "Head", "hip", "thigh", "calf"]
        penalize_contacts_on = []

    class domain_randomization(Go2Cfg.domain_randomization):
        class startup(Go2Cfg.domain_randomization.startup):
            contact_friction_range = [0.8, 1.2]
            link_mass_scale_range = [0.95, 1.05]

        class episode(Go2Cfg.domain_randomization.episode):
            scale_ranges = {"p_gains": [0.9, 1.1], "d_gains": [0.9, 1.1]}

    class handstand_hold:
        # Recorded by scripts/handstand_handover_states.py.
        dataset = "{GYM_ROOT_DIR}/resources/handstand/handover_states.npz"
        history = 4  # earlier frames the policy sees; must match deployment
        slew_fraction = 0.7  # of the joint speed limit, per control period

        # Sensor and actuator errors, drawn per episode at full size. Each
        # grows linearly from zero over the first `ramp_steps` control steps
        # of training (all environments together).
        ramp_steps = 1500
        command_delay_substeps = 3  # 0..3 physics steps (0-6 ms)
        strength = [0.9, 1.0]  # available torque
        zero_offset = 0.005  # rad, joint zero error
        gyro_bias = 0.01  # rad/s
        attitude_bias = 0.01  # rad, IMU attitude error
        gyro_noise = 0.005  # rad/s
        encoder_noise = 0.0005  # rad
        encoder_velocity_noise = 0.05  # rad/s

        # Episode ends early on: anything but a foot touching (asset above),
        # this much tilt from the handstand, or the body this low.
        max_tilt = 0.7  # rad
        min_base_height = 0.3  # m

    class reward_settings(Go2Cfg.reward_settings):
        tilt_sigma = 0.15  # rad
        heading_sigma = 0.3  # rad
        support_sigma = 0.03  # m, COM from the front-foot segment
        com_speed_sigma = 0.15  # m/s
        angular_speed_sigma = 0.3  # rad/s
        height_sigma = 0.04  # m
        rear_clearance = 0.10  # m above the ground for full reward
        min_front_load = 0.05  # fraction of body weight per front foot


class Go2HandstandHoldRunnerCfg(Go2RunnerCfg):
    class actor(Go2RunnerCfg.actor):
        hidden_dims = [256, 256, 128]
        activation = "elu"
        obs = ["hold_obs"]
        # Inputs are already centred and scaled (handstand_hold_features.py),
        # identically in deployment; a running normaliser would have to be
        # exported too.
        normalize_obs = False
        actions = ["hold_action"]
        add_noise = False  # sensor noise is part of the task
        init_noise_std = 0.3

        class noise:
            pass

    class critic(Go2RunnerCfg.critic):
        hidden_dims = [512, 256, 128]
        normalize_obs = True
        obs = ["hold_obs", "hold_privileged"]

        class reward:
            class weights:
                strict_hold = 2.0
                upright = 1.0
                heading = 0.2
                com_over_support = 1.0
                stillness = 0.5
                heights = 0.5
                rear_clearance = 0.5
                front_support = 0.5
                front_slip = 0.5
                action_rate = 0.2
                torques = 1.0e-5
                dof_vel_limits = 1.0

            class termination_weight:
                termination = 2.0

    class algorithm(Go2RunnerCfg.algorithm):
        rollout_size = 2**16
        entropy_coef = 0.005

    class runner(Go2RunnerCfg.runner):
        experiment_name = "go2handstandhold"
        max_iterations = 2000
        save_interval = 100
