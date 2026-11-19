"""go2handstandhold_v2: v1's task, scored like the 67/100 evaluation.

See go2handstandhold_v2.py. Physics, errors, observations and actions are v1's
(go2handstandhold_config.py); only the gate, rewards and terminations differ.
Training starts from v1's actor (resources/handstand/hold_v1_actor.pt) with a
fresh critic.
"""

from gym.envs.go2.go2handstandhold_config import (
    Go2HandstandHoldCfg,
    Go2HandstandHoldRunnerCfg,
)


class Go2HandstandHoldV2Cfg(Go2HandstandHoldCfg):
    class reward_settings(Go2HandstandHoldCfg.reward_settings):
        unbroken_horizon = 5.0  # s; the evaluator's required unbroken hold
        slip_limit = 0.05  # m/s; the gate's front-foot slip limit


class Go2HandstandHoldV2RunnerCfg(Go2HandstandHoldRunnerCfg):
    class actor(Go2HandstandHoldRunnerCfg.actor):
        # v1 ended with 10 of 12 joints pinned at its 0.05 floor (25 mrad of
        # white noise on the targets, enough to make the front feet chatter in
        # every training rollout) and two at ~0.6. v2 restarts every joint at
        # init_actor_std (runner, below) with a lower floor; dual-clip PPO
        # (algorithm) keeps the narrow policy's tail samples bounded.
        std_min = 0.02

    class critic(Go2HandstandHoldRunnerCfg.critic):
        obs = ["hold_obs", "hold_privileged", "hold_progress"]

        class reward:
            class weights:
                strict_hold = 1.0
                # Only an unbroken hold reaches full pay, after 5 s.
                unbroken_hold = 3.0
                upright = 1.0
                heading = 0.2
                com_over_support = 1.0
                stillness = 0.5
                heights = 0.5
                rear_clearance = 0.5
                front_support = 1.0
                front_slip = 0.2
                action_rate = 1.0  # v1: 0.2, where it was ~1% of the return
                torques = 1.0e-5
                dof_vel_limits = 1.0

            class termination_weight:
                termination = 2.0

    class algorithm(Go2HandstandHoldRunnerCfg.algorithm):
        # Fine-tuning a trained actor (std ~0.145): the default start of 1e-3
        # moved it so far in one update that the PPO ratio reached ~1e6 and
        # the hold was lost (CPU smoke test). Start at v1's settled rate.
        learning_rate = 1.0e-5
        lr_range = [1e-6, 3e-4]
        dual_clip = 3.0

    class runner(Go2HandstandHoldRunnerCfg.runner):
        experiment_name = "go2handstandhold_v2"
        max_iterations = 1000
        save_interval = 50
        # v1's trained actor (logs/go2handstandhold/Nov19_03-34-56_/model_2000.pt,
        # actor weights only). Resuming a v2 run overrides it.
        init_actor_from = "{GYM_ROOT_DIR}/resources/handstand/hold_v1_actor.pt"
        init_actor_std = 0.03  # policy units (x 0.5 rad): 15 mrad
        actor_warmup_iterations = 50  # critic-only updates while it learns
