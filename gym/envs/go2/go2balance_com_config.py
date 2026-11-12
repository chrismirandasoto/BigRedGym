"""go2balance with the centre-of-mass offset from the front-foot support.

The offset is added to both observations and penalized; everything else is
inherited, so a run of this task compares directly against go2balance.
"""

from gym.envs.go2.go2balance_config import Go2BalanceCfg, Go2BalanceRunnerCfg


class Go2BalanceComCfg(Go2BalanceCfg):
    class balance_failure(Go2BalanceCfg.balance_failure):
        # The tilt test starts at most 30° over, so recoveries stay legal.
        max_tilt_degrees = 60.0

    class reward_settings(Go2BalanceCfg.reward_settings):
        # Metres of offset that earn the full penalty; about 26° of rigid tilt.
        com_offset_scale = 0.25

    class scaling(Go2BalanceCfg.scaling):
        com_cop_offset = 0.1  # metres


class Go2BalanceComRunnerCfg(Go2BalanceRunnerCfg):
    class actor(Go2BalanceRunnerCfg.actor):
        obs = Go2BalanceRunnerCfg.actor.obs + ["com_cop_offset"]

    class critic(Go2BalanceRunnerCfg.critic):
        obs = Go2BalanceRunnerCfg.critic.obs + ["com_cop_offset"]

        class reward(Go2BalanceRunnerCfg.critic.reward):
            class weights(Go2BalanceRunnerCfg.critic.reward.weights):
                balance_com_offset = 5.0

    class runner(Go2BalanceRunnerCfg.runner):
        experiment_name = "go2balance_com"
