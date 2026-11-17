import torch
import numpy as np
from gym.envs.go2.go2trot_config import Go2TrotCfg
from go2_deploy.utility.import_custom_impl import custom_impl


_custom_cfg = custom_impl.CustomDeployConfig


class DeployConfig:
    # A custom implementation may override any of the policy-coupled settings
    # below by defining the same attribute on its CustomDeployConfig.
    task_name = getattr(_custom_cfg, "task_name", "go2trot")

    ctrl_freq = getattr(_custom_cfg, "ctrl_freq", 100)  # Hz

    kp = getattr(_custom_cfg, "kp", 30.0)  # Default stiffness constant
    kd = getattr(_custom_cfg, "kd", 2.0)  # Default damping constant

    # Apply exponential moving average to actions
    exp_moving_avg = getattr(_custom_cfg, "exp_moving_avg", True)
    ema_time_const = 0.095      # seconds
    ema_smoothing_factor = -np.expm1(-(1/ctrl_freq)/ema_time_const)     # alpha in EMA

    obs_vector = custom_impl.CustomDeployConfig.obs_vector

    # Scale observations
    class DeployScaling(custom_impl.CustomDeployConfig.DeployScaling):
        pass

    # Set ranges of joystick commands
    command_limits = {
        "lin_vel_x": 2.0,  # m/s
        "lin_vel_y": 1.0,  # m/s
        "yaw_vel": 3.0,  # rad/s
    }

    # Specify size of observation vector pieces
    obs_sizes = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "commands": 3,
        "dof_pos_obs": 12,
        "dof_vel": 12,
        "dof_accel": 12,
        "dof_pos_target": 12,
    }
    obs_sizes.update(custom_impl.CustomDeployConfig.custom_obs_sizes)

    # Canonical FL, FR, RL, RR order; hip, thigh, calf within each leg.
    default_dof_pos = (
        torch.as_tensor(_custom_cfg.default_dof_pos, dtype=torch.float32)
        if hasattr(_custom_cfg, "default_dof_pos")
        else torch.tensor(
            [
                angle
                for leg in ("FL", "FR", "RL", "RR")
                for angle in (
                    Go2TrotCfg.init_state.default_joint_angles["hip_joint"],
                    Go2TrotCfg.init_state.default_joint_angles[f"{leg}_thigh_joint"],
                    Go2TrotCfg.init_state.default_joint_angles["calf_joint"],
                )
            ]
        )
    )

    # s: how often terminal should print ctrl freq, instructions
    terminal_log_period = 2.0

    # Set by policy_bundle.load_bundle(): the bundle's deploy_meta.json and folder
    bundle_meta = None
    bundle_dir = None

    def __init__(self):
        self.validate()

    def obs_dim(self):
        return sum(self.obs_sizes[obs] for obs in self.obs_vector)

    # Check for unsafe config
    def validate(self):
        if self.ctrl_freq < 20:
            raise ValueError("ctrl_freq should be > 20")
        if self.kp <= 0:
            raise ValueError("kp should be > 0")
        if self.kd <= 0:
            raise ValueError("kd should be > 0")
        for obs in self.obs_vector:
            if obs not in self.obs_sizes:
                raise KeyError("observation " + obs + " not supported")
