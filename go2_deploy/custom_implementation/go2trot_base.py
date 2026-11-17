import math
import torch
from gym.envs.go2.go2trot_config import Go2TrotCfg

class CustomDeployConfig:
    obs_vector = [
        "base_ang_vel",
        "projected_gravity",
        "commands",
        "dof_pos_obs",
        "dof_vel",
        "dof_pos_target",
        "phase_obs",
        "phase_frequency",
    ]

    # Length of observation vector piece
    custom_obs_sizes = {
        "phase_obs": 2,
        "phase_frequency": 1,
    }

    class DeployScaling(Go2TrotCfg.scaling):
        pass

# Initialize any custom buffers your implementation uses (for 1 environment)
def get_custom_buffers_dict():
    test_tensor = torch.zeros(3)

    buffer_dict = {
        "phase_frequency": 2.0,
        "phase": 0.0,
        "test_tensor": test_tensor,
    }

    return buffer_dict

# Define how your custom buffers should be updated at the control frequency
def update_custom_buffers(main_controller, t):
    main_controller.phase = 2 * math.pi * main_controller.phase_frequency * t % (2 * math.pi)

# Return a dict mapping observation names to functions that generate the observation pieces
def get_custom_obs_pieces():
    obs_pieces = {
        "phase_obs": _get_obs_phase_obs,
        "phase_frequency": _get_obs_phase_frequency,
    }
    return obs_pieces

# Return the phase observation piece of the observation vector
def _get_obs_phase_obs(main_controller, lowstate_msg):
    phase = torch.tensor([main_controller.phase])
    return torch.tensor([torch.sin(phase), torch.cos(phase)])

# Return the phase frequency observation piece
def _get_obs_phase_frequency(main_controller, lowstate_msg):
    return torch.tensor([main_controller.phase_frequency])
