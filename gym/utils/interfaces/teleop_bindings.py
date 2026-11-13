"""Shared keyboard teleop bindings and command logic for every viewer.

One table, one set of limits — so the MuJoCo and vsim interfaces cannot
drift apart.  Each interface only supplies the plumbing for its viewer's
input model (MuJoCo: key-event callback; vsim: polled key state).

Key choice (2026-07-23) — the constraint is what the viewers reserve:

* vsim hard-reserves **W A S D** (fly the camera), **P** (pause) and
  **O** (single-step).  Those are actively disruptive, which is why the
  original WASD scheme had to go.  It also only accepts ALPHANUMERIC keys
  as strings; special keys must use its UserKey enum.
* MuJoCo's viewer binds nearly every letter to a visualisation toggle
  (I=inertia, J=joint, K=skybox, L=additive, M=CoM, N=constraint, …), so
  no letter is truly free there; a teleop press may also flip a vis flag.
  Launching the viewer with the side panels hidden disables those
  shortcuts if it ever becomes annoying.

That leaves the IJKL diamond, with yaw dropped onto N/M because the
natural U/O pair collides with vsim's step toggle:

      I              I / K   forward / back
    J K L            J / L   strafe left / right
     N M             N / M   yaw left / right
                     R       reset envs
                     Esc     quit (or close the window)
                     Up/Down raise / lower desired base height
                     H       kick up into a handstand (go2energy)

H is free in vsim. In MuJoCo it also toggles convex-hull rendering when the
side panels are shown, like every other letter here.

GLFW (MuJoCo) uses ASCII codes for letter keys, so `ord(letter)` is the
keycode. Each viewer adapter translates named special keys such as UP/DOWN
into its native key representation.
"""

import torch

# action name → uppercase letter or named special key
BINDINGS = {
    "forward": "I",
    "back": "K",
    "strafe_left": "J",
    "strafe_right": "L",
    "yaw_left": "N",
    "yaw_right": "M",
    "up": "UP",
    "down": "DOWN",
    "reset": "R",
    "stop": "X",
    "handstand": "H",
}

# key name → action name (translated by each viewer adapter)
KEY_TO_ACTION = {key: action for action, key in BINDINGS.items()}

HELP_LINES = (
    "  I/K           forward / back",
    "  J/L           strafe left / strafe right",
    "  N/M           yaw left / yaw right",
    "  Up/Down       raise / lower base height by 0.01 m",
    "  R             restart all robots in the configured start pose (keep commands)",
    "  X             stop translation and yaw (keep height)",
    "  H             kick up into a handstand and hold it until R (go2energy only)",
    "  Esc / window  quit",
    "  velocity steps: 0.1 m/s; yaw steps: 0.1 rad/s",
)


class TeleopCommands:
    """Velocity-command limits, increments, and the effect of each action.

    Engine-agnostic: it only touches ``env.commands`` and the reset path.
    """

    @torch.inference_mode()
    def __init__(self, env):
        self.env = env

        ranges = getattr(env, "command_ranges", {})
        forward_range = ranges.get("lin_vel_x", [-1.0, 2.0])
        self.max_vel_forward = max(forward_range)
        self.max_vel_backward = min(forward_range)
        self.increment_x = 0.1

        def magnitude(value):
            if isinstance(value, (list, tuple)):
                return max(abs(v) for v in value)
            return abs(value)

        self.max_vel_sideways = magnitude(ranges.get("lin_vel_y", 1.0))
        self.increment_y = 0.1
        self.max_vel_yaw = magnitude(ranges.get("yaw_vel", 2.0))
        self.increment_yaw = 0.1

        # added to increment height

        # Height control parameters
        height_ranges = getattr(env, "command_ranges", {}).get(
            "base_height", [0.30, 0.45]
        )

        self.min_height = height_ranges[0]
        self.max_height = height_ranges[1]
        self.increment_height = 0.01

        env.commands[:] = 0.0
        if env.commands.shape[1] > 3:
            env.commands[:, 3] = min(
                self.max_height,
                max(self.min_height, env.cfg.reward_settings.base_height_target),
            )
        if hasattr(env.cfg, "commands"):
            env.cfg.commands.resampling_time = env.max_episode_length_s + 1
        # Training starts the handstand after a random delay; here the key does.
        if hasattr(env, "await_handstand_trigger"):
            env.await_handstand_trigger()

    @torch.inference_mode()
    def apply(self, action: str) -> None:
        """Apply one discrete press of ``action`` (see BINDINGS)."""
        c = self.env.commands
        if action == "forward":
            c[:, 0] = torch.clamp(c[:, 0] + self.increment_x, max=self.max_vel_forward)
        elif action == "back":
            c[:, 0] = torch.clamp(c[:, 0] - self.increment_x, min=self.max_vel_backward)
        elif action == "strafe_left":
            c[:, 1] = torch.clamp(c[:, 1] + self.increment_y, max=self.max_vel_sideways)
        elif action == "strafe_right":
            c[:, 1] = torch.clamp(
                c[:, 1] - self.increment_y, min=-self.max_vel_sideways
            )
        elif action == "yaw_left":
            c[:, 2] = torch.clamp(c[:, 2] + self.increment_yaw, max=self.max_vel_yaw)
        elif action == "yaw_right":
            c[:, 2] = torch.clamp(c[:, 2] - self.increment_yaw, min=-self.max_vel_yaw)

        # two new functions for up and down actions
        elif action == "up":
            if c.shape[1] > 3:
                current_height = torch.clamp(
                    c[:, 3],
                    min=self.min_height,
                    max=self.max_height,
                )

                c[:, 3] = torch.clamp(
                    current_height + self.increment_height,
                    min=self.min_height,
                    max=self.max_height,
                )

        elif action == "down":
            if c.shape[1] > 3:
                current_height = torch.clamp(
                    c[:, 3],
                    min=self.min_height,
                    max=self.max_height,
                )

                c[:, 3] = torch.clamp(
                    current_height - self.increment_height,
                    min=self.min_height,
                    max=self.max_height,
                )

        elif action == "stop":
            c[:, :3] = 0.0
        elif action == "handstand":
            if hasattr(self.env, "trigger_handstand"):
                self.env.trigger_handstand()
                print("Handstand commanded; press R to reset to standing.")
            else:
                print("This task has no handstand command.")
        elif action == "reset":
            requested_commands = c.clone()
            self.env.timed_out[:] = True
            self.env.reset()
            self.env.commands.copy_(requested_commands)
            print("Reset all robots; velocity and height commands preserved.")
        else:
            raise ValueError(f"unknown teleop action {action!r}")

    def print_help(self, viewer_name: str) -> None:
        print("______________________________________________________________")
        print(f"Keyboard teleop ({viewer_name})")
        for line in HELP_LINES:
            print(line)
        print("______________________________________________________________")
