import argparse
import json
import os
import signal
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Deploy a policy to a Unitree Go2.")
    parser.add_argument("interface", help="Robot network interface, e.g. eth0")
    parser.add_argument(
        "--bundle",
        help="Policy bundle directory from scripts/export_go2_policy.py. "
        "Without it, the actor is built from the latest training run in logs/.",
    )
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Go2 deployment requires Linux (timerfd).")

    try:
        from unitree_sdk2py.core import channel
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except ModuleNotFoundError as exc:
        if exc.name != "unitree_sdk2py":
            raise
        parser.error(
            "Unitree SDK is not installed. Follow README_DEPLOY.md, then run "
            "uv run --frozen --extra unitree_sdk python -m go2_deploy.deploy "
            "<interface>."
        )

    # A bundle names the custom implementation it was exported with; select it
    # before any go2_deploy module imports the custom implementation.
    if args.bundle:
        with open(os.path.join(args.bundle, "deploy_meta.json")) as f:
            wanted = json.load(f).get("custom_impl")
        if wanted:
            os.environ.setdefault("GO2_CUSTOM_IMPL", wanted)

    from go2_deploy.utility.csv_logger import new_run_dir
    from go2_deploy.utility.terminal_log import TerminalLog

    # First, so terminal.log also covers policy loading and connection errors
    run_dir = new_run_dir()
    terminal_log = TerminalLog(run_dir)
    print(f"go2_deploy {' '.join(sys.argv[1:])}")

    from go2_deploy.main_controller import MainController
    from go2_deploy.keyboard_handler import KeyboardHandler
    from go2_deploy.rl_controller import RLController

    # Load the policy before touching the robot, so a bad bundle fails here.
    if args.bundle:
        from go2_deploy.policy_bundle import load_bundle

        policy, cfg = load_bundle(args.bundle)
        print(f"Loaded policy bundle {args.bundle}")
    else:
        from go2_deploy.deploy_config import DeployConfig

        policy, cfg = None, DeployConfig()
    rl_controller = RLController(cfg, policy)

    # CycloneDDS 0.10.2 bug workaround
    channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
        "<Verbosity>config</Verbosity>", "<Verbosity>none</Verbosity>"
    )

    ChannelFactoryInitialize(
        0,
        args.interface,
    )
    controller = MainController(cfg, rl_controller, run_dir=run_dir)
    terminal_log.state = lambda: controller._state.name

    # SIGTERM (the launcher's L2 + R2) and SIGHUP (tmux kill-session): run the
    # e-stop (lie down, stand back up) and exit, instead of dying mid-policy
    stop_signal = []

    def request_shutdown(signum, frame):
        stop_signal.append(signal.Signals(signum).name)
        controller.request_shutdown()

    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGHUP, request_shutdown)
    keyboard_handler = KeyboardHandler(controller)  # noqa: F841

    print(
        "\nConnection successful! Keep the robot away from obstacles and "
        + "always have access to the remote controller\n"
    )

    while not controller.shut_down:
        time.sleep(0.2)
    terminal_log.record_exit(f"stopped ({stop_signal[0]})")


if __name__ == "__main__":
    main()
