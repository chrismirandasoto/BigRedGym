"""Start and stop go2_deploy from the remote, without the PC.

Hold L1 + R1 (1 s): start go2_deploy with logs/policies/current in the tmux
    session `go2_deploy` (attach over the cable for the keyboard controls).
Hold L2 + R2 (1 s): SIGTERM go2_deploy, which runs its e-stop (lie down,
    stand back up) and exits (deploy.py). If it is still running after
    STOP_TIMEOUT_S, SIGKILL it and have sport mode stand the robot up here.

Runs on the Jetson as the go2_launcher service (scripts/go2_launcher.service,
installed by `deploy_go2_remote.py --install`). No torch import, so it is up
seconds after boot. Every press is appended to logs/deploy/launcher.log.
"""

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime

from go2_deploy.combos import START, STOP, ComboDetector, held

TMUX_SESSION = "go2_deploy"
LAUNCHER_LOG = "logs/deploy/launcher.log"
STOP_TIMEOUT_S = 60.0  # e-stop + stand-up is ~25 s, longer if X was mid-way
# The deploy's python, not the tmux server (whose argv keeps the whole command)
DEPLOY_PATTERN = r"^\S*python[0-9.]* -m go2_deploy\.deploy( |$)"


def log(message):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)  # journalctl -u go2_launcher
    os.makedirs(os.path.dirname(LAUNCHER_LOG), exist_ok=True)
    with open(LAUNCHER_LOG, "a") as f:
        f.write(line + "\n")


def deploy_pids():
    out = subprocess.run(["pgrep", "-f", DEPLOY_PATTERN], capture_output=True)
    return [int(pid) for pid in out.stdout.split()]


def tmux_command(repo_dir, interface, bundle, python=sys.executable):
    deploy = (
        f"cd {shlex.quote(repo_dir)} && {shlex.quote(python)} -m go2_deploy.deploy "
        f"{shlex.quote(interface)} --bundle {shlex.quote(bundle)}; exec bash --norc"
    )
    pane = f"bash --noprofile --norc -c {shlex.quote(deploy)}"
    return ["tmux", "new-session", "-d", "-s", TMUX_SESSION, pane]


def start(interface, bundle_link):
    if deploy_pids():
        return log("L1 + R1: go2_deploy already running, ignoring")
    if not os.path.isfile(os.path.join(bundle_link, "deploy_meta.json")):
        return log(f"L1 + R1: no policy bundle at {bundle_link}, ignoring")
    # a session left by a stopped run only holds a shell; replace it
    subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], capture_output=True)
    bundle = os.path.relpath(os.path.realpath(bundle_link))  # real name in logs
    log(f"L1 + R1: starting go2_deploy with {bundle}")
    subprocess.run(tmux_command(os.getcwd(), interface, bundle))


def sport_recover():
    """Fallback after a SIGKILL: put sport mode in charge and stand up."""
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
        MotionSwitcherClient,
    )
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    msc, sport = MotionSwitcherClient(), SportClient()
    for client in (msc, sport):
        client.SetTimeout(5.0)
        client.Init()
    time.sleep(5)  # let MSC start after LowCmd_ stops, as switch_to_recovery does
    msc.SelectMode("mcf")
    for _ in range(10):
        if sport.RecoveryStand() == 0:
            return log("L2 + R2: RecoveryStand() succeeded")
        time.sleep(1)
    log("L2 + R2: RecoveryStand() did not succeed")


def stop():
    if not deploy_pids():
        return log("L2 + R2: go2_deploy not running, ignoring")
    log("L2 + R2: asking go2_deploy to e-stop and exit")
    for pid in deploy_pids():
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + STOP_TIMEOUT_S
    while time.monotonic() < deadline:
        if not deploy_pids():
            return log("L2 + R2: go2_deploy exited")
        time.sleep(0.5)
    log(f"L2 + R2: go2_deploy still running after {STOP_TIMEOUT_S:.0f}s, killing")
    for pid in deploy_pids():
        os.kill(pid, signal.SIGKILL)
    sport_recover()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("interface", help="Jetson's robot NIC, e.g. eth0")
    parser.add_argument("--bundle", default="logs/policies/current")
    args = parser.parse_args()

    from unitree_sdk2py.core import channel
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    # CycloneDDS 0.10.2 bug workaround (same as go2_deploy.deploy)
    channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
        "<Verbosity>config</Verbosity>", "<Verbosity>none</Verbosity>"
    )
    ChannelFactoryInitialize(0, args.interface)
    remote = bytes(40)

    def on_lowstate(msg):
        nonlocal remote
        remote = bytes(msg.wireless_remote)

    subscriber = ChannelSubscriber("rt/lowstate", LowState_)  # keep a reference
    subscriber.Init(on_lowstate, 10)
    start_combo, stop_combo = ComboDetector(), ComboDetector()
    log(f"Launcher ready: L1 + R1 starts {args.bundle}, L2 + R2 stops it")

    while True:
        time.sleep(0.05)
        now = time.monotonic()
        stop_now = stop_combo.update(held(remote, STOP), now)
        start_now = start_combo.update(held(remote, START), now)
        if stop_now:  # stop wins if both fire together
            stop()
        elif start_now:
            start(args.interface, args.bundle)


if __name__ == "__main__":
    main()
