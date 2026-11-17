# Deploying a custom policy

## Setup

### Install required packages

To deploy a policy on the robot from your computer, you need to install the additional libraries ```cyclonedds``` and ```unitree-sdk2py```. To do this, run the following in the terminal:

```bash
uv run --frozen scripts/fetch_unitree_sdk.py
```

Then run this:

```bash
mkdir -p "$HOME/.local/src"
git clone --depth 1 --branch 0.10.2 \
    https://github.com/eclipse-cyclonedds/cyclonedds.git \
    "$HOME/.local/src/cyclonedds-0.10.2"
export CYCLONEDDS_HOME="$HOME/.local/opt/cyclonedds-0.10.2"
cmake -S "$HOME/.local/src/cyclonedds-0.10.2" \
    -B "$HOME/.local/src/cyclonedds-0.10.2/build" \
    -DCMAKE_INSTALL_PREFIX="$CYCLONEDDS_HOME" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
cmake --build "$HOME/.local/src/cyclonedds-0.10.2/build" \
    --config RelWithDebInfo --target install --parallel 4
```

Then run this:

```bash
uv sync --frozen --extra unitree_sdk
```

The output should be something along the lines of:

```
Built unitree-sdk2py @ <location>
Built cyclonedds==0.10.2
...
```

### Set up ethernet connection

Instructions are for Linux (Ubuntu)
- Connect an ethernet cable to the robot and to your device.
- Go to Settings -> Network -> click the settings icon under Wired -> IPv4
- Switch IPv4 Method to Manual
- Under Addresses, in the first row, type `192.168.123.99` under Addresses, and `255.255.255.0` under Netmask
- Click Apply
- Turn on the connection under Wired

## Configure deployment

Settings can be configured in `go2_deploy/deploy_config.py`. Options include:
- Control frequency
- kp (Stiffness constant)
- kd (Damping constant)
- Observation vector: a list of available observations can be found in `obs_sizes`
- Command limits (ex. pushing the joystick all the way to the right commands x m/s)
- Gait frequency (`phase_frequency`, in Hz): controls the Go2Trot phase clock and its frequency observation

For current Go2Trot policies, joint-position observations are relative to the
static nominal posture from `Go2TrotCfg` (hip 0, thigh 0.96, calf -1.36 radians).
Scaled policy actions are offsets from that posture. The gait reference is a
training reward only; deployment retains the sine/cosine phase observations and
observes the controller's active gait frequency, divided by its observation scale.
The previous-action observation contains the applied offset after optional EMA
smoothing. Neither that offset nor the resulting position target is clipped.
Deployment sends position targets and PD gains to Unitree; this path does not
compute or clamp torques locally.

Offline utility tests cover these mappings using synthetic messages. They do
not establish hardware behavior or compatibility with an older checkpoint's
controller semantics.

## Deployment

If anything goes wrong, the emergency stop on the wireless controller is X or the enter key in the terminal.

- Make sure your trained policy is in `logs/`, and that the observation vector, scaling, and other settings in `deploy_config.py` are all correct.
- Turn on the Go2 and wait until it is in the standing position.
- In the terminal, run `uv run --extra unitree_sdk python -m go2_deploy.deploy eth0`, but replace `eth0` with the actual network configuration. Keep `CYCLONEDDS_HOME` exported. If this is successful, there should be a terminal output every few seconds telling you about the keyboard controls, observation frequency, etc.
- There are 4 states the robot can be in:
    - `RECOVERY`: Initially when the script starts running, the robot is in this mode. You must enter this state from `EMERGENCY_STOP` otherwise. The default Unitree controller can be used to make the robot walk, do tricks, etc. The `F1` key on the remote control can be used to disable/enable the A, B, X, and Y buttons, so that the default Unitree button combinations can be used without accidentally turning on a different state.
    - `EMERGENCY_STOP`: The robot will immediately start damping, wait around 15 seconds, and then stand up again. The robot will automatically enter the `RECOVERY` state.
    - `INTERMEDIATE`: The robot will hold its current position. This is for testing to make sure the kp and kd values are good, etc.
    - `CUSTOM_CTRL`: The robot will use the policy you trained to move.
- To stop running the script, return to `RECOVERY`, lie the robot on the ground, and press `Ctrl+C` in the terminal.
- Logs of the robot data can be found in `logs/deploy/`.

## Onboard deployment (Go2 EDU, wireless)

The control loop can run on the Go2 EDU's Jetson, so DDS never leaves the
robot's wired network and Wi-Fi only carries SSH. The RC works exactly as
below (it reaches the code through `rt/lowstate`), with or without the PC.

1. **Network.** eduroam won't work (per-user credentials, client isolation).
   Over the ethernet cable, `ssh unitree@192.168.123.18` and check `iw dev`.
   If the Jetson has Wi-Fi, start a hotspot:
   `nmcli dev wifi hotspot ifname wlan0 ssid go2-lab password <pw>`; the PC
   joins it and reaches the Jetson at `10.42.0.1`. Otherwise use a travel router.
2. **One-time Jetson setup.** Install `uv` and `tmux`, `uv python install 3.11`
   (JetPack ships 3.8), build CycloneDDS 0.10.2 as in Setup above. The first
   deploy syncs the repo; `uv run` then installs the locked aarch64 wheels.
3. **Export a policy bundle on the PC:**
   `uv run --frozen python scripts/export_go2_policy.py --name trot_v1 [--run_name <run>]`
   writes `logs/policies/trot_v1/` with `policy.pt` and `deploy_meta.json`.
   The bundle carries the obs layout, every obs/action scale, the default
   posture, gains, command limits, provenance and the network's hash, so the
   scaling can't drift from the weights. Export right after training.
4. **Deploy:**
   `uv run --frozen python scripts/deploy_go2_remote.py --bundle logs/policies/trot_v1 --host unitree@10.42.0.1`
   rsyncs the repo and bundle and starts `go2_deploy.deploy eth0 --bundle ...`
   in a tmux session `go2_deploy` on the Jetson. Detach with `Ctrl+b d`;
   rerun the command to reattach. `--dry_run` prints the commands.
5. **Logs:** `... deploy_go2_remote.py --host unitree@10.42.0.1 --pull_logs`.
   Each run's `meta.json` includes the full bundle metadata. Each run directory
   also has `terminal.log` (everything the script printed, timestamped per
   line) and `exit.json` (why and when it ended: L2 + R2 or `tmux kill-session`,
   Ctrl+C, or a crash with its traceback, plus the state and uptime). No
   `exit.json` means it was SIGKILLed or lost power.
   `logs/deploy/launcher.log` lists every L1 + R1 / L2 + R2 press and launcher
   boot.

One policy per run: to change policy, return to `RECOVERY`, stop the run,
and deploy another bundle.

### Without the PC: start and stop from the remote (L1 + R1, L2 + R2)

1. Over the cable, install the bundle:
   `uv run --frozen python scripts/deploy_go2_remote.py --bundle logs/policies/trot_v1 --host unitree@192.168.123.18 --install`
   This syncs the repo and bundle, makes it `logs/policies/current` on the
   Jetson, and installs the `go2_launcher` systemd service. sudo asks for the
   Jetson's password. The launcher starts on every boot.
2. Unplug. With the robot lying down in a clear area, **hold L1 + R1 for 1 s**.
   The launcher (`go2_deploy/launcher.py`) only listens to the remote, and
   starts `go2_deploy.deploy` in the tmux session `go2_deploy`. About 20 s
   later the robot stands up: that's the sign it worked.
3. From there it's the normal RC flow: `A` runs the policy, `B` holds, `X` is
   the e-stop. L1 + R1 while it's already running does nothing.
4. **Hold L2 + R2 for 1 s to stop.** The launcher signals `go2_deploy`, which
   runs the same e-stop as `X` (damp, lie down, stand back up) and then exits,
   leaving the robot standing under Unitree's controller. In `RECOVERY` it
   just exits. `tmux kill-session -t go2_deploy` does the same. If the script
   is still running after 60 s, the launcher kills it and stands the robot up
   through sport mode. L1 + R1 starts a fresh run afterwards.
5. Back on the cable, `--pull_logs` as usual. To switch policy, stop with
   L2 + R2 and `--install` the new bundle.

Tuning mid-run over the cable: `ssh unitree@192.168.123.18`, then
`tmux attach -t go2_deploy` for the script's output and the keyboard controls
below. Detach with `Ctrl+b d`; the run keeps going. Debug the launcher with
`journalctl -u go2_launcher -f`.

## Known risks

- **Deploy process dies in `CUSTOM_CTRL`/`INTERMEDIATE`.** If the Python
  process exits outright (OOM, kill, Jetson crash), `LowCmd_` stops and the
  RC's X can no longer reach our code, because the e-stop is implemented in
  this process. Exceptions inside the control loop are caught by
  `RecurrentThread` and do not cause this. Current mitigations: run in tmux
  (an SSH drop does not kill it), and test hung or in a harness first.
  Candidate fixes: a supervisor that publishes damping `LowCmd_` when the
  process exits, or a separate minimal damping watchdog process that
  watches for `LowCmd_` going stale.

## Remote controls

- Switch modes
    - Enter `EMERGENCY_STOP`: `X`
    - Enter `INTERMEDIATE`: `B`
    - Enter `CUSTOM_CTRL`: `A`
- Start the script (hold 1 s): `L1 + R1` (needs `--install`)
- Stop the script: e-stop, stand back up, exit (hold 1 s): `L2 + R2` (needs `--install`)
- Disable/enable ABXY buttons: `F1`
- Increase kp by 10%: `Up` (below the left joystick)
- Decrease kp by 10%: `Down`
- Increase kd by 10%: `Right`
- Decrease kd by 10%: `Left`

## Keyboard controls
Type the following letters in the terminal:

- Switch modes
    - Enter `EMERGENCY_STOP`: `<enter key>`
    - Enter `INTERMEDIATE`: `q`
    - Enter `CUSTOM_CTRL`: `c`
- Increase kp by 10%: `i`
- Decrease kp by 10%: `k`
- Increase kd by 10%: `l`
- Decrease kd by 10%: `j`

## Something went wrong

- `deploy.py` is printing errors in the terminal
    - Make sure the robot is either lying down, or standing up while in sport mode, and exit the script in the terminal.
- The robot is shaking violently when running a custom policy
    - In `deploy_config.py`, make sure that your observation vector is correct and the values in the `DeployScaling` class match what you used during training. Also make sure you are running the correct policy from `logs/`.

## Go2 handstand

The handstand capture controller deploys through this framework as a custom
implementation (`GO2_CUSTOM_IMPL=handstand_capture.py`, selected automatically
by its bundle). It has only been run in simulation and is not ready for a
robot; read [docs/go2_handstand_deploy.md](docs/go2_handstand_deploy.md) first.
