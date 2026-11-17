# Running BigRedGym

## Setup (PC)
```bash
uv sync --frozen                 # sim + training deps (Python 3.11 is managed by uv)
uv run --frozen pytest           # default unit-test gate (no SDK, no GPU)
```

## Go2 hardware extras (PC or Jetson)
```bash
uv run --frozen scripts/fetch_unitree_sdk.py      # pinned SDK -> thirdparty/
# build CycloneDDS 0.10.2 into ~/.local/opt/cyclonedds-0.10.2 (see README_DEPLOY.md)
export CYCLONEDDS_HOME="$HOME/.local/opt/cyclonedds-0.10.2"
uv sync --frozen --extra unitree_sdk
uv run --frozen --extra unitree_sdk pytest -m unitree   # SDK tests
```
On the Jetson, add `--no-dev` to `uv sync`/`uv run`. The dev group pulls in
`mujoco-warp` → `warp-lang`, which has no wheel for the Jetson (aarch64, glibc 2.31).
pytest is in that group too, so skip the SDK tests there.

## Deploying to the Go2
- **Tethered (policy on PC):** `uv run --extra unitree_sdk python -m go2_deploy.deploy <nic>`
  (the actor is built from the latest run in `logs/`).
- **Onboard (policy on the Jetson, PC over Wi-Fi):**
  ```bash
  uv run --frozen python scripts/export_go2_policy.py --name trot_v1 [--run_name <run>]
  uv run --frozen python scripts/deploy_go2_remote.py --bundle logs/policies/trot_v1 --host unitree@10.42.0.1
  uv run --frozen python scripts/deploy_go2_remote.py --host unitree@10.42.0.1 --pull_logs
  ```
  The full procedure, RC controls and known risks are in `README_DEPLOY.md`.

## SSH into the Go2 EDU Jetson (factory defaults)
- Cable: plug the PC into the robot's ethernet port and give it a static `192.168.123.99/24` address (see README_DEPLOY.md).
- `ssh unitree@192.168.123.18`. The default password is `123`, the same password works for `sudo`, and `unitree` is a sudoer. These are Unitree's published defaults; check them on your unit. `192.168.123.161` is the motion MCU, which has no shell.
- Optionally run `ssh-copy-id unitree@192.168.123.18` so later logins don't ask for a password.
- **The Jetson has no internet by default**, and `uv`, the wheels and the CycloneDDS clone all need it. Share the PC's connection over the cable (only the PC authenticates to eduroam):
  ```bash
  # PC (WAN=wlan0, LAN=the ethernet NIC to the robot)
  sudo sysctl -w net.ipv4.ip_forward=1
  sudo iptables -t nat -A POSTROUTING -o wlan0 -j MASQUERADE
  sudo iptables -A FORWARD -i <LAN> -o wlan0 -j ACCEPT
  sudo iptables -A FORWARD -i wlan0 -o <LAN> -m state --state RELATED,ESTABLISHED -j ACCEPT
  # Jetson
  sudo ip route replace default via 192.168.123.99
  echo "nameserver 1.1.1.1" | sudo tee /etc/resolv.conf   # temporary
  ping -c1 pypi.org
  ```
  None of this survives a reboot on either side.
- Don't `apt upgrade` the Jetson: it can break JetPack and Unitree's services. Use `apt install` for specific packages (`tmux`, `cmake`, `build-essential`) and let `uv` handle Python.

## Gotchas
- `CYCLONEDDS_HOME` must be exported for `uv sync`/`uv run` with the `unitree_sdk` extra.
- `Actor.export()` also writes ONNX and fails without `onnxscript`; the bundle
  exporter writes TorchScript only, which is all the robot needs.
- eduroam can't link PC and robot: use the Jetson hotspot (`10.42.0.1`) or a travel router.
- The onboard run lives in tmux session `go2_deploy`; an SSH drop doesn't stop it.
  The RC's X is the e-stop, not the PC.
- Bundles (`logs/policies/`) and `*.pt` are gitignored; move them with the deploy script.
