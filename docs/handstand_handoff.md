# Handstand deployment and robustness: where it stands

Written 2026-11-17. Branch `cm2233/handstand-deploy`. Everything below is
simulation; nothing has run on a robot. Full results and tables are in
[go2_handstand_deploy.md](go2_handstand_deploy.md).

## Status in one paragraph

The released handstand controller (`artifacts/capture_ppo_latest.pt`, 67/100)
now deploys through Cornell's `go2_deploy` framework and reproduces training
in simulation. It is not ready for a robot: it loses the handstand with a
2 ms command delay, a 2% joint stiffness error, or a 0.03° attitude error.
Fine-tuning its gains did not help. A structurally different training run is
implemented and waiting to be run on a server; whether it works is unknown.

## The one thing left to do

Run the server training and see whether the error scale climbs.

1. Get the code to the server. The work is on `cm2233/handstand-deploy`; if
   `git status` still shows changes, commit and push first.
2. Get the artifacts to the server. `artifacts/` is git-ignored. It comes
   from the release `handstand-capture-67of100-v1` (see
   [handstand_reproduce.md](handstand_reproduce.md)), or copy
   `~/mae5110/BigRedGym-handstand/reproduce/artifacts/` from the laptop. The
   two files that matter:
   - `artifacts/capture_ppo_latest.pt`, SHA256 `90117fc2ff80033663578e12ba664e4b1ff4c15c4c04f19619409a800458ba6c`
   - `artifacts/handstand_reference/trajectory.npz`, SHA256 `21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92`
3. Train (CPU cores, no GPU; run inside `tmux` so it survives a disconnect):

   ```bash
   uv sync --frozen --no-dev
   .venv/bin/python scripts/handstand_robust_train.py --output output/robust_history_v1
   # stop with Ctrl+C; continue with the same command plus --resume
   ```

   If `uv sync` fails on the missing Unitree SDK, use an existing environment
   with `torch`, `mujoco` 3.11.0 and `numpy`; the training does not need the SDK.
4. Watch `output/robust_history_v1/curve.jsonl` (same lines as the terminal).

## Reading the training output

| Field | Meaning | Healthy |
| --- | --- | --- |
| `scale` | error size being trained on, as a fraction of the full `deploy` profile | rises from 0.005 toward 1.0 |
| `randomized_success` | success of noisy training episodes at that scale | hovers around 0.15–0.4 while the scale climbs |
| `validation` | noise-free successes on fixed seeds at scales 0, 0.03, 0.1, 0.3, 1.0 | scale 0 stays near 32/32; the others rise |
| `rolled_back` | the update was undone for moving the policy too far | rare after the first few updates |
| `approx_kl` | size of the policy change | around 0.01 |
| `noise_std` | exploration noise | starts at 0.003 |

Update 0 in the file is the untrained policy, for comparison. On the laptop
(8 cores, 32 episodes per update) one update took about 40 s. Default is 1000
updates.

**Stop rule:** if `scale` has not moved after about 200 updates, stop. The
approach is not working with these settings.

## After training

```bash
.venv/bin/python scripts/export_handstand_bundle.py --name robust_history_v1 \
  --checkpoint output/robust_history_v1/best.pt --foot-correction 0.3
.venv/bin/python scripts/handstand_randomized.py evaluate --deployed \
  --bundle logs/policies/robust_history_v1 --profiles deploy \
  --scales 0 0.1 0.3 1 --trials 100 --evaluation-seed 801000 \
  --output output/robust_history_v1/heldout
.venv/bin/python scripts/handstand_randomized.py sweep --deployed --trials 12 \
  --bundle logs/policies/robust_history_v1 --output output/robust_history_v1/sweep
```

Compare against the untrained numbers (16 trials, deploy stack):

| Error scale | 0 | 1% | 3% | 10% | 30% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Released controller | 16/16 | 10/16 | 6/16 | 1/16 | 0/16 |
| Same, balance limit ±0.2 (the training default) | 16/16 | 11/16 | 9/16 | 0/16 | 0/16 |

A result worth taking to hardware would hold most trials at scale 1.0 and
keep scale 0 intact.

## If the training does not work

Things to try, most promising first. None is implemented.

- **More authority or a different split.** `--bound` (default 0.2) limits the
  learned correction; the failures are forward tips during the first 2 s of
  the hold, so a balance-phase-only correction with a larger bound is a
  natural variant.
- **Exploration that is not white noise.** 1 mrad of per-step noise already
  drops the error-free case from 16/16 to 4/16. Smoother (time-correlated)
  exploration would disturb the handstand less for the same information.
- **Train the balance separately.** `go2balance` in the main framework learns
  a top hold on the GPU; a robust hold policy could take over at the 2.2 s
  hand-over in place of the released balance feedback.
- **Re-optimise the reference** with delay and gain error in the loop. The
  current one was optimised for the exact nominal model.

## What exists

Deployment (robot side, no MuJoCo needed):

- `go2_deploy/custom_implementation/handstand_capture.py`: the custom
  implementation; 47-channel observation, hooks, operator sequence.
- `go2_deploy/handstand/`: `kinematics.py` (COM and feet from joint angles),
  `estimator.py` (IMU integration, standing calibration, foot correction),
  `maneuver.py` (stages, phase, target filter), `policy.py`, `assets.py`.
- Framework edits, all optional and default-off: `GO2_CUSTOM_IMPL` selection,
  `on_lowstate`, `process_action`, `on_key` hooks, bundle fields for 50 Hz and
  no smoothing.

PC side:

- `scripts/export_handstand_bundle.py`: checkpoint to bundle, any of the
  three checkpoint kinds.
- `experiments/handstand_robustness/deploy_sil.py`: flies the framework's
  real control loop against the simulator.
- `experiments/handstand_robustness/randomization.py`: sensor and actuator
  error model and profiles.
- `scripts/handstand_randomized.py`: `compare`, `sweep`, `evaluate`, and
  `train` (the fine-tuning that did not help).
- `experiments/handstand_robustness/history_policy.py`,
  `history_training.py`, `scripts/handstand_robust_train.py`: the server
  training.

Tests: `tests/unit_tests/test_handstand_deploy.py`,
`test_handstand_randomization.py`, `test_handstand_history.py`. All 260 unit
tests passed on 2026-11-17. The artifact-dependent ones need:

```bash
export GO2_HANDSTAND_TRAJECTORY=$PWD/artifacts/handstand_reference/trajectory.npz
export GO2_HANDSTAND_SHA256=21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92
export GO2_HANDSTAND_CHECKPOINT=$PWD/artifacts/capture_ppo_latest.pt
.venv/bin/python -m pytest tests/unit_tests -q
```

Results on the laptop only (git-ignored `output/` and `logs/`):

- `output/deploy_compare_100/`: deploy stack against training, 100 seeds.
- `output/robust_sweep_native/`, `output/robust_sweep_handstand_capture_v1/`,
  `output/robust_sweep_handstand_capture_v1_feet/`: one-error-at-a-time sweeps.
- `output/robust_train_both002/`, `output/robust_heldout_*/`: the fine-tuning
  and its held-out comparison.
- `logs/policies/handstand_capture_v1/` and `..._feet/`: baseline bundles,
  without and with foot correction.

## Before any hardware attempt

Not done, in rough order:

1. A policy that survives realistic errors in simulation. The released one
   does not.
2. Check the IMU: quaternion frame and order, gyro and accelerometer axes,
   whether `tick` is milliseconds, real 500 Hz delivery. Log a stationary
   robot and a hand-tilted one through `go2_deploy` and compare with the
   observation layout in `handstand_capture.py`.
3. Check `foot_force` units and order. The foot correction assumes newtons
   and a 25 N threshold; the Go2 reports raw sensor values.
4. Check that the motor-side PD at kp 60, kd 3 behaves like the simulated
   one, including torque limits. The framework sends targets and gains only.
5. Dry run of the operator sequence with the robot suspended: A, wait for
   `ready`, do not press Y. Confirm it reaches and holds the start posture.
6. Add an automatic abort on tilt and joint speed. Today the only protections
   are the operator's stop and a fault on lost IMU data.

## Things that are easy to forget

- The training start state cannot be stood in; the bundle's `standing_target`
  exists for that. Re-export bundles after changing gains or the reference.
- The robot must stand still at least 5 s before `ready`. Calibrating after
  2 s lost every simulated handstand.
- The operator-sequence result of 100/100 is one start state repeated, not
  100 independent trials.
- Error sizes in the profiles are guesses for a Go2, not measurements.
- On the laptop `uv run` fails (Unitree SDK not fetched); use `.venv/bin/python`.
