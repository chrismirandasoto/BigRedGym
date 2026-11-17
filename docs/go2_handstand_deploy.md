# Go2 handstand: deployment through `go2_deploy`, and robustness

Everything here is simulation. No robot was connected and no hardware command
was sent. The controller is the released `capture_residual_v1` checkpoint
(`artifacts/capture_ppo_latest.pt`, SHA256 `90117fc2…58ba6c`, 67/100 in
[handstand_reproduce.md](handstand_reproduce.md)); it is only ever read.

**Bottom line.** The deployment software is in place and reproduces training
in simulation. The released controller is not ready for a robot: in
simulation it loses the handstand with a 2 ms command delay, a 2% joint
stiffness error, or a 0.03° attitude error.

## Part 1: deployment through Cornell's framework

The framework is `go2_deploy` from `LampLighterLab/BigRedGym` branch
`fixdeploy` at `8ca9bf9`, copied into this branch. The handstand is a custom
implementation, [handstand_capture.py](../go2_deploy/custom_implementation/handstand_capture.py),
plus a NumPy/Torch package the robot can run without MuJoCo,
[go2_deploy/handstand/](../go2_deploy/handstand/).

### The four items

| Item | What was built | Check |
| --- | --- | --- |
| 47-channel observation | Order and scaling below, assembled by the framework's own `lowstate_to_obs` | Equals the training observation to 1.8e-7 over whole maneuvers |
| COM estimation | IMU integrated at 500 Hz in a new `on_lowstate` hook; COM from a rigid-body tree exported from the MuJoCo model | Tree matches MuJoCo to 1e-15 on random poses |
| Action scaling | 2 rad per unit, then the training target filter (joint-limit clip, slew at 0.7 × speed limit) in a new `process_action` hook; the filtered target is what is sent and what the policy sees next | Applied targets equal training's to 1.2e-7 |
| Phase-dependent control | One TorchScript module holding the 320-step tables; step = commands issued since the trigger; feedback law switches at step 110, last step held after 319 | Equals the checkpoint's actions to 4.1e-7 at steps 0–400 |

Observation, in order (47 values):

| Piece | Size | Value |
| --- | ---: | --- |
| `projected_gravity` | 3 | gravity direction in the body frame |
| `hs_gyro` | 3 | body angular velocity, rad/s, calibrated bias removed |
| `dof_pos_obs` | 12 | joint angle − nominal posture, rad |
| `dof_vel` | 12 | joint velocity ÷ `[30.1, 30.1, 15.7]` per leg |
| `dof_pos_target` | 12 | (previous applied target − nominal) ÷ 2 |
| `hs_elapsed` | 1 | commands issued × 0.02 ÷ 8 |
| `hs_com` | 4 | estimated COM height ÷ 0.6, COM velocity xyz |

Runtime settings travel in the bundle: 50 Hz, kp 60, kd 3, action smoothing
off. The framework defaults (100 Hz, kp 30, smoothing on) would be wrong.

### Changes to the framework

All are optional hooks or overrides; `go2trot_base.py` behaves as before.

- `GO2_CUSTOM_IMPL` selects the custom implementation file. A bundle records
  which one it needs and `deploy.py` selects it.
- `CustomDeployConfig` may set `ctrl_freq`, `kp`, `kd`, `exp_moving_avg`,
  `default_dof_pos`, `task_name`.
- New optional hooks: `on_lowstate` (every LowState message),
  `process_action` (bound the policy's offset before it is stored and sent),
  `on_key` (extra keyboard keys).
- Bundles now record `ctrl_freq` and `exp_moving_avg`, and loading applies
  them. Upstream `export_bundle` read `cfg.phase_frequency`, which no longer
  exists there; it is now optional.
- `tests/unit_tests/test_go2_deploy_obs.py` had the same stale
  `phase_frequency` reference and is fixed.

### Operator sequence

1. Export: `python scripts/export_handstand_bundle.py --name handstand_capture_v1`
   (PC; writes `logs/policies/handstand_capture_v1/`).
2. Deploy as usual with `--bundle logs/policies/handstand_capture_v1`.
3. **A** (or `c`) enters custom control. The robot moves at 0.5 rad/s to the
   start posture and stands.
4. After at least 5 s, once the body is still and the accelerometer average
   has stopped moving, the terminal prints `ready`.
5. **Y** (or `h`) starts the handstand. **X** or Enter is the emergency stop.

Lost or invalid IMU data during the maneuver requests the emergency stop.

### Two things the simulated deployment exposed

**The training start state cannot be stood in.** Training begins with the
joint target equal to the joint angles, so the motors carry no load at t = 0.
Holding that target with kp 60, the robot sags 0.12 rad at the knees and
2.4 cm in height. The bundle therefore carries a `standing_target` that holds
the training posture to 0.0002 rad; the robot stands on it and switches to
the training target at the trigger.

**Calibrating while still settling loses the handstand.** With 2 s of
standing, the accelerometer bias estimate absorbed 0.012 m/s² of settling
motion and 0/8 runs succeeded. Hence the 5 s minimum and the drift check.

### Deploy stack against the training policy

Seeds 301001–301100, the released start-state scatter, strict 5 s gate.
`python scripts/handstand_randomized.py compare --trials 100 --output output/deploy_compare_100`

| Variant | Successes | Same outcome as training policy |
| --- | ---: | ---: |
| Training policy (native) | 67/100 | — |
| Deploy stack, training start, training's start height | 67/100 | 100/100 |
| Deploy stack, training start, height from the feet | 74/100 | 93/100 |
| Same, accelerometer at the real IMU mount | 53/100 | 84/100 |
| Deploy stack, operator sequence | 100/100 | 67/100 |
| Operator sequence, accelerometer at the IMU mount | 100/100 | 67/100 |

The first deploy row is the software check: identical outcomes on every seed.
The next two rows differ from training only by estimate changes of a few
millimetres, yet outcomes flip on 7–16 seeds; the controller is that
sensitive. The operator-sequence rows are not 100 independent successes:
settling erases the start scatter, so they are close to one start state
repeated.

### Not done

- Nothing has run on a robot. IMU frame, foot-force units, motor-side PD
  behaviour and real 500 Hz delivery are unverified.
- The simulated IMU is ideal apart from the errors listed in Part 2.
- No automatic abort on tilt or joint speed; the operator's stop is the only
  protection besides the IMU-data fault.

## Part 2: robustness

Tools: [randomization.py](../experiments/handstand_robustness/randomization.py)
(error model), `scripts/handstand_randomized.py` (`sweep`, `evaluate`,
`train`). With no errors the randomized environment is bit-identical to the
released one, and the reference still replays with ≤1e-10 state error.
Error sizes are engineering guesses for a Go2, not measurements.

### One error at a time

Successes out of trials. "Training start" is the training policy from the
released start scatter (24 seeds). The other two columns fly the deploy stack
through the operator sequence (12 seeds), without and with the planted-foot
drift correction described below. With no added error: 17/24, 12/12, 12/12.

| Error | Size | Training start | Operator sequence | + foot correction |
| --- | --- | ---: | ---: | ---: |
| Gyro bias, rad/s | 0.002 | 13/24 | 12/12 | 12/12 |
| | 0.02 | 9/24 | 12/12 | 12/12 |
| Accelerometer bias, m/s² | 0.01 | 0/24 | 12/12 | 12/12 |
| | 0.3 | 0/24 | 12/12 | 12/12 |
| Standing-height error, m | 0.01 | 8/24 | 12/12 | 12/12 |
| Gyro noise, rad/s | 0.005 | 18/24 | 9/12 | 9/12 |
| | 0.02 | 6/24 | 3/12 | 3/12 |
| Accelerometer noise, m/s² | 0.05 | 6/24 | 4/12 | 9/12 |
| | 0.1 | 4/24 | 1/12 | 6/12 |
| Attitude error, rad | 0.002 | 0/24 | 0/12 | 0/12 |
| Encoder noise, rad | 0.0005 | 6/24 | 5/12 | 5/12 |
| Joint-velocity noise, rad/s | 0.05 | 12/24 | 7/12 | 6/12 |
| Joint zero offset, rad | 0.001 | 7/24 | 4/12 | 4/12 |
| | 0.003 | 4/24 | 1/12 | 1/12 |
| Joint stiffness kp | ±2% | 8/24 | 2/12 | 2/12 |
| | ±5% | 0/24 | 0/12 | 0/12 |
| Joint damping kd | ±5% | 9/24 | 5/12 | 5/12 |
| Available torque | 95–100% | 4/24 | 1/12 | 3/12 |
| Command delay | 2 ms | 0/24 | 0/12 | 0/12 |
| Extra joint friction, N·m·s/rad | 0.02 | 22/24 | 12/12 | 12/12 |
| Ground friction | ±10% | 8/24 | 4/12 | 4/12 |

Full tables (four sizes per error) are in `output/robust_sweep_*/sweep.json`.

- **Handled by the deployment procedure:** constant gyro bias, constant
  accelerometer bias and standing-height error. Standing calibration and
  feet-based height remove them at every size tried.
- **Not handled by anything yet:** attitude error, command delay, stiffness,
  torque, zero offsets and ground friction. These fail at sizes far below
  what a real Go2 will have.
- **Attitude error** fails even at 0.0005 rad (0.03°), in roll, pitch or yaw.
  It acts through two paths, each fatal alone at 0.002 rad of pitch: the
  velocity integrator, and the policy's gravity input.

### Planted-foot drift correction (optional, off by default)

The trained estimator integrates the accelerometer for the whole maneuver
with nothing to correct it. `--foot-correction 0.3` at export enables a
correction: while foot force sensors show a foot planted, leg kinematics give
the root height (within 1 mm along the nominal handstand) and, once the
handstand is held, the root velocity (within 2 mm/s). During the kick the
feet's velocity is wrong by up to 0.9 m/s, so velocity is corrected only from
step 110. It keeps the nominal 12/12, helps with accelerometer noise
(4/12 → 9/12 at 0.05 m/s²), and does not help with attitude error.

### Randomized training

`scripts/handstand_randomized.py train` fine-tunes with PPO on whole
maneuvers, each under its own drawn sensor and actuator errors, 64 episodes
per update on 8 worker processes. The policy's action law is unchanged and
update 0 is the released controller exactly; the balance feedback used after
2.2 s is made trainable alongside the per-step gains. Output goes to a new
directory, as checkpoint kind `capture_robust_v1`, which exports like the
baseline.

One run was made: the combined sensor + actuator profile at 2% of the assumed
sizes (the largest scale at which the baseline still sometimes succeeds),
20 updates, learning rate 1e-5, about 90 s per update.

**It did not improve robustness.** On 16 validation seeds the randomized
condition went from 3/16 to between 1/16 and 4/16 across updates. On 100
held-out seeds (501001–501100):

| Checkpoint | Released start scatter | + 2% sensor and actuator errors |
| --- | ---: | ---: |
| Released (`capture_ppo_latest.pt`) | 63/100 | 14/100 |
| Fine-tuned, best of 20 updates (update 6) | 64/100 | 13/100 |

(These two rows predate a change to how fractional command-delay ranges are
drawn, so rerunning gives slightly different draws.) The weights moved by at
most 0.0012. A longer run of the same kind may behave
differently; this one gives no reason to expect so. The earlier capture PPO
also found no gain over its starting point
([handstand_capture.md](handstand_capture.md)).

### What this means

- The released controller is a high-gain feedback law around one trajectory.
  Its margin to realistic sensor and actuator error is roughly a factor of
  100 too small (it survives the combined profile at 1% of the assumed sizes:
  10/24; at 3%: 0/24).
- Small-step fine-tuning of the existing gains does not widen that margin.
  The likelier routes are structural: a reference and feedback law designed
  with delay and gain error in the loop, an attitude-tolerant observation,
  or training from scratch under randomization rather than from this policy.
- The released baseline is untouched: same checkpoint hash, exact reference
  replay, 67/100 reproduced natively and through the deploy stack.

## Server training: a history-aware correction

The fine-tuning above could not help because it only nudged the released
gains. `scripts/handstand_robust_train.py` trains something structurally
different, and is meant for a machine with many CPU cores (no GPU needed).

- **Policy** (`capture_history_v1`): the released controller, frozen, plus a
  learned correction of up to ±0.2 action units (±0.4 rad) that also sees the
  previous four control periods. Untrained, it is the released controller
  exactly. The released balance feedback is also allowed ±0.2 instead of
  ±0.05, which is identical on the nominal maneuver.
- **Episodes** fly the deploy stack itself: operator sequence, standing
  calibration, robot-side estimator with foot correction, target filter. What
  is trained is what the bundle runs.
- **Errors** come from the `deploy` profile in `randomization.py` (for
  example attitude 0.01 rad, kp and kd ±10%, torque 90–100%, command delay
  0–6 ms). A curriculum starts at 0.5% of those sizes and multiplies by 1.15
  whenever 40% of the noisy training episodes succeed.
- **Safeguards** for this controller's sensitivity: exploration noise starts
  at about 1 mrad, the step size adapts to keep each update small, and an
  update that moves the policy too far is undone.

Measured starting point through the deploy stack (16 seeds, no training):

| Error scale | 0 | 1% | 3% | 10% | 30% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Released controller | 16/16 | 10/16 | 6/16 | 1/16 | 0/16 |
| Same, balance limit ±0.2 | 16/16 | 11/16 | 9/16 | 0/16 | 0/16 |

With 1 mrad of exploration noise and no errors, the released controller
succeeds 4/16; with the wider balance limit, 7/16. Failures are the robot
holding for 0.3–1.7 s and then tipping forward onto its head.

**Whether this training succeeds is not known.** Locally it has only been run
for 4 updates to check that it is stable, resumes, exports and flies through
the deploy stack.

```bash
# On the server, from the repository root, with artifacts/ in place:
uv sync --frozen --no-dev            # or use an existing environment
python scripts/handstand_robust_train.py --output output/robust_history_v1
# Stop any time with Ctrl+C; continue with:
python scripts/handstand_robust_train.py --output output/robust_history_v1 --resume
```

It uses one worker per core and four episodes per worker per update. On the
8-core development machine an update of 32 episodes takes about 40 s; more
cores give larger batches at about the same time per update. The default is
1000 updates, with a validation every 10.

What to watch in `output/robust_history_v1/curve.jsonl` (also printed):

- `scale`: the error size being trained on. This is the progress measure; it
  starts at 0.005 and the goal is 1.0.
- `validation`: successes of the noise-free policy on fixed seeds at scales
  0, 0.03, 0.1, 0.3 and 1.0. Update 0 is the untrained policy.
- `rolled_back`: should be rare after the first few updates.

If `scale` has not moved after about 200 updates, stop: the approach is not
working with these settings.

Afterwards, export the best checkpoint and test it like the baseline:

```bash
python scripts/export_handstand_bundle.py --name robust_history_v1 \
  --checkpoint output/robust_history_v1/best.pt --foot-correction 0.3
python scripts/handstand_randomized.py evaluate --deployed \
  --bundle logs/policies/robust_history_v1 --profiles deploy \
  --scales 0 0.1 0.3 1 --trials 100 --evaluation-seed 801000 \
  --output output/robust_history_v1/heldout
python scripts/handstand_randomized.py sweep --deployed --trials 12 \
  --bundle logs/policies/robust_history_v1 --output output/robust_history_v1/sweep
```

## Reproduce

```bash
export GO2_HANDSTAND_TRAJECTORY=$PWD/artifacts/handstand_reference/trajectory.npz
export GO2_HANDSTAND_SHA256=21d2c7ca8a9b55b27d1a58d4cc7fae17fdbea2dbedee8ecbacd56cc324e9bb92
export GO2_HANDSTAND_CHECKPOINT=$PWD/artifacts/capture_ppo_latest.pt
python -m pytest tests/unit_tests/test_handstand_deploy.py \
  tests/unit_tests/test_handstand_randomization.py tests/unit_tests/test_go2_deploy_obs.py -q

python scripts/export_handstand_bundle.py --name handstand_capture_v1
python scripts/handstand_randomized.py compare --trials 100 --output output/deploy_compare_100
python scripts/handstand_randomized.py sweep --trials 24 --output output/robust_sweep_native
python scripts/handstand_randomized.py sweep --deployed --trials 12 \
  --bundle logs/policies/handstand_capture_v1 --output output/robust_sweep_deployed
python scripts/handstand_randomized.py train --profile both --scale 0.02 \
  --updates 20 --output output/robust_train_both002
```

`artifacts/` comes from the release in [handstand_reproduce.md](handstand_reproduce.md).
