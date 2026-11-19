"""Run the go2_deploy handstand stack against the training simulator.

The robot-side code path is exercised as it will run on hardware: MuJoCo
state is packed into Unitree LowState-shaped messages (Unitree joint order,
w-x-y-z quaternion, body-frame gyro and accelerometer, millisecond tick),
delivered to MainController._on_lowstate_msg every 2 ms, and
MainController._control_loop is called every 20 ms. The joint targets it
publishes in LowCmd drive the same motor model training used.

Only the Unitree SDK transport is replaced. This checks the deployment
software against the simulator; it says nothing about the real robot's
sensors, motors or contacts.
"""

import os
import sys
import types

import mujoco
import numpy as np

from gym.envs.go2.handstand_replay import CONTROL_DT, KD, KP, PHYSICS_DT

from experiments.handstand_robustness.randomization import (
    RandomizedEnvironment,
    _quaternion,
)

CUSTOM_IMPL = "handstand_capture.py"
POLICY_TO_UNITREE = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]  # its own inverse


class _Record:
    def __init__(self, **fields):
        self.__dict__.update(fields)


def _low_cmd():
    motors = [
        _Record(mode=0, q=0.0, kp=0.0, dq=0.0, kd=0.0, tau=0.0) for _ in range(20)
    ]
    return _Record(head=[0, 0], level_flag=0, gpio=0, motor_cmd=motors, crc=0)


def install_sdk_stubs():
    """Stand-ins for the Unitree SDK modules go2_deploy imports, if absent."""
    try:
        import unitree_sdk2py  # noqa: F401

        return False
    except ModuleNotFoundError:
        pass

    class Unused:
        def __init__(self, *args, **kwargs):
            pass

    class CRC:
        def Crc(self, message):
            return 0

    contents = {
        "unitree_sdk2py": {},
        "unitree_sdk2py.core": {},
        "unitree_sdk2py.core.channel": dict(
            ChannelSubscriber=Unused,
            ChannelPublisher=Unused,
            ChannelFactoryInitialize=lambda *a, **k: None,
            ChannelConfigHasInterface="",
        ),
        "unitree_sdk2py.idl": {},
        "unitree_sdk2py.idl.default": dict(unitree_go_msg_dds__LowCmd_=_low_cmd),
        "unitree_sdk2py.idl.unitree_go": {},
        "unitree_sdk2py.idl.unitree_go.msg": {},
        "unitree_sdk2py.idl.unitree_go.msg.dds_": dict(
            LowCmd_=Unused, LowState_=Unused, SportModeState_=Unused
        ),
        "unitree_sdk2py.go2": {},
        "unitree_sdk2py.go2.sport": {},
        "unitree_sdk2py.go2.sport.sport_client": dict(SportClient=Unused),
        "unitree_sdk2py.comm": {},
        "unitree_sdk2py.comm.motion_switcher": {},
        "unitree_sdk2py.comm.motion_switcher.motion_switcher_client": dict(
            MotionSwitcherClient=Unused
        ),
        "unitree_sdk2py.utils": {},
        "unitree_sdk2py.utils.crc": dict(CRC=CRC),
        "unitree_sdk2py.utils.future": dict(Future=Unused, FutureResult=Unused),
        "unitree_sdk2py.utils.timerfd": dict(
            timerfd_create=None,
            timerfd_settime=None,
            timerfd_gettime=None,
            itimerspec=Unused,
            timespec=Unused,
        ),
    }
    for name, attributes in contents.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, module)
    return True


class _Sink:
    """Receives what the controller would log or publish."""

    def __init__(self):
        self.lowcmd = None
        self.dropped_rows = 0
        self.run_dir = None

    def Write(self, lowcmd):
        self.lowcmd = lowcmd

    def log_lowstate(self, *args):
        pass

    def log_control(self, *args):
        pass


def make_controller(bundle_dir):
    """A real MainController wired to in-memory messages instead of DDS."""
    if os.environ.setdefault("GO2_CUSTOM_IMPL", CUSTOM_IMPL) != CUSTOM_IMPL:
        raise ValueError(f"GO2_CUSTOM_IMPL must be {CUSTOM_IMPL}")
    install_sdk_stubs()
    from go2_deploy.main_controller import MainController
    from go2_deploy.policy_bundle import load_bundle
    from go2_deploy.rl_controller import RLController
    from go2_deploy.state import State
    from go2_deploy.unitree_remote_controller import UnitreeRemoteController
    from go2_deploy.utility import deploy_utility
    from unitree_sdk2py.utils.crc import CRC

    policy, cfg = load_bundle(bundle_dir)
    controller = object.__new__(MainController)  # skip the DDS/sport-mode setup
    controller._state = State.RECOVERY
    controller._estop_flag = False
    controller.cfg = cfg
    controller.rl_controller = RLController(cfg, policy)
    controller.remote_controller = UnitreeRemoteController(cfg)
    controller.rc_handler = _Record(_process_input=lambda: None)
    sink = _Sink()
    controller.csv_logger = sink
    controller.lowcmd_publisher = sink
    controller.obs_vec_size = cfg.obs_dim()
    controller._init_buffers()
    controller._init_custom_buffers()
    controller.crc = CRC()
    controller.default_lowcmd = deploy_utility.default_lowcmd(cfg)
    controller.kp_mult = controller.kd_mult = 1.0
    return controller


def published_target(controller):
    """Joint targets of the last LowCmd, policy joint order."""
    motors = controller.lowcmd_publisher.lowcmd.motor_cmd
    return np.array([motors[i].q for i in POLICY_TO_UNITREE], dtype=float)


class DeployedEnvironment(RandomizedEnvironment):
    """Training environment whose sensors feed a deploy controller.

    accelerometer:
        "origin"  specific force of the base origin, the ideal sensor training
                  assumed
        "imu"     specific force where the Go2's IMU is mounted
    """

    def __init__(
        self,
        model,
        trajectory,
        controller,
        seed=0,
        condition=None,
        horizon=400,
        accelerometer="origin",
    ):
        self.controller = None  # no messages while the parent builds the state
        self._step_acceleration = np.zeros(model.nv)
        self._tick = 0
        self.solver_forces = np.zeros((4, 3))
        super().__init__(model, trajectory, seed, condition, horizon)
        self.accelerometer = accelerometer
        self.imu_position = model.body_pos[model.body("imu").id].copy()
        self.controller = controller

    def after_physics_step(self):
        super().after_physics_step()
        # mj_step leaves the acceleration that produced this step's velocity;
        # the mj_forward that follows overwrites it.
        self._step_acceleration = self.d.qacc.copy()

    def record_metrics(self):
        super().record_metrics()
        if self.controller is not None:
            self._tick += 2
            self.controller._on_lowstate_msg(self.message())

    def stand(self, target):
        """One control period at a fixed joint target, before the maneuver.

        Same motor model as Environment.step, but the maneuver's step count
        does not advance.
        """
        self.target = np.asarray(target, dtype=float).copy()
        kp = KP * self.condition.get("kp_scale", 1.0)
        kd = KD * self.condition.get("kd_scale", 1.0)
        effort = self.effort * self.condition.get("strength", 1.0)
        for _ in range(round(CONTROL_DT / PHYSICS_DT)):
            velocity = self.d.qvel[self.vids].copy()
            error = self.target - self.d.qpos[self.qids] - self.joint_zero_offset
            torque = np.clip(kp * error - kd * velocity, -effort, effort)
            torque[(abs(velocity) >= self.speed) & (torque * velocity > 0)] = 0
            self.d.qfrc_applied[:] = 0
            self.d.qfrc_applied[self.vids] = torque
            mujoco.mj_step(self.m, self.d)
            self.after_physics_step()
            mujoco.mj_forward(self.m, self.d)
            self.measure(velocity)
            self.record_metrics()

    def message(self):
        """The current state as the robot would report it, errors included."""
        d = self.d
        errors = self.errors
        rotation = d.xmat[self.base].reshape(3, 3)
        rate = d.qvel[3:6]
        force = rotation.T @ (self._step_acceleration[:3] - self.m.opt.gravity)
        if self.accelerometer == "imu":
            r = self.imu_position
            alpha = self._step_acceleration[3:6]
            force = force + np.cross(alpha, r) + np.cross(rate, np.cross(rate, r))
        elif self.accelerometer != "origin":
            raise ValueError(self.accelerometer)
        quaternion = (
            _quaternion(errors.rotation(rotation)) if errors.active else d.qpos[3:7]
        )
        q, dq = errors.joints(d.qpos[self.qids], d.qvel[self.vids])
        motors = [
            _Record(q=float(q[i]), dq=float(dq[i]), ddq=0.0, tau_est=0.0)
            for i in POLICY_TO_UNITREE
        ]
        return _Record(
            tick=self._tick,
            imu_state=_Record(
                quaternion=np.asarray(quaternion).tolist(),
                gyroscope=np.asarray(errors.gyro(rate)).tolist(),
                accelerometer=np.asarray(errors.accel(force)).tolist(),
            ),
            motor_state=motors,
            # Vertical foot loads in newtons, Unitree order FR, FL, RR, RL.
            foot_force=[float(self.solver_forces[i, 2]) for i in (1, 0, 3, 2)],
            wireless_remote=bytes(40),
        )


def run_deployed(env, steps=400, start="direct", height="feet", compare=True):
    """Fly the maneuver through the deploy controller.

    start:
        "direct"   begin the maneuver on the first control period, as training
                   does; IMU biases are not calibrated
        "prepare"  the operator sequence: enter custom control, stand until the
                   controller reports ready (biases calibrated), then trigger
    height (direct start only):
        "feet"     initial root height from the feet, as on the robot
        "training" the constant training used, for exact comparison

    Returns the environment's result plus, when `compare` is set, the largest
    difference between the deploy observation and the training observation and
    between the applied joint targets.
    """
    from go2_deploy.handstand.maneuver import Stage
    from go2_deploy.state import State

    controller = env.controller
    custom = sys.modules["go2_deploy.main_controller"].custom_impl
    first = env.message()
    controller._on_lowstate_msg(first)
    maneuver = custom._maneuver(controller)
    if env.accelerometer == "origin":
        maneuver.estimator.imu_offset = np.zeros(3)
    compare = compare and not env.errors.active and start == "direct"
    controller._state = State.CUSTOM_CTRL
    prepare_steps = 0
    if start == "direct":
        q, _ = custom._joints(first)
        _, _, quat = custom._imu(first)
        maneuver.start_now(
            q,
            quat,
            height=env.imu_observer.height if height == "training" else None,
        )
        if height != "training":
            maneuver.estimator.height += env.errors.height_error
    elif start == "prepare":
        while maneuver.stage != Stage.RUN:
            if maneuver.stage == Stage.READY:
                custom.on_key(controller, "h")
            controller._control_loop()
            env.stand(published_target(controller))
            prepare_steps += 1
            if controller._estop_flag or prepare_steps > 1000:
                break
    else:
        raise ValueError(start)
    nominal = env.trajectory["nominal"]
    observation_error = target_error = 0.0
    for _ in range(steps):
        if controller._estop_flag or maneuver.stage != Stage.RUN:
            break
        expected = env.observation() if compare else None
        controller._control_loop()
        if controller._estop_flag:
            break
        if compare:
            # The simulator gives the 47-value capture observation; a hold
            # bundle appends past frames (hs_history) after it.
            observed = controller.last_obs.numpy()[: expected.size]
            observation_error = max(
                observation_error, float(abs(observed - expected).max())
            )
        target = published_target(controller)
        _, _, done = env.step((target - nominal) / 2)
        target_error = max(target_error, float(abs(env.target - target).max()))
        if done:
            break
    result = env.result()
    result.update(
        emergency_stop=bool(controller._estop_flag),
        started=maneuver.stage == Stage.RUN,
        prepare_seconds=prepare_steps * 0.02,
        gyro_bias_estimate=maneuver.estimator.gyro_bias.tolist(),
        accel_bias_estimate=maneuver.estimator.accel_bias.tolist(),
        observation_error=observation_error if compare else None,
        target_error=target_error,
    )
    return result


def run_native(env, policy, steps=400):
    """The same maneuver through the training-side policy, for comparison."""
    import torch

    for _ in range(steps):
        with torch.no_grad():
            action = policy(torch.tensor(env.observation())[None])[0].numpy()
        _, _, done = env.step(action)
        if done:
            break
    return env.result()


def state_vector(env):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.m, spec))
    mujoco.mj_getState(env.m, env.d, state, spec)
    return state
