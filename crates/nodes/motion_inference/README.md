# Motion inference contracts

The reference used for the September 2026 audit is
[B-Human MachineLearning at `5d54ac1441cd3270801782dfc90ee8bdfa752f1f`](https://github.com/bhuman/MachineLearning/tree/5d54ac1441cd3270801782dfc90ee8bdfa752f1f/IsaacGymRL/pre-trained),
particularly `WalkAndKick/WalkingEngine.cpp`, `GetUp/GetUpEngine.cpp`, and their
K1 configuration files. The local `bhuman-walking` sources provide additional
implementation context. These references differ in some kick tuning and
motor-provider behavior; model compatibility alone does not establish hardware
control equivalence.

## Timing and execution

Each service accepts `InferenceRequest<Command>` with a generation, request time,
and exclusive deadline. `motion` owns generations. Damping, Prepare, inference
failure, and transitions between locomotion and get-up invalidate the previous
execution. Walk, Stand, Kick, and SoftKick share locomotion history. Get-up
variants have separate executions. A stopped generation cannot be reopened by
a delayed request. Restarting an execution reseeds history from measurements.

The inference node checks deadlines before dispatch, at worker start, and on
completion. A rejected completion discards controller state and retained joint
targets. The caller checks again after inference and head motion finish, just
before publishing motor targets. Both nodes must be updated together because
the service request schema changed.

`timing.policy_period` controls minimum spacing between actual inference starts
and the caller's timer. Late cycles skip missed periods. Each policy update
requires a new sensor timestamp. Sensor reception and velocity estimation
continue while inference runs or waits for its next eligible start. Timing
telemetry reports actual worker starts, source sensor times, and generations.
The inference node owns the timing parameters and publishes its validated startup
settings on the retained `motion_inference/timing_parameters` topic. Motion binds
only its own parameter set and receives those settings from the owner. Changes to
inference timing require restarting both nodes.

## Policy behavior

- Walk frequency feedback clips to ±0.5 with the shipped parameters. Kick
  feedback retains the training-compatible ±2 range; phase frequency still
  limits the correction to ±0.5. The initial feedback value remains 1.5.
- Standing after fast movement or kicking requires one second below the
  configured slow-command thresholds. A nonzero command keeps walking active.
- Walking arms blend from a fixed measured entry pose over one second.
- Kick observations rotate ball position, velocity, and direction from the
  command's source epoch into the selected sensor frame. Source-stamped IMU yaw
  provides the relative rotation because odometry subtracts a fixed yaw offset.
  This correction excludes translation and ball prediction.
- SoftKick uses unshifted ball coordinates. Kick uses shifted coordinates;
  both representations retain their own previous sample across policy switches.
- SlowGetUp ankle-up gains include the reference's 50% stiffness: kp 25, kd 1.

FastGetUp still uses the existing position-PD output. Equivalence to B-Human's
motor-provider torque path remains unresolved and requires separate validation.

## Checks

Run `cargo test -p motion_inference -p motion --lib --locked` for regression tests
covering policy features, resets, deadline rejection, cadence, source timestamp
handling, yaw interpolation, and arm blending. These tests do not require ONNX
Runtime and do not execute the neural networks or validate robot stability.
