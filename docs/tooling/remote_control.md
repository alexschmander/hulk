# Remote Control

`./pepsi gammaray <robot>` disables the manufacturer controller services for HULK remote control.
`./pepsi boosterize <robot>` restores the `RemoteController` section in `/opt/booster/Daemon/bin/child.ini` and enables and starts `joystick_ros2`.

## Controls

Put the robot in `Initial`, then press Start on the controller to enable remote control.
Remote control cannot override `Damping`, `Prepare`, or `Stop`.

| Control | Action |
| --- | --- |
| Press Start | Toggle remote control on/off (starts off) |
| X / Square | Damping, including during manual standup |
| Press A / Cross | Start one manual standup when fall detection reports ready |
| Primary state changes | Disable remote control |
| Controller disconnected or input stale | Stand while remote control remains enabled |
| Left stick up/down | Walk forward/backward |
| Left stick left/right | Walk sideways |
| Right stick left/right | Turn the robot |
| D-pad left/right | Move the head left/right |
| D-pad up/down | Move the head up/down |
| Hold left shoulder, L1/LB | Rumpelstilzchen kick |
| Hold right shoulder, R1/RB | Schlong kick |
| Analog triggers, L2/LT and R2/RT | Unbound |
| Right stick up/down | Unbound |

## Manual standup

A / Cross starts one standup attempt when a valid `fall_detection/status` reports
`Fallen { ready_for_standup: true }`. Remote control does not need to be enabled.
There is no separate manual pose check or required sequence of primary states.
The detector owns pose readiness, settling, failure detection, and the standup timeout.

The controller must be connected with input no older than 250 ms. Release A / Cross before
pressing it; a held or rejected press is not queued for later. X / Square takes priority,
including when both buttons are pressed together. After a stop, release and press again.

An accepted press enters `ManualStandUp` and continues through detector `StandingUp`.
A valid `Upright` report returns to `Initial` with remote control disabled. Missing or
expired detector data, lost controller input, `Falling`, or `Fallen` without readiness
abort to `Damping`. Another attempt needs a new press and detector readiness again.
Ordinary game-controller updates and injected states cannot replace an active attempt;
stop requests still take effect.

## Restore Manufacturer Controls

Boosterize restores the `RemoteController` section in `/opt/booster/Daemon/bin/child.ini` and enables and starts `joystick_ros2`.
