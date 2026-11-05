# Manual kickstand control

`manual_kickstand` replaces `rc_kickstand`. It publishes the same
`std_msgs/msg/Bool` on `/trackstand/kickstands_up` (reliable, depth 10).
It starts **down** and repeats the selected state at 50 Hz, satisfying the
existing policy authority timeout while waiting for terminal input.

## Build and start

With ROS 2 sourced:

```bash
cd ~/trackstand_rl/deploy
colcon build --symlink-install --packages-select trackstand_deploy
source install/setup.bash
ros2 run trackstand_deploy manual_kickstand
```

Type `up` or `down`, then press Enter. Case and surrounding whitespace do not
matter. Invalid commands leave the selected state unchanged. `quit`, `exit`,
Ctrl-C, SIGTERM, SIGHUP and input EOF request down before exiting when ROS is
still available; delivery is best effort.

Run this node **instead of** `rc_kickstand`, never alongside it. Keep the policy
controller and `kickstand_driver` running: the existing flow remains
`kickstands_up` Bool → policy → `kickstands_command` String → servo driver.
The manual node has no application subscriptions and sends no direct servo or
motor commands.

## Complete v2 launch

In one sourced terminal:

```bash
ros2 launch trackstand_deploy trackstand_v2.launch.py use_rc:=false
```

In a second sourced terminal:

```bash
ros2 run trackstand_deploy manual_kickstand
```

The launch default remains `use_rc:=true`. The interactive node deliberately
runs in its own foreground terminal.

## Existing interlocks and limitations

Up remains subject to policy readiness and fault checks. A held up request may
arm when readiness becomes valid, as with the RC. Down disables balancing and
requests the existing down servo pose; v2 integral reset follows the existing
motor-disable path. Existing fault/re-arm requirements remain unchanged.

If the publisher disappears without delivering down, the existing policy
authority timeout disables motors but does **not** automatically lower the
kickstands. The Pi-local heartbeat does not prove SSH connection health until
the operating system reports EOF, hangup or an input error. Shutdown messages
are not physical servo acknowledgements. No policy, motor or servo logic was
changed by this terminal replacement.
