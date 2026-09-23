BACKUP: the previous sensor-based deploy environment (2026-09-18)

What this is
------------
A faithful copy of the MuJoCo deploy environment as it was BEFORE the
qvel-direct refactor — i.e. the version where the trackstand deployment
used an imu site with gyro + velocimeter sensors, and BikeBalanceEnv.observe()
read the velocity channels from sensordata. The live environment was
refactored the same day: observe() now reads base_free qvel directly and the
augmentation adds no sensors/site (see cfg_record.txt, entries
"MUJOCO DIRECT STATE API + THE SENSOR-FREE REFACTOR" and its VERIFICATION).

Files
-----
  env.py                          previous src/newbike_mujoco/env.py:
                                  _imu_sid/_vel_sid in __init__, observe()
                                  reads sensordata (gyro + velocimeter),
                                  vel_b = R.T @ (velocimeter reading) — the
                                  double-rotation that the refactor fixed.
  run_deploy_viewer.py            previous scripts/run_deploy_viewer.py:
                                  augment_trackstand_model() adds the imu
                                  site (alpha-0 material) + gyro/vel sensors.
  .trackstand_deploy_augmented.xml
                                  the previous augmented scene (regenerated
                                  from the backup code above on
                                  models/trackstand_rl_0916_bike/
                                  trackstand_rl_0916.xml). The original
                                  generated file was deleted during the
                                  refactor; this is byte-for-byte the same
                                  augmentation output (the augmentation is
                                  deterministic).

To restore the previous environment
-----------------------------------
  cp env.py /home/lmbike/lmbike_cyclerl-mod/mujoco_sim2sim/src/newbike_mujoco/env.py
  cp run_deploy_viewer.py /home/lmbike/lmbike_cyclerl-mod/mujoco_sim2sim/scripts/run_deploy_viewer.py
  cp .trackstand_deploy_augmented.xml /home/lmbike/lmbike_cyclerl-mod/mujoco_sim2sim/models/trackstand_rl_0916_bike/

  (the viewer regenerates the augmented scene from the mjcf on every launch,
  so the XML copy is only needed to run the scene directly, e.g. a headless
  env on the old scene without re-running the augmentation.)

Note
----
The sensors were proven equivalent to qvel (gyro == qvel[3:6] exactly;
velocimeter == R^T . qvel[0:3] sensor-frame), so this backup exists for
regression/comparison, not because the sensor path was more correct — its
vel_b double-rotation is the bug the refactor fixed.
