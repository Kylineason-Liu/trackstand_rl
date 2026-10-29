# Deployment audit verification — 2026-10-29

See the [audit findings and fixes](../../../docs/AUDIT_2026-10-29.md).
This is offline software validation, not a hardware experiment or proof of balancing.

- [Deployment suite](pytest.log): **649 passed, one skipped**; baseline was 584/1.
  The skipped smoke test needs a kernel socket option disallowed in this sandbox.
- [Isolated ROS 2 Humble package build](colcon_build.log): seven entry points,
  package data and models installed in temporary directories.
- [Real ROS message check](ros_message_check.json), [source](ros_message_check.py):
  actual ROS codecs and synthetic data; no Node/UART/CAN is opened. The unused
  vendor I2C import is stubbed, not tested. The sensor-freshness/codec checks do
  not establish Raspberry Pi timing or real sensor accuracy.
- [Verification/source hashes](verification.json).

From the repository root:

```bash
PYTHONPATH=deploy/source/trackstand_deploy PYTHONDONTWRITEBYTECODE=1 \
  python3 -m pytest -q -rs -p no:cacheprovider deploy/source/trackstand_deploy/test

# With the matching ROS distribution sourced (Humble on the audit computer):
source /opt/ros/humble/setup.bash
python3 -B deploy/experiments/reports/20261029_deployment_audit/ros_message_check.py
```

The isolated build used a copy of `deploy/source/trackstand_deploy` beneath
`/tmp/trackstand_deployment_audit_vnddz9vg/`, with separate `--build-base`,
`--install-base`, and `--log-base`. Production build/install directories were
not changed. All final Python inputs were synchronized to that copy; installed
symlink targets match current source hashes.
