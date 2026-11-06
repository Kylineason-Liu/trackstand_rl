#!/usr/bin/env bash
# Supported bike; only the diagnostic writes CAN. Existing ROS environment required.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
duration="${1:-120}"
output="${2:-$script_dir/logs/$(date +%Y%m%d_%H%M%S)_latency_ab}"
command -v ros2 >/dev/null
python3 -c "import rclpy"
python3 "$script_dir/can_latency_probe.py" --duration "$duration"
mkdir -p "$output"
observer_pid=""
recorder_pid=""
cleanup() {
    if [[ -n "$recorder_pid" ]]; then kill -INT "$recorder_pid" 2>/dev/null || true; wait "$recorder_pid" || true; recorder_pid=""; fi
    if [[ -n "$observer_pid" ]]; then kill -INT "$observer_pid" 2>/dev/null || true; wait "$observer_pid" || true; observer_pid=""; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
# ABBA reduces a simple warm-up/time-order confound. Observer/publishers stay
# present in both conditions; only rosbag recording changes.
for trial in 1_minimal 2_full 3_full 4_minimal; do
    trial_dir="$output/$trial"
    mkdir -p "$trial_dir"
    printf "INCOMPLETE\n" > "$trial_dir/trial_status.txt"
    python3 "$script_dir/ros_latency_observer.py" --source probe --duration 86400 --output-dir "$trial_dir" > "$trial_dir/observer.log" 2>&1 &
    observer_pid=$!
    if [[ "$trial" == *full ]]; then
        ros2 bag record -o "$trial_dir/bag" \
          /trackstand/latency_probe/state /trackstand/latency_probe/control \
          /trackstand/latency_probe/status /trackstand/latency_probe/fault \
          /trackstand/imu /rosout > "$trial_dir/recorder.log" 2>&1 &
        recorder_pid=$!
    fi
    sleep 3
    kill -0 "$observer_pid" # Fail before CAN if observer did not start.
    if [[ -n "$recorder_pid" ]]; then kill -0 "$recorder_pid"; fi
    python3 "$script_dir/can_latency_probe.py" --execute --mode disabled --ros \
      --duration "$duration" --label "$trial" --output-dir "$trial_dir"
    kill -0 "$observer_pid"
    if [[ -n "$recorder_pid" ]]; then kill -0 "$recorder_pid"; fi
    cleanup
    printf "COMPLETE\n" > "$trial_dir/trial_status.txt"
    sleep 2
 done
printf 'Finished. Copy the whole directory for analysis: %s\n' "$output"
