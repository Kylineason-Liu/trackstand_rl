"""Decode recorded Jazzy standard messages using MCAP and local ROS serializers.

Run after sourcing ROS. This reads bags only; caches are written separately.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


BAGS = {
    "A": "rosbag2_2026_10_01-16_11_49",
    "B": "rosbag2_2026_10_01-17_18_11",
    "C": "rosbag2_2026_10_01-17_32_00",
}


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def parameters(values):
    attributes = {
        1: "bool_value", 2: "integer_value", 3: "double_value", 4: "string_value",
        5: "byte_array_value", 6: "bool_array_value", 7: "integer_array_value",
        8: "double_array_value", 9: "string_array_value",
    }
    result = {}
    for parameter in values:
        attribute = attributes.get(parameter.value.type)
        value = getattr(parameter.value, attribute) if attribute else None
        if attribute and "array" in attribute:
            value = list(value)
        result[parameter.name] = value
    return result


def decode(stage, directory, cache_dir):
    metadata = yaml.safe_load((directory / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    start_ns = metadata["starting_time"]["nanoseconds_since_epoch"]
    numeric = defaultdict(list)
    text = defaultdict(list)
    schemas = {}
    classes = {}
    counts = Counter()
    files = []
    for relative in metadata["relative_file_paths"]:
        path = directory / relative
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
        files.append({"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size})
        with path.open("rb") as source:
            reader = make_reader(source, validate_crcs=True)
            for schema, channel, record in reader.iter_messages(log_time_order=True):
                topic = channel.topic
                counts[topic] += 1
                if schema.name not in classes:
                    classes[schema.name] = get_message(schema.name)
                message = deserialize_message(record.data, classes[schema.name])
                elapsed_s = (record.log_time - start_ns) * 1e-9
                key = topic.rsplit("/", 1)[-1]
                if schema.name in ("std_msgs/msg/Float32MultiArray", "std_msgs/msg/Float64MultiArray"):
                    values = list(message.data)
                    if key in ("motor_control", "policy_timing"):
                        label = message.layout.dim[0].label
                        fields = label.split(":", 1)[1].split(",")
                        assert len(fields) == len(values), (topic, label, len(values))
                        assert schemas.get(key, fields) == fields, (stage, topic, "schema changed")
                        schemas[key] = fields
                    numeric[key].append([elapsed_s, *values])
                elif schema.name == "std_msgs/msg/Bool":
                    numeric[key].append([elapsed_s, int(message.data)])
                elif schema.name == "sensor_msgs/msg/Imu":
                    q, g, a = message.orientation, message.angular_velocity, message.linear_acceleration
                    numeric[key].append([
                        elapsed_s, stamp_seconds(message.header.stamp), q.x, q.y, q.z, q.w,
                        g.x, g.y, g.z, a.x, a.y, a.z,
                        message.orientation_covariance[0], message.angular_velocity_covariance[0],
                        message.linear_acceleration_covariance[0],
                    ])
                elif schema.name == "sensor_msgs/msg/JointState":
                    assert len(message.name) == len(message.effort) == 1, message
                    assert message.name[0] in ("steering_joint", "rear_wheel_joint"), message.name
                    numeric[key].append([
                        elapsed_s, stamp_seconds(message.header.stamp),
                        int(message.name[0] == "rear_wheel_joint"), message.effort[0],
                    ])
                elif schema.name == "std_msgs/msg/String":
                    try:
                        value = json.loads(message.data)
                    except json.JSONDecodeError:
                        value = {"text": message.data}
                    text[key].append({"t": elapsed_s, **value})
                elif schema.name == "rcl_interfaces/msg/Log":
                    text[key].append({
                        "t": elapsed_s, "stamp_s": stamp_seconds(message.stamp), "level": message.level,
                        "node": message.name, "message": message.msg, "file": message.file,
                        "function": message.function, "line": message.line,
                    })
                elif schema.name == "rcl_interfaces/msg/ParameterEvent":
                    text[key].append({
                        "t": elapsed_s, "node": message.node,
                        "new": parameters(message.new_parameters),
                        "changed": parameters(message.changed_parameters),
                        "deleted": [p.name for p in message.deleted_parameters],
                    })
                else:
                    raise ValueError(f"unhandled nonempty topic {topic}: {schema.name}")
    expected = {
        item["topic_metadata"]["name"]: item["message_count"]
        for item in metadata["topics_with_message_count"]
    }
    count_differences = {
        topic: {"metadata": count, "decoded": counts.get(topic, 0)}
        for topic, count in expected.items() if counts.get(topic, 0) != count
    }
    if count_differences:
        print(stage, "metadata/physical-record count discrepancy:", count_differences, flush=True)
    arrays = {key: np.asarray(rows, dtype=np.float64) for key, rows in numeric.items()}
    np.savez_compressed(cache_dir / f"{stage}.npz", **arrays)
    details = {
        "stage": stage, "start_ns": start_ns, "duration_s": metadata["duration"]["nanoseconds"] * 1e-9,
        "counts": dict(counts), "metadata_counts": expected, "count_differences": count_differences,
        "schemas": schemas, "text": dict(text), "files": files,
    }
    (cache_dir / f"{stage}.json").write_text(json.dumps(details, indent=2))
    print(stage, "decoded", sum(counts.values()), "messages; shapes", {k: v.shape for k, v in arrays.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for stage, directory in BAGS.items():
        decode(stage, args.repository / "rosbags" / directory, args.cache_dir)


if __name__ == "__main__":
    main()
