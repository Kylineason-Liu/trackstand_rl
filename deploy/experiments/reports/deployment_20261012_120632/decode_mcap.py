"""Read recoverable MCAP messages without modifying the recording."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from mcap.reader import NonSeekingReader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--bag', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
BAG, OUT = args.bag.resolve(), args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
numeric, texts, schemas, classes = defaultdict(list), defaultdict(list), {}, {}
counts = Counter()
issues = []
first_ns = last_ns = None

def stamp(s):
    return s.sec + s.nanosec * 1e-9

def parameters(ps):
    names = {1:'bool_value', 2:'integer_value', 3:'double_value', 4:'string_value',
             5:'byte_array_value', 6:'bool_array_value', 7:'integer_array_value',
             8:'double_array_value', 9:'string_array_value'}
    return {p.name: getattr(p.value, names[p.value.type]) if p.value.type in names else None for p in ps}

with BAG.open('rb') as source:
    reader = NonSeekingReader(source, validate_crcs=True)
    try:
        for schema, channel, record in reader.iter_messages(log_time_order=False):
            counts[channel.topic] += 1
            if first_ns is None:
                first_ns = record.log_time
            last_ns = record.log_time
            t = (record.log_time - first_ns) * 1e-9
            key = channel.topic.rsplit('/', 1)[-1]
            if schema.name not in classes:
                classes[schema.name] = get_message(schema.name)
            try:
                m = deserialize_message(record.data, classes[schema.name])
                if schema.name.endswith('MultiArray'):
                    if m.layout.dim:
                        label = m.layout.dim[0].label
                        if ':' in label:
                            fields = label.split(':',1)[1].split(',')
                            assert len(fields) == len(m.data)
                            assert schemas.get(key,fields) == fields
                            schemas[key] = fields
                    numeric[key].append([t, *m.data])
                elif schema.name == 'std_msgs/msg/Bool':
                    numeric[key].append([t, float(m.data)])
                elif schema.name == 'sensor_msgs/msg/Imu':
                    q, w, a = m.orientation, m.angular_velocity, m.linear_acceleration
                    numeric[key].append([t,stamp(m.header.stamp),q.x,q.y,q.z,q.w,w.x,w.y,w.z,a.x,a.y,a.z])
                elif schema.name == 'std_msgs/msg/String':
                    try:
                        value = json.loads(m.data)
                    except json.JSONDecodeError:
                        value = {'text':m.data}
                    texts[key].append({'t':t, **value})
                elif schema.name == 'rcl_interfaces/msg/Log':
                    texts[key].append({'t':t,'stamp_s':stamp(m.stamp),'level':m.level,'node':m.name,
                                       'message':m.msg,'file':m.file,'function':m.function,'line':m.line})
                elif schema.name == 'rcl_interfaces/msg/ParameterEvent':
                    texts[key].append({'t':t,'node':m.node,'new':parameters(m.new_parameters),
                                       'changed':parameters(m.changed_parameters),'deleted':parameters(m.deleted_parameters)})
                else:
                    issues.append({'t':t,'unhandled':channel.topic,'schema':schema.name})
            except Exception as error:
                issues.append({'t':t,'topic':channel.topic,'decode_error':repr(error)})
    except Exception as error:
        issues.append({'stream_error':repr(error),'file_position':source.tell()})

arrays = {key:np.asarray(values,dtype=float) for key,values in numeric.items()}
np.savez_compressed(OUT/'data.npz',**arrays)
details = {'bag':str(BAG),'bytes':BAG.stat().st_size,'sha256':hashlib.sha256(BAG.read_bytes()).hexdigest(),
           'first_ns':first_ns,'last_ns':last_ns,'duration_s':(last_ns-first_ns)*1e-9,
           'counts':dict(counts),'schemas':schemas,'text':dict(texts),'issues':issues}
(OUT/'details.json').write_text(json.dumps(details,indent=2))
print(json.dumps({k:v for k,v in details.items() if k not in ('text','schemas')},indent=2))
print('shapes', {k:v.shape for k,v in arrays.items()})
