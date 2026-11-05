from pathlib import Path
from collections import defaultdict
import json
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import yaml
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
BAG=ROOT/'deploy/experiments/logs/rosbag2_2026_11_05-16_15_26'
d=defaultdict(list)
counts=defaultdict(int)
start=yaml.safe_load((BAG/'metadata.yaml').read_text())['rosbag2_bagfile_information']['starting_time']['nanoseconds_since_epoch']
with next(BAG.glob('*.mcap')).open('rb') as stream:
    for schema,c,record in make_reader(stream,validate_crcs=True).iter_messages():
        m=deserialize_message(record.data,get_message(schema.name)); t=(record.log_time-start)*1e-9
        counts[c.topic]+=1
        k=c.topic.rsplit('/',1)[-1]
        if hasattr(m,'data'):
            v=m.data
            if isinstance(v,str):
                try:v=json.loads(v)
                except ValueError:pass
            elif hasattr(v,'tolist'):v=v.tolist()
            row={'t':t,'data':v}
            if hasattr(m,'layout'):row['labels']=[x.label for x in m.layout.dim]
            d[k].append(row)
        elif k=='imu':
            q=m.orientation
            d[k].append({'t':t,'data':[q.x,q.y,q.z,q.w]})
        elif k=='rosout':d[k].append({'t':t,'name':m.name,'msg':m.msg,'level':m.level})
(HERE/'decoded.json').write_text(json.dumps(d,indent=2))
for k in ('rosout','policy_event','motor_event','kickstands_command'):
    print('\n',k)
    for row in d[k]:print(row)
print('AUTHORITY TRANSITIONS')
last=None
for row in d['kickstands_up']:
    if row['data']!=last:print(row);last=row['data']
print('COUNTS',{k:len(v) for k,v in d.items()})
print('FIRST MOTOR STATUS',d['motor_status'][0])
print('LAST MOTOR STATUS',d['motor_status'][-1])

(HERE/'counts.json').write_text(json.dumps(counts,indent=2))
