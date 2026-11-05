import json,collections,numpy as np
p='deploy/experiments/reports/deployment_20261105_161526/'
d=json.load(open(p+'decoded.json'))
prev=None; starts={}; ends={}
for r in d['policy_event']:
 v=r['data']; ep=v['episode']
 if v['mode']=='balancing':starts.setdefault(ep,r)
 if ep in starts and v['mode']!='balancing':ends.setdefault(ep,r)
for ep,s in starts.items():
 e=ends.get(ep);print('EP',ep,round(s['t'],3),round(e['t']-s['t'],3),e['data']['latched_fault_reasons'])
 ev=[r for r in d['motor_event'] if s['t']-.08<r['t']<e['t']+.06]
 for r in ev:
  v=r['data'];print(' MOTOR',round(r['t'],3),v['gate_enabled'],v['fault_reason'],v['disable_reasons'])
print('unique motor faults',collections.Counter(r['data']['fault_reason'] for r in d['motor_status']))
for key in ('policy_timing','motor_control'):
 names=d[key][0]['labels'][0].split(':',1)[1].split(',');a=np.array([r['data'] for r in d[key]])
 print(key)
 fields=['cycle_interval_ms','work_duration_ms','inference_ms','steering_send_duration_ms','drive_send_duration_ms','mit_actual_send_interval_ms','drive_mit_actual_send_interval_ms','steering_pair_age_ms','drive_pair_age_ms','imu_age_ms','motor_age_ms','command_age_ms']
 active=a[:,names.index('policy_enabled' if key=='policy_timing' else 'gate_enabled')]>0
 for f in fields:
  if f not in names:continue
  for mask,label in [(np.ones(len(a),dtype=bool),'all'),(active,'active')]:
   v=a[mask,names.index(f)];v=v[np.isfinite(v)];print(f,label,len(v),np.round(np.percentile(v,[50,95,99,100]),3) if len(v) else [])
 if key=='motor_control':
  for f in ['steering_send_attempted','steering_send_success','drive_send_attempted','drive_send_success']:
   print(f, sum(a[:,names.index(f)]==1))
print('errors')
for r in d['rosout']:
 if r['level']>=30:print(r)
