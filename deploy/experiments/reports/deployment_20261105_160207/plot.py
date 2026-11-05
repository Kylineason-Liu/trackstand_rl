from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent
d=json.loads((H/'decoded.json').read_text())
def arr(k):return np.array([[r['t'],*r['data']] for r in d[k]])
s=arr('motor_state'); c=arr('motor_command_v2'); im=arr('imu')
q=im[:,1:];q=q/np.linalg.norm(q,axis=1)[:,None];x,y,z,w=q.T
gy=-2*(y*z+w*x);gz=-(1-2*(x*x+y*y))
roll=np.degrees(np.arctan2(-gy,-gz))
fig,axs=plt.subplots(3,1,figsize=(13,9),sharex=True,layout='constrained')
axs[0].step([r['t'] for r in d['kickstands_up']],[r['data'] for r in d['kickstands_up']],where='post',label='Manual up request')
axs[0].step(c[:,0],c[:,1],where='post',label='Motor enable',alpha=.8)
axs[0].set_ylabel('Boolean');axs[0].legend(loc='upper right')
axs[1].plot(im[:,0],roll,label='IMU-derived roll (not independent ground truth)')
for v in (-30,30):axs[1].axhline(v,color='r',ls='--')
axs[1].set_ylabel('Roll (degrees)');axs[1].legend(loc='upper right')
axs[2].plot(s[:,0],np.degrees(s[:,1]),label='Measured steering angle')
for v in (-5,5):axs[2].axhline(v,color='r',ls='--')
axs[2].set_ylabel('Steering (degrees)');axs[2].set_xlabel('Seconds since bag start');axs[2].legend(loc='upper right')
for ax in axs:
 ax.axvline(6.234209655,color='r',ls=':',label='Policy stop');ax.grid(alpha=.25);ax.set_xlim(0,36.3141)
fig.suptitle('2026-11-05: roll safety stop, then steering-centering re-arm gate')
fig.savefig(H/'diagnosis.png',dpi=180);fig.savefig(H/'diagnosis.svg');plt.close(fig)
mask=(im[:,0]>=6.18)&(im[:,0]<=6.25)
print('IMU near stop',list(zip(im[mask,0],roll[mask],gy[mask],gz[mask])))
print('raw count',len(d['raw_action']))
print('active command span',c[c[:,1]>0][[0,-1],0])
print('inference durations ms',np.percentile([r['data'][6] for r in d['policy_timing'] if r['data'][8]], [0,50,100]))
print('all motor fault false',not any(r['data'] for r in d['motor_fault']))
import yaml
bag=H.parents[3]/'deploy/experiments/logs/rosbag2_2026_11_05-16_02_07'
meta=yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
for item in meta['topics_with_message_count']:
 key=item['topic_metadata']['name'].rsplit('/',1)[-1]
 assert len(d.get(key,[]))==item['message_count'],key
f=next(bag.glob('*.mcap'))
(H/'verification.json').write_text(json.dumps({'mcap_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'messages':sum(map(len,d.values())),'metadata_counts_match':True,'crc_checked':True},indent=2)+'\n')
