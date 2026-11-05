"""Paired first-episode DR comparisons, no production code/model overwrite."""
from pathlib import Path
import sys,json,hashlib,time,importlib,copy,os
import xml.etree.ElementTree as ET
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'mujoco_sim2sim/src'))
OLD=ROOT/'mujoco_sim2sim/models/trackstand_rl_0916_bike/trackstand_rl_0916.xml'
NEW=OLD.with_name('trackstand_rl_0916_measured_masses_20261105.xml')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 return v

def main():
 original=OLD.read_bytes();xml=ET.fromstring(original)
 masses={'front_wheel_link':.1107,'rear_wheel_link':.18028,'base_link':1.99502,'steering_link':.1581}
 # Preserve source formatting: replace only the exact four authored mass tokens.
 text=original.decode()
 for body in xml.iter('body'):
  if body.get('name') in masses:
   old=body.find('inertial').get('mass');token='mass="'+old+'"';assert text.count(token)==1
   text=text.replace(token,'mass="'+str(masses[body.get('name')])+'"')
 if NEW.exists():assert NEW.read_text()==text
 else:NEW.write_text(text)
 a=ET.fromstring(original);b=ET.fromstring(NEW.read_bytes())
 for x,y in zip(a.iter(),b.iter()):
  if x.tag=='inertial': y.set('mass',x.get('mass'))
 assert ET.tostring(a)==ET.tostring(b)
 models={'current':OLD,'measured':NEW}
 policies={'v1':ROOT/'deploy/source/trackstand_deploy/models/trackstand_rl_rsl_2026-10-21_11-18-29_model_2500.onnx','v2':ROOT/'mujoco_sim2sim/policies/trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx'}
 manifest={'seeds':list(range(42,106)),'episodes_per_case':64,'dr':True,'pure_survival':True,'horizon_s':30,'policy_hz':50,'inertia_change':False,'mass_kg':masses,'models':{k:{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for k,p in models.items()},'policies':{k:{'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for k,p in policies.items()}}
 (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2))
 for version,hz in [('v1',400),('v1',800),('v2',1200)]:
  pkg=importlib.import_module('newbike_mujoco'+('_v2' if version=='v2' else ''))
  groupclass=importlib.import_module(pkg.__name__+'.viewer_envs').ViewerEnvGroup
  policy=pkg.DeployedPolicy(onnx_path=policies[version])
  # Single-thread inference avoids a thread pool dominating these small networks.
  import onnxruntime as ort
  opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
  policy._ort_sess=ort.InferenceSession(str(policies[version]),sess_options=opt,providers=['CPUExecutionProvider'])
  previous=None
  for label,path in models.items():
   name=f'{version}_{hz}hz_{label}';out=HERE/name;out.mkdir(exist_ok=True)
   first,_=pkg.build_env_from_mjcf(path,42,enable_dr=True,physics_hz=hz)
   group=groupclass(first,64,42)
   for env in group.envs:env.enabled_terms={'time_out','fallen'}
   obs=group.reset();dr=[clean(env.dr) for env in group.envs]
   if previous is not None:assert dr==previous,'DR draws differ between mass configurations'
   previous=dr
   (out/'dr.json').write_text(json.dumps(dr,indent=2))
   actual=[{env.model.body(int(i)).name:float(env.model.body_mass[i]) for i in env._dr_body_ids} for env in group.envs]
   (out/'sampled_masses.json').write_text(json.dumps(actual,indent=2))
   active=np.ones(64,dtype=bool);records=[];t0=time.monotonic();print('START',name,flush=True)
   for step in range(1500):
    ids=np.flatnonzero(active)
    if not len(ids):break
    acts=policy.act_batch(obs[ids])
    for i,action in zip(ids,acts):
     env=group.envs[i];obs[i],done=env.step(action,observe_terminal=False)
     if done:
      st=env.stats;records.append({'env':int(i),'seed':42+int(i),'steps':int(st.length),'seconds':st.length*.02,'timeout':bool(st.timeout),'death':st.death,'flags':dict(st.terminations),'roll_max_rad':float(st.roll_max)})
      active[i]=False
    if (step+1)%250==0:print(name,'step',step+1,'remaining',sum(active),'wall_s',round(time.monotonic()-t0,1),flush=True)
   assert len(records)==64 and not active.any()
   records.sort(key=lambda x:x['env']);length=np.array([r['seconds'] for r in records]);timeouts=sum(r['timeout'] for r in records)
   summary={'version':version,'physics_hz':hz,'mass_configuration':label,'episodes':64,'timeouts':timeouts,'falls':64-timeouts,'timeout_rate':timeouts/64,'mean_seconds':float(length.mean()),'median_seconds':float(np.median(length)),'min_seconds':float(length.min()),'max_seconds':float(length.max()),'wall_seconds':time.monotonic()-t0}
   (out/'episodes.json').write_text(json.dumps(records,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2));print('RESULT',summary,flush=True)
 assert OLD.read_bytes()==original
 print('DONE original XML preserved',flush=True)
if __name__=='__main__':main()
