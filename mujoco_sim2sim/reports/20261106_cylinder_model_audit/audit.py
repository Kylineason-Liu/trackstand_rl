"""Independent read-only check of cylinder variant geometry and preserved plant."""
from pathlib import Path
import hashlib, json, math, sys, subprocess
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'mujoco_sim2sim/src'))
from newbike_mujoco.model import load_trackstand_model as v1
from newbike_mujoco_v2.model import load_trackstand_model as v2
old=ROOT/'mujoco_sim2sim/models/trackstand_rl_0916_bike/trackstand_rl_0916.xml'
new=ROOT/'mujoco_sim2sim/models/trackstand_rl_0916_bike_cylinder/trackstand_rl_0916.xml'
assert old.read_bytes()==subprocess.check_output(['git','show','HEAD:'+str(old.relative_to(ROOT))],cwd=ROOT)
results={'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'mujoco_version':mujoco.__version__,'new_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'original_matches_HEAD':True,'loaders':{}}
for name,load in [('v1',v1),('v2',v2)]:
 a,b=load(old),load(new)
 fields=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_pos','jnt_axis','jnt_range','jnt_limited','geom_contype','geom_conaffinity','geom_friction','geom_solref','geom_solimp']
 for field in fields: assert np.array_equal(getattr(a,field),getattr(b,field)),field
 da,db=mujoco.MjData(a),mujoco.MjData(b)
 mujoco.mj_forward(a,da);mujoco.mj_forward(b,db)
 checks={}
 for stem,joint,offset,roll,yaw in [('rear','rearwheel_joint',[0,-.018,0],-1.572371,.013264),('front','frontwheel_joint',[0,0,0],1.570911,.004721)]:
  body=stem+'_wheel_link';gid=b.geom(body+'_'+body+'_collision00_geom').id;bid=b.body(body).id;jid=a.joint(joint).id
  axis=np.array([math.sin(yaw)*math.sin(roll),-math.cos(yaw)*math.sin(roll),math.cos(roll)])
  center=da.xanchor[jid]+offset
  err=float(np.max(np.abs(db.geom_xpos[gid]-center)));axiserr=float(np.max(np.abs(db.geom_xmat[gid].reshape(3,3)[:,2]-axis)))
  assert err<1e-12 and axiserr<1e-12,(err,axiserr)
  assert b.geom_type[gid]==mujoco.mjtGeom.mjGEOM_CYLINDER
  assert np.allclose(b.geom_size[gid,:2],[.07,.0005],atol=0,rtol=1e-15)
  checks[stem]={'center_max_abs_error_m':err,'axis_max_abs_error':axiserr,'center_world_m':center.tolist()}
 results['loaders'][name]={'preserved_arrays':fields,'total_mass_kg':float(b.body_mass.sum()),'wheel_checks':checks}
# Direct XML compilation without loader (all mesh references must resolve).
raw=mujoco.MjModel.from_xml_path(str(new));results['direct_xml_compiles']=True
out=Path(__file__).with_name('audit.json');out.write_text(json.dumps(results,indent=2)+'\n');print(out)
