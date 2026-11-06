"""Reproduce the cylinder variant and verify geometric/inertial preservation."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
def rpy_matrix(values):
    cx,cy,cz=np.cos(values); sx,sy,sz=np.sin(values)
    return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]) @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]) @ np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'mujoco_sim2sim/models/trackstand_rl_0916_bike/trackstand_rl_0916.xml'
NEW = ROOT / 'mujoco_sim2sim/models/trackstand_rl_0916_bike_cylinder/trackstand_rl_0916.xml'
URDF = ROOT / 'isaaclab_train/trackstand_rl_0916_description_cylinder/urdf/trackstand_rl_0916.urdf'
OUT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
original_sha = sha(OLD)
m = mujoco.MjModel.from_xml_path(str(OLD))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
u = ET.parse(URDF).getroot()
# URDF zero-pose link transforms relative to base_link, derived from joint origins.
poses = {'base_link': (np.zeros(3), np.eye(3))}
while len(poses) < len(u.findall('link')):
    progress = False
    for j in u.findall('joint'):
        parent, child = j.find('parent').get('link'), j.find('child').get('link')
        if parent in poses and child not in poses:
            o = j.find('origin')
            p, r = poses[parent]
            poses[child] = (p+r@np.fromstring(o.get('xyz'), sep=' '), r@rpy_matrix(np.fromstring(o.get('rpy'),sep=' ')))
            progress = True
    assert progress, 'URDF hierarchy incomplete'
text = OLD.read_text()
text = text.replace('model="trackstand_rl_0916_bike"','model="trackstand_rl_0916_bike_cylinder"')
# Keep the variant standalone; production loaders set meshdir to the XML parent.
NEW.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(OLD.parent / 'meshes', NEW.parent / 'meshes', dirs_exist_ok=True)
for mesh in ET.fromstring(text).findall('asset/mesh'):
    old = mesh.get('file')
    text = text.replace('file="'+old+'"', 'file="'+old.replace('\\','/')+'"')
checks = {}
for body in ('rear_wheel_link','front_wheel_link'):
    link = u.find(f"link[@name='{body}_1']")
    col = link.find('collision'); orig = col.find('origin'); cyl = col.find('geometry/cylinder')
    p_u, r_u = poses[body+'_1']
    b, j = m.body(body).id, m.joint(body).id
    # Common physical frame verified using axle anchors and joint directions.
    root_p = d.xpos[m.body('model_root').id]
    anchor_error = np.max(np.abs(d.xanchor[j]-(root_p+p_u)))
    assert anchor_error < 1e-6
    center_world = root_p + p_u + r_u@np.fromstring(orig.get('xyz'),sep=' ')
    rotation_world = r_u@rpy_matrix(np.fromstring(orig.get('rpy'),sep=' '))
    rb = d.xmat[b].reshape(3,3)
    center = rb.T@(center_world-d.xpos[b]); rotation = rb.T@rotation_world
    quat = np.zeros(4); mujoco.mju_mat2Quat(quat,rotation.reshape(9))
    geom_name = body+'_'+body+'_collision00_geom'
    old = ET.fromstring(text).find(f".//geom[@name='{geom_name}']")
    replacement = ET.Element('geom', {k:v for k,v in old.attrib.items() if k not in ('mesh','type','pos','quat')})
    replacement.set('type','cylinder')
    replacement.set('size', f"{float(cyl.get('radius')):.17g} {float(cyl.get('length'))/2:.17g}")
    replacement.set('pos',' '.join(f'{v:.17g}' for v in center))
    replacement.set('quat',' '.join(f'{v:.17g}' for v in quat))
    original_line = next(line.strip() for line in text.splitlines() if f'name="{geom_name}"' in line)
    text = text.replace(original_line,ET.tostring(replacement,encoding='unicode'))
    checks[body] = dict(center_body_m=center.tolist(), center_world_m=center_world.tolist(), axis_world=rotation_world[:,2].tolist(), quat_wxyz=quat.tolist(), axle_anchor_error_m=float(anchor_error))
NEW.write_text(text)
n = mujoco.MjModel.from_xml_path(str(NEW)); nd=mujoco.MjData(n);mujoco.mj_forward(n,nd)
fields = ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_pos','jnt_axis','jnt_range','jnt_limited')
for field in fields:
    assert np.array_equal(getattr(m,field),getattr(n,field)), field
for body, row in checks.items():
    g=n.geom(body+'_'+body+'_collision00_geom').id
    assert n.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert np.allclose(n.geom_size[g,:2],[.07,.0005],rtol=0,atol=1e-15)
    assert np.allclose(nd.geom_xpos[g],row['center_world_m'],rtol=0,atol=1e-12)
    assert np.allclose(nd.geom_xmat[g].reshape(3,3)[:,2],row['axis_world'],rtol=0,atol=1e-12)
    row['axis_abs_dot_joint'] = float(abs(nd.geom_xmat[g].reshape(3,3)[:,2]@nd.xaxis[n.joint(body).id]))
    assert row['axis_abs_dot_joint'] > .9999999
assert sha(OLD)==original_sha
summary=dict(original_sha256=original_sha,new_sha256=sha(NEW),urdf_sha256=sha(URDF),mujoco_version=mujoco.__version__,total_mass_kg=float(n.body_mass.sum()),unchanged_arrays=list(fields),wheel_checks=checks)
(OUT/'geometry_verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
