"""Independently validate a plain, portable mesh-wheel URDF without Isaac startup."""
from pathlib import Path
import hashlib,json,subprocess,xml.etree.ElementTree as ET
import shutil,tempfile,mujoco
ROOT=Path(__file__).resolve().parents[3]
newdir=ROOT/'isaaclab_train/trackstand_rl_0916_description_mesh'
new=newdir/'urdf/trackstand_rl_0916.urdf'
old=ROOT/'isaaclab_train/trackstand_rl_0916_description_cylinder/urdf/trackstand_rl_0916.urdf'
mesh=ROOT/'isaaclab_train/trackstand_rl_0916_description/urdf/trackstand_rl_0916.urdf'
def norm(el,paths=False):
 attrs=dict(el.attrib)
 if paths and 'filename' in attrs: attrs['filename']=Path(attrs['filename']).name
 return [el.tag,sorted(attrs.items()),(el.text or '').strip(),[norm(c,paths) for c in el]]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a,b,c=ET.parse(old).getroot(),ET.parse(new).getroot(),ET.parse(mesh).getroot()
text=new.read_text()
assert '0914' not in text and 'xacro' not in text and 'package://' not in text and '$(' not in text
assert not b.findall('.//cylinder')
assert len(b.findall('link'))==4 and len(b.findall('joint'))==3
checked=[]
for link in b.findall('link'):
 name=link.attrib['name']; orig=a.find(f"link[@name='{name}']"); meshlink=c.find(f"link[@name='{name}']")
 assert norm(link.find('inertial'))==norm(orig.find('inertial')),name
 assert norm(link.find('visual'),True)==norm(orig.find('visual'),True),name
 assert norm(link.find('collision'),True)==norm(meshlink.find('collision'),True),name
 checked.append(name)
for joint in b.findall('joint'):
 assert norm(joint)==norm(a.find(f"joint[@name='{joint.attrib['name']}']"))
refs={}
for item in b.findall('.//mesh'):
 rel=item.attrib['filename'];p=(new.parent/rel).resolve()
 assert p.is_relative_to(newdir.resolve()) and p.is_file() and not p.is_symlink(),rel
 source=old.parents[1]/'meshes'/p.name
 assert sha(source)==sha(p),p
 refs[rel]=sha(p)
# Compare tracked active training asset tree to HEAD, without editing it.
asset='isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/assets'
assert not subprocess.check_output(['git','diff','HEAD','--name-only','--',asset],cwd=ROOT).strip()
result={'candidate':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'new_urdf_sha256':sha(new),'checked_links':checked,'joints_preserved':3,'unique_local_meshes':refs,'all_8_mesh_references_resolve':len(b.findall('.//mesh'))==8,'active_usd_tree_matches_HEAD':True,'actual_isaac_import_performed':False}
with tempfile.TemporaryDirectory(prefix='mesh-urdf-audit-') as tmp:
 copied=Path(tmp)/'model';shutil.copytree(newdir,copied)
 parsed=mujoco.MjModel.from_xml_path(str(copied/'urdf/trackstand_rl_0916.urdf'))
 result['isolated_mujoco_urdf_parse']={'succeeded':True,'nbody':parsed.nbody,'ngeom':parsed.ngeom,'note':'Resource/parser smoke only, not Isaac import or dynamics equivalence.'}
Path(__file__).with_name('audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
