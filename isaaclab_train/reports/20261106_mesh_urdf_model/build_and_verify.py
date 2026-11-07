"""Build a self-contained mesh-collision URDF and check asset preservation."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
import mujoco

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'isaaclab_train/trackstand_rl_0916_description_cylinder'
ORIGINAL = ROOT/'isaaclab_train/trackstand_rl_0916_description'
DEST = ROOT/'isaaclab_train/trackstand_rl_0916_description_mesh'
OUT = Path(__file__).resolve().parent
URDF = 'urdf/trackstand_rl_0916.urdf'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protected = [p for directory in (SOURCE, ORIGINAL) for p in directory.rglob('*') if p.is_file()]
before = {str(p.relative_to(ROOT)):sha(p) for p in protected}
base = ET.parse(SOURCE/URDF).getroot()
original = ET.parse(ORIGINAL/URDF).getroot()
new = copy.deepcopy(base)
new.set('name','trackstand_rl_0916_mesh')
for node in list(new):
    if node.tag.startswith('{'):
        new.remove(node)
# Inline the material used by all visuals, preserving source RGBA exactly.
material = ET.parse(SOURCE/'urdf/materials.xacro').getroot().find('material')
new.insert(0,copy.deepcopy(material))
for wheel in ('rear_wheel_link_1','front_wheel_link_1'):
    link = new.find(f"link[@name='{wheel}']")
    link.remove(link.find('collision'))
    link.append(copy.deepcopy(original.find(f"link[@name='{wheel}']/collision")))
for mesh in new.findall('.//mesh'):
    filename = Path(mesh.get('filename')).name
    mesh.set('filename','../meshes/'+filename)
    shutil.copy2(SOURCE/'meshes'/filename,DEST/'meshes'/filename)
shutil.copy2(SOURCE/'LICENSE',DEST/'LICENSE')
ET.indent(new,space='  ')
ET.ElementTree(new).write(DEST/URDF,encoding='utf-8',xml_declaration=True)
# Structural equality: normalize only local resource names, not transforms.
def structure(node):
    attrs = dict(node.attrib)
    if node.tag=='mesh': attrs['filename']=Path(attrs['filename']).name
    return [node.tag,attrs,[structure(c) for c in node]]
for link in new.findall('link'):
    previous=base.find(f"link[@name='{link.get('name')}']")
    for element in ('inertial','visual'):
        assert structure(link.find(element))==structure(previous.find(element))
    expected=original.find(f"link[@name='{link.get('name')}']/collision")
    assert structure(link.find('collision'))==structure(expected)
for joint,previous in zip(new.findall('joint'),base.findall('joint')):
    assert structure(joint)==structure(previous)
assert len(new.findall('.//cylinder'))==0
assert len(new.findall('link'))==4 and len(new.findall('joint'))==3
assert all('$(find' not in value and 'package://' not in value for n in new.iter() for value in n.attrib.values())
meshes={}
for mesh in new.findall('.//mesh'):
    path=((DEST/URDF).parent/mesh.get('filename')).resolve()
    assert path.is_relative_to(DEST.resolve()) and path.is_file()
    assert sha(path)==sha(SOURCE/'meshes'/path.name)
    meshes[path.name]=sha(path)
with tempfile.TemporaryDirectory(prefix='trackstand-mesh-urdf-') as temp:
    isolated=Path(temp)/'bike'
    shutil.copytree(DEST,isolated)
    copied=ET.parse(isolated/URDF).getroot()
    assert all(((isolated/URDF).parent/n.get('filename')).resolve().is_file() for n in copied.findall('.//mesh'))
    parsed=mujoco.MjModel.from_xml_path(str(isolated/URDF))
    parser_check=dict(library="MuJoCo",version=mujoco.__version__,body_count=parsed.nbody,joint_count=parsed.njnt,collision_geom_count=parsed.ngeom)
    assert parsed.njnt==3 and parsed.ngeom==4
    isolated_check=True
assert before=={str(p.relative_to(ROOT)):sha(p) for p in protected}
result=dict(urdf_sha256=sha(DEST/URDF),source_urdf_sha256=sha(SOURCE/URDF),original_mesh_urdf_sha256=sha(ORIGINAL/URDF),mesh_sha256=meshes,protected_file_count=len(protected),source_files_unchanged=True,isolated_resource_resolution=isolated_check,link_count=4,joint_count=3,visual_and_inertial_and_joint_values_preserved=True,wheel_collision_origins={wheel:new.find(f"link[@name='{wheel}']/collision/origin").attrib for wheel in ('rear_wheel_link_1','front_wheel_link_1')},mass_total_kg=sum(float(n.get('value')) for n in new.findall('link/inertial/mass')),fresh_isaac_import_performed=False,isolated_urdf_parser=parser_check)
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
