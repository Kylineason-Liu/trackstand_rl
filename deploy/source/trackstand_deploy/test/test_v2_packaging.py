"""Check installed entry points/assets and unchanged shared hardware configuration."""

import ast
from pathlib import Path
import runpy

import yaml


PACKAGE = Path(__file__).parents[1]


def test_v2_entry_points_and_model_sidecar_are_packaged(monkeypatch):
    import setuptools
    configuration = {}
    monkeypatch.setattr(setuptools, 'setup', lambda **kwargs: configuration.update(kwargs))
    monkeypatch.chdir(PACKAGE)
    runpy.run_path(str(PACKAGE/'setup.py'))
    entries = configuration['entry_points']['console_scripts']
    for name in ('motor_mit_io_v2', 'policy_controller_v2'):
        assert f'{name} = trackstand_deploy.{name}:main' in entries
        tree = ast.parse((PACKAGE/'trackstand_deploy'/f'{name}.py').read_text())
        assert any(isinstance(node, ast.FunctionDef) and node.name == 'main' for node in tree.body)
    packaged = {Path(file).name for _, files in configuration['data_files'] for file in files}
    assert {'params_v2.yaml', 'trackstand_v2.launch.py',
            'trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx',
            'trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.contract.json'} <= packaged


def test_v2_preserves_hardware_and_has_consistent_node_names():
    old = yaml.safe_load((PACKAGE/'config/params.yaml').read_text())
    new = yaml.safe_load((PACKAGE/'config/params_v2.yaml').read_text())
    for name in ('trackstand_imu_publisher', 'trackstand_rc_kickstand', 'trackstand_kickstand_driver'):
        assert old[name] == new[name]
    for kind in ('motor_io', 'policy_controller'):
        previous = old[f'trackstand_{kind}']['ros__parameters']
        current = new[f'trackstand_{kind}_v2']['ros__parameters']
        for key in previous.keys() & current.keys():
            assert current[key] == previous[key], key
    launch = (PACKAGE/'launch/trackstand_v2.launch.py').read_text()
    assert '"params_v2.yaml"' in launch
    for executable, name in (('motor_mit_io_v2', 'trackstand_motor_io_v2'),
                             ('policy_controller_v2', 'trackstand_policy_controller_v2')):
        assert f'node("{executable}", "{name}")' in launch and name in new
