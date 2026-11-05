"""Pure v2 action, PI, wire-unit and durable-restore contract checks."""

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from trackstand_deploy.contracts_v2 import (
    VelocityIntegral, encoded_drive_kd, integral_limit, rear_f0_payload,
    scale_policy_action_v2, validate_policy_manifest, velocity_packet,
)
from trackstand_deploy.mit_protocol import parse_f0, parse_mit_motion
from trackstand_deploy.mit_settings import SettingsJournal


def test_action_units_and_bounded_beta_contract():
    assert scale_policy_action_v2([1, -1]) == (1.57, -61.6)
    assert scale_policy_action_v2([0, 0]) == (0, 0)
    for bad in ([1.001, 0], [0, float('nan')], [0], [0, 0, 0]):
        with pytest.raises(ValueError):
            scale_policy_action_v2(bad)


def test_integral_primes_retains_history_uses_elapsed_time_and_resets():
    pi = VelocityIntegral()
    pi.update(1, 10, .3, 6.28)
    assert pi.torque_nm == 0
    pi.update(1, 10.005, .3, 6.28)
    assert pi.torque_nm == pytest.approx(3.3 * .005)
    pi.update(-1, 10.025, .3, 6.28)  # New target keeps history; trapezoid increment is zero.
    assert pi.torque_nm == pytest.approx(3.3 * .005)
    assert pi.last_dt_s == pytest.approx(.02)
    with pytest.raises(ValueError, match="backwards"):
        pi.update(1, 9, .3, 6.28)
    pi.reset()
    assert pi.previous_time is None and pi.torque_nm == pi.error_rad == 0


def test_conditional_antiwindup_freezes_outward_and_permits_unwind():
    pi = VelocityIntegral()
    pi.update(1, 10, .3, 6.28)
    pi.update(1, 11, .3, 6.28)
    assert pi.torque_nm == pytest.approx(3.3)
    pi.update(100, 11.1, .3, 6.28)
    assert pi.torque_nm == pytest.approx(3.3)
    assert pi.limited_updates == 1
    pi.update(-1, 11.1, .3, 6.28)
    pi.update(-1, 11.2, .3, 6.28)
    assert pi.torque_nm < 3.3


def test_integral_matches_actual_isaac_class_across_targets_saturation_and_resets():
    import ast
    import math
    from collections.abc import Sequence
    import torch

    root = Path(__file__).resolve().parents[4]
    source = root / ('isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/'
                     'trackstand/mdp/stationary/velocity_pi.py')
    # Load the actual pure controller class, without starting/importing Isaac Sim.
    definition = next(node for node in ast.parse(source.read_text()).body
                      if isinstance(node, ast.ClassDef) and node.name == 'VelocityIntegral')
    namespace = {'torch': torch, 'math': math, 'Sequence': Sequence}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
    sim = namespace['VelocityIntegral'](torch.zeros((1, 1), dtype=torch.float64), 3.3)
    host = VelocityIntegral()
    kd, limit = torch.tensor([[.3]], dtype=torch.float64), torch.tensor([[6.28]], dtype=torch.float64)
    for index in range(1000):
        if index == 410:
            sim.reset()
            host.reset()
        error = float(100 if index < 20 else np.sin(index * .07) * 4)
        sim.update(torch.tensor([[error]], dtype=torch.float64), .005, kd, limit)
        host.update(error, 100 + index * .005, .3, 6.28)
        assert host.torque_nm == pytest.approx(sim.torque_nm.item(), abs=1e-11)


@pytest.mark.parametrize("target", [-61.6, -4, 0, .01, 61.6])
@pytest.mark.parametrize("integral_nm", [-6, 0, 6])
def test_full_velocity_range_sign_and_quantization(target, integral_nm):
    original = bytes([0xf0]) + struct.pack('<HHH', 955, 1300, 1800)
    ranges = parse_f0(rear_f0_payload(original))
    packet = velocity_packet(target, .7, integral_nm, ranges)
    wire = parse_mit_motion(packet.payload, ranges)
    assert wire.kp_nm_per_rad == 0
    assert wire.kd_nm_s_per_rad == pytest.approx(encoded_drive_kd())
    assert wire.feedforward_torque_nm == pytest.approx(-integral_nm, abs=ranges.torque_nm / 4095)
    assert packet.effective_target_rad_s == pytest.approx(target, abs=ranges.velocity_rad_s / 4095 + 1e-12)
    assert packet.estimated_total_torque_nm == pytest.approx(
        -wire.kd_nm_s_per_rad * (wire.velocity_rad_s + .7) - wire.feedforward_torque_nm)
    assert integral_limit(ranges) < 6.28


def test_large_proportional_demand_is_reported_not_silently_clipped():
    ranges = parse_f0(rear_f0_payload(b'\xf0' + struct.pack('<HHH', 955, 4500, 1800)))
    assert velocity_packet(61.6, -61.6, 0, ranges).estimated_total_torque_nm > 6.28


def test_bundled_v2_model_has_verified_sidecar_and_real_bounded_inference():
    from trackstand_deploy.policy import OnnxPolicy
    model = Path(__file__).parents[1] / 'models/trackstand_rl_v2_rsl_2026-11-02_22-40-01_model_50100.onnx'
    manifest = validate_policy_manifest(model)
    assert manifest['training_domain_randomization'] is True
    actor = OnnxPolicy(str(model))
    for magnitude in (0, 1, 10, 100):
        action = actor.act(np.full(11, magnitude, dtype=np.float32))
        assert action.shape == (2,) and np.isfinite(action).all() and (abs(action) <= 1).all()


def test_manifest_rejects_same_shape_torque_actor(tmp_path):
    model = tmp_path / 'actor.onnx'
    model.write_bytes(b'placeholder')
    doc = {'task': 'Trackstand-Balance-v1', 'policy_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}
    model.with_suffix('.contract.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='v2 velocity'):
        validate_policy_manifest(model)


def originals():
    return {f'{name}:{op}': payload.hex() for name in ('steering', 'drive') for op, payload in (
        ('F0', b'\xf0' + struct.pack('<HHH', 955, 4500, 1800)), ('CD', b'\xcd\0\0\0\0'))}


def test_journal_exclusive_ownership_and_recovery(tmp_path):
    path = tmp_path / 'journal.json'
    journal = SettingsJournal(path, {'steering': 2, 'drive': 1})
    journal.claim(originals(), originals())
    other = SettingsJournal(path, journal.device_ids)
    with pytest.raises(FileExistsError):
        other.claim(originals(), {})
    assert other.document is None
    with pytest.raises(RuntimeError, match='incomplete'):
        journal.finish()
    other.load()
    for key in list(other.pending):
        other.restored(key)
    other.finish()
    assert not path.exists()


def test_failed_journal_update_keeps_pending_original(tmp_path, monkeypatch):
    journal = SettingsJournal(tmp_path/'j.json', {'steering': 2, 'drive': 1})
    journal.claim(originals(), originals())
    monkeypatch.setattr('trackstand_deploy.mit_settings.os.replace', lambda *_: (_ for _ in ()).throw(OSError('disk')))
    with pytest.raises(OSError):
        journal.restored('drive:F0')
    assert 'drive:F0' in journal.pending and 'drive:F0' in json.loads(journal.path.read_text())['pending']
