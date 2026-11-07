import sys
from pathlib import Path
import unittest
import tempfile
import json
import csv
import struct
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import drive_v2_characterization as m

class Log:
    def emit(self,row):pass
    def check(self):pass
class Bus:
    def __init__(self):self.frames=[]
    def send(self,frame):self.frames.append(frame);return len(frame)
class Tests(unittest.TestCase):
    def test_plan(self):
        plan=m.make_plan(m.parse_args([]))
        self.assertEqual(len(plan),605)
        self.assertEqual([t.target_rad_s for t in plan[:600]],[sign*i/10 for i in range(1,301) for sign in (1,-1)])
        self.assertTrue(all(t.duration_s==5 and t.rest_s==2 for t in plan[:600]))
        self.assertEqual(sum(t.duration_s+t.rest_s for t in plan),4260)
        self.assertEqual([t.trial for t in plan],list(range(1,606)))
        for trial in plan[600:]:
            self.assertEqual(trial.kind,'sine')
            self.assertAlmostEqual(trial.target(.25/trial.frequency_hz),4.)
            self.assertAlmostEqual(trial.target(.75/trial.frequency_hz),-4.)
    def test_parity_and_rear_only(self):
        bus=Bus();io=m.CharacterizationIO(m.parse_args([]),bus,Log())
        io.command_ranges=m.parse_f0(m.contract.rear_f0_payload(bytes.fromhex('f0bb0300110807')))
        io.velocity=.5;now=io.clock.monotonic();io.velocity_at=now
        expected=m.contract.VelocityIntegral()
        for dt,target in [(0,1),(.005,2),(.01,30),(.02,-1)]:
            expected.update(target-.5,now+dt,m.contract.encoded_drive_kd(),m.contract.integral_limit(io.command_ranges))
            packet=m.contract.velocity_packet(target,.5,expected.torque_nm,io.command_ranges)
            io.command(target,now+dt)
            cid,n,payload=m.shared.CAN_FRAME.unpack(bus.frames[-1])
            self.assertEqual(cid,0x501);self.assertEqual(payload[:n],packet.payload)
            self.assertEqual(io.integral.torque_nm,expected.torque_nm)
        for cid in (0x102,0x502,0,0x401):
            with self.assertRaises(ValueError):io.send(b'\xcf',cid)
        with self.assertRaises(ValueError):io.send(b'\xb1')
        io.reset_integral();self.assertEqual(io.integral.torque_nm,0)
    def test_scheduler_and_rest(self):
        class Clock:
            now=10.
            def monotonic(self):return self.now
            def time(self):return self.now+1000
        clock=Clock();bus=Bus();io=m.CharacterizationIO(m.parse_args([]),bus,Log(),clock=clock)
        io.trial=m.Trial(1,'hold',1.,0.,.1)
        io.command_ranges=m.parse_f0(m.contract.rear_f0_payload(bytes.fromhex('f0bb0300110807')))
        io.velocity=0.;io.velocity_at=clock.now
        io.drain=lambda:None
        with patch.object(m.select,'select',side_effect=lambda r,w,e,timeout:setattr(clock,'now',clock.now+max(timeout,.000001))):
            io.run_window(.1,True)
            packets=[m.shared.CAN_FRAME.unpack(f)[0] for f in bus.frames]
            self.assertEqual(packets.count(0x501),20)
            bus.frames.clear();start=clock.now
            io.run_window(2.,False)
            self.assertAlmostEqual(clock.now-start,2.,places=5)
            self.assertTrue(all(m.shared.CAN_FRAME.unpack(f)[0]==0x101 for f in bus.frames))
    def test_restore_and_settings_guard(self):
        io=m.CharacterizationIO(m.parse_args([]),Bus(),Log())
        with self.assertRaises(ValueError):io.send(bytes.fromhex('cd01f40100'))
        originals=[bytes.fromhex('f0bb0300110807'),bytes.fromhex('cd00000000')]
        io.saved=originals.copy();events=[]
        io.disable=lambda:events.append('CF')
        io.configure_setting=lambda p:events.append(p)
        io.cleanup()
        self.assertEqual(events,['CF',originals[1],originals[0]])
        self.assertEqual(io.saved,[])
    def test_mocked_complete_run(self):
        class Clock:
            now=100.
            def monotonic(self):return self.now
            def time(self):return self.now+1000
        class ReplyBus(Bus):
            def __init__(self):
                super().__init__();self.queue=[]
                self.settings={0xF0:bytes.fromhex('f0bb0300110807'),0xCD:bytes.fromhex('cd00000000')}
            def close(self):pass
            def send(self,frame):
                super().send(frame)
                cid,n,data=m.shared.CAN_FRAME.unpack(frame);data=data[:n];op=data[0]
                if cid==0x501:return len(frame)
                if op in self.settings:
                    if n>1:self.settings[op]=data
                    reply=self.settings[op]
                elif op==0xB0:reply=b'\xb0\x00'+struct.pack('<f',.3)+b'\x00'
                elif op in (0xAE,0xCF):reply=bytes([op])+bytes(7)
                elif op==0xA3:reply=bytes([op])+bytes(6)
                else:reply=bytes([op])+bytes(4)
                self.queue.append(m.shared.CAN_FRAME.pack(1,len(reply),reply))
                return len(frame)
            def recvmsg(self,*args):
                if not self.queue:raise BlockingIOError
                return self.queue.pop(0),[],0,None
        clock=Clock();bus=ReplyBus();cls=m.CharacterizationIO
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'test.csv'
            with patch.object(m.legacy,'open_bus',return_value=bus), patch.object(m,'make_plan',return_value=[m.Trial(1,'hold',1.,0.,.025,.02)]), patch.object(m,'CharacterizationIO',side_effect=lambda *a,**kw:cls(*a,clock=clock,**kw)), patch.object(m.select,'select',side_effect=lambda r,w,e,timeout:setattr(clock,'now',clock.now+max(timeout,.000001))):
                self.assertEqual(m.main(['--execute','--output',str(out)]),0)
            report=json.loads(out.with_suffix('.json').read_text())
            self.assertTrue(report['completed'] and report['disabled_and_restored'])
            self.assertEqual(report['pending_restore'],[])
            with out.open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertTrue(any(r['record_type']=='command' for r in rows))
            self.assertTrue(all(int(r['can_id'],16) in (1,0x101,0x501) for r in rows if r['can_id']))
            self.assertEqual(bus.settings[0xCD],bytes.fromhex('cd00000000'))
            self.assertEqual(bus.settings[0xF0],bytes.fromhex('f0bb0300110807'))
            recovery_path=Path(d)/'interrupted.json'
            recovery_path.write_text(json.dumps(dict(schema='drive_v2_characterization_v1',motor_id=1,
                pending_restore=['f0bb0300110807','cd00000000'])))
            bus.frames.clear();bus.settings[0xCD]=bytes.fromhex('cd01f40100')
            with patch.object(m.legacy,'open_bus',return_value=bus), patch.object(m,'CharacterizationIO',side_effect=lambda *a,**kw:cls(*a,clock=clock,**kw)):
                self.assertEqual(m.main(['--execute','--recover-settings',str(recovery_path),'--output',str(Path(d)/'recover.csv')]),0)
            self.assertTrue(all(m.shared.CAN_FRAME.unpack(f)[0]==0x101 for f in bus.frames))
            self.assertEqual(bus.settings[0xCD],bytes.fromhex('cd00000000'))

    def test_stale_velocity(self):
        io=m.CharacterizationIO(m.parse_args([]),Bus(),Log())
        with self.assertRaises(RuntimeError):io.check_feedback(io.clock.monotonic())
if __name__=='__main__':unittest.main()
