"""Offline tests only: no CAN interface, motor, ROS runtime or shell processes."""
import csv
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import can_latency_probe as probe
from latency_common import ClockMap,CsvLog
import ros_latency_observer as observer
import analyze_latency as analysis


class MemoryLog:
    def __init__(self):self.rows=[]
    def emit(self,kind,**kw):self.rows.append(dict(kind=kind,**kw))


class FakeBus:
    instances=[]
    def __init__(self,interface):
        self.frames=[];self.sent=[];self.sock=object();self.closed=False
        self.instances.append(self)
    def send(self,motor,op):
        self.sent.append((motor,op))
        self.frames.append((motor,bytes([op])+bytes(probe.LENGTHS[op]-1),time.time(),time.monotonic(),False,0))
    def receive(self):
        return self.frames.pop(0) if self.frames else None
    def close(self):self.closed=True


class Tests(unittest.TestCase):
    def test_load_matches_nominal_bits_not_frames(self):
        p=probe.load_plan()
        self.assertAlmostEqual(p['reference_bits_per_s'],p['probe_expected_bits_per_s'])
        self.assertAlmostEqual(p['f1_queries_per_motor_hz'],270.88607594936707)
        self.assertGreater(p['probe_expected_frames_per_s'],p['reference_frames_per_s'])
        self.assertAlmostEqual(probe.load_plan(.5)['probe_expected_bits_per_s'],p['reference_bits_per_s']/2)

    def test_single_byte_allowlist(self):
        b=probe.Bus.__new__(probe.Bus)
        class Sock:
            def send(self,data):
                can,dlc,body=probe.FRAME.unpack(data)
                assert can==0x101 and dlc==1 and body[0]==0xCF
                return len(data)
        b.sock=Sock();b.send(1,0xCF)
        for op in (0xC0,0xC1,0xC2,0xC3,0xB1):
            with self.assertRaises(ValueError):b.send(1,op)
        with self.assertRaises(ValueError):b.send(3,0xF1)

    def test_clock_interruption_logged_invalid_not_exception(self):
        c=ClockMap();self.assertTrue(c.observe(10,1010,10.0001)[3])
        self.assertFalse(c.observe(11,1011,11.005)[3])
        # Step invalidates one mapping and rebaselines.
        self.assertFalse(c.observe(12,2012,12.0001)[3])
        self.assertTrue(c.observe(13,2013,13.0001)[3])

    def test_pairing_clean(self):
        t=probe.Tracker();t.sent(1,0xA2,3,10)
        seq,rtt,valid=t.reply(1,0xA2,10.001,True)
        self.assertEqual(seq,3);self.assertAlmostEqual(rtt,1);self.assertTrue(valid)

    def test_overlap_stays_ambiguous(self):
        t=probe.Tracker();t.sent(1,0xA2,1,10);t.sent(1,0xA2,2,10.01)
        self.assertFalse(t.reply(1,0xA2,10.012,True)[2])
        self.assertFalse(t.reply(1,0xA2,10.013,True)[2])
        t.sent(1,0xA2,3,11);self.assertFalse(t.reply(1,0xA2,11.001,True)[2])

    def test_timeout_cannot_mislabel_late_reply_as_new(self):
        t=probe.Tracker();t.sent(1,0xA3,1,10);self.assertEqual(len(t.expire(10.1)),1)
        t.sent(1,0xA3,2,10.2);self.assertFalse(t.reply(1,0xA3,10.201,True)[2])

    def test_invalid_clock_and_unsolicited(self):
        t=probe.Tracker();self.assertEqual(t.reply(1,0xAE,1,True),(None,None,False))
        t.sent(1,0xA2,1,1);self.assertFalse(t.reply(1,0xA2,None,False)[2])

    def test_csv_flush_and_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.csv';log=CsvLog(path);log.emit('send',seq=1);log.close()
            self.assertEqual(list(csv.DictReader(path.open()))[0]['seq'],'1')
            with self.assertRaises(FileExistsError):CsvLog(path)

    def test_probe_state_schema(self):
        vals=[2,100]+[99,99.01,99.02]*4
        self.assertEqual(observer.decode_probe(vals),(2,100,99))
        vals[2]=-1;self.assertIsNone(observer.decode_probe(vals)[2])
        with self.assertRaises(ValueError):observer.decode_probe([1])

    def test_preview_opens_no_hardware(self):
        with patch.object(probe,'Bus',side_effect=AssertionError('hardware')):
            self.assertEqual(probe.main([]),0)

    def run_fake(self,mode):
        args=SimpleNamespace(interface='not_real',mode=mode,load_scale=1.,duration=.04)
        log=MemoryLog();p=probe.Probe(args,log,threading.Event())
        with patch.object(probe,'Bus',FakeBus),patch.object(probe.select,'select',return_value=([],[],[])):
            p.run()
        return p,log,FakeBus.instances[-1]

    def test_disabled_end_to_end(self):
        p,log,b=self.run_fake('disabled')
        self.assertIsNone(p.error);self.assertTrue(b.closed)
        self.assertEqual(p.verified,{1,2});self.assertTrue(any(op==0xA3 for _,op in b.sent))
        self.assertTrue(all(op in probe.LENGTHS for _,op in b.sent))
        self.assertEqual(b.sent[-2:],[(1,0xCF),(2,0xCF)])
        self.assertTrue(any(r['kind']=='measurement_start' for r in log.rows))

    def test_passive_never_sends_even_exit(self):
        p,log,b=self.run_fake('passive')
        self.assertEqual(b.sent,[]);self.assertIsNone(p.error)

    def test_active_f1_aborts_disabled_probe(self):
        p=probe.Probe(SimpleNamespace(mode='disabled'),MemoryLog(),threading.Event())
        p.bus=FakeBus('');p.disable_sent={1:0,2:0}
        p.bus.frames=[(1,bytes([0xF1,0,0,0,0,0,1]),time.time(),time.monotonic(),False,0)]
        with self.assertRaisesRegex(RuntimeError,'F1 active'):p.receive()

    def test_foreign_writer_aborts(self):
        p=probe.Probe(SimpleNamespace(mode='disabled'),MemoryLog(),threading.Event());p.bus=FakeBus('')
        p.bus.frames=[(0x501,bytes(8),time.time(),time.monotonic(),True,0)]
        with self.assertRaisesRegex(RuntimeError,'another CAN writer'):p.receive()

    def test_passive_accepts_external_motion_capture(self):
        p=probe.Probe(SimpleNamespace(mode='passive'),MemoryLog(),threading.Event());p.bus=FakeBus('')
        p.bus.frames=[(0x501,bytes(8),time.time(),time.monotonic(),True,0)]
        self.assertTrue(p.receive());self.assertEqual(p.tx,0)

    def test_analysis_distinguishes_publication_and_callback(self):
        with tempfile.TemporaryDirectory() as tmp:
            h=Path(tmp);(h/'metadata.json').write_text('{"source":"probe"}')
            log=CsvLog(h/'ros.csv')
            log.emit('ros_callback',seq=1,publish_s=10,callback_s=10.005,source_age_ms=8)
            log.emit('consumer_tick',seq=1,publish_s=10,callback_s=10.005,tick_s=10.02,source_age_ms=23)
            log.close();out=analysis.analyze(h)
            self.assertAlmostEqual(out['timing']['publish_to_callback_ms']['p50'],5)
            self.assertAlmostEqual(out['timing']['callback_to_50hz_consumer_ms']['p50'],15)

    def test_receive_decodes_new_kernel_timestamp(self):
        b=probe.Bus.__new__(probe.Bus);b.drops=0
        b.sock=SimpleNamespace(recvmsg=lambda *_:(probe.FRAME.pack(1,5,b'\xa2\0\0\0\0'),
                              [(1,64,struct.pack('=qq',1700000000,123)),(1,40,struct.pack('=I',7))],0,None))
        row=b.receive();self.assertEqual(row[0],1);self.assertEqual(row[-1],7);self.assertEqual(row[2],1700000000+123e-9)

    def test_truncated_metadata_rejected(self):
        b=probe.Bus.__new__(probe.Bus);b.drops=0
        b.sock=SimpleNamespace(recvmsg=lambda *_:(probe.FRAME.pack(1,1,b'\xa2'),[],probe.socket.MSG_CTRUNC,None))
        with self.assertRaisesRegex(RuntimeError,'truncated'):b.receive()


    def test_main_writes_complete_run_artifacts_without_hardware(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(probe,'Bus',FakeBus),patch.object(probe.select,'select',return_value=([],[],[])),patch.object(probe,'system_snapshot',return_value={'test':True}):
                rc=probe.main(['--execute','--duration','.02','--output-dir',tmp])
            self.assertEqual(rc,0)
            folder=next(Path(tmp).iterdir())
            result=json.loads((folder/'result.json').read_text())
            self.assertTrue(result['worker_stopped']);self.assertIsNone(result['error'])
            out=analysis.analyze(folder)
            self.assertIn('achieved_expected_ratio',out['traffic'])
            self.assertGreater(out['traffic']['tx_frames_per_s'],0)

    def test_main_passive_artifacts_contain_no_sends(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(probe,'Bus',FakeBus),patch.object(probe.select,'select',return_value=([],[],[])),patch.object(probe,'system_snapshot',return_value={}):
                rc=probe.main(['--execute','--mode','passive','--duration','.02','--output-dir',tmp])
            self.assertEqual(rc,0)
            folder=next(Path(tmp).iterdir());result=json.loads((folder/'result.json').read_text())
            self.assertEqual(result['tx'],0)

    def test_bus_setup_portable_error_filter_and_cleanup(self):
        from unittest.mock import Mock
        sock=Mock();sock.bind.side_effect=OSError('interface down')
        with patch.object(probe.socket,'socket',return_value=sock):
            with self.assertRaisesRegex(OSError,'interface down'):probe.Bus('notreal')
        sock.close.assert_called_once()
        self.assertTrue(any(call.args[:2]==(probe.socket.SOL_CAN_RAW,getattr(probe.socket,'CAN_RAW_ERR_FILTER',2)) for call in sock.setsockopt.call_args_list))

    def test_ros_probe_publishes_only_diagnostic_topics(self):
        from unittest.mock import Mock
        import types
        stop=threading.Event();created=[];callbacks=[]
        class Msg:
            def __init__(self,**kw):self.__dict__.update(kw)
        class Node:
            def __init__(self,*a):pass
            def create_publisher(self,kind,topic,depth):
                created.append(topic);return SimpleNamespace(publish=lambda msg:None)
            def create_timer(self,dt,callback):callbacks.append(callback)
            def destroy_node(self):pass
        ros=types.ModuleType('rclpy');ros.init=lambda **kw:None;ros.ok=lambda:True;ros.try_shutdown=lambda:None
        def spin(*a,**kw):
            callbacks[0]();callbacks[0]();stop.set()
        ros.spin_once=spin
        modules={'rclpy':ros,'rclpy.node':SimpleNamespace(Node=Node),
                 'rclpy.signals':SimpleNamespace(SignalHandlerOptions=SimpleNamespace(NO=0)),
                 'std_msgs.msg':SimpleNamespace(Float64MultiArray=Msg,String=Msg,Bool=Msg)}
        p=SimpleNamespace(lock=threading.Lock(),latest={},tx=0,rx=0,error=None)
        with patch.dict(sys.modules,modules):probe.run_ros(p,MemoryLog(),stop,'test')
        self.assertEqual(len(created),4)
        self.assertTrue(all(t.startswith('/trackstand/latency_probe/') for t in created))


if __name__=='__main__':unittest.main()
