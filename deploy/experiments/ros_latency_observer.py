#!/usr/bin/env python3
"""Same-Pi ROS callback/50-Hz consumer timing; never publishes or commands hardware."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import signal
import threading
import time

from latency_common import CsvLog, dump


def decode_probe(values):
    if len(values)!=14 or not all(math.isfinite(x) for x in values):
        raise ValueError('invalid probe state schema (expected 14 finite values)')
    seq,published=values[:2]
    kernel=values[2::3]
    source=min(kernel) if all(x>0 for x in kernel) else None
    return int(seq),published,source


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',choices=('probe','production'),default='probe')
    p.add_argument('--duration',type=float,default=600.)
    p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'logs')
    args=p.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration<=0:p.error('duration must be positive')
    import rclpy
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import Imu
    stop=threading.Event()
    handlers={s:signal.signal(s,lambda *_:stop.set()) for s in (signal.SIGINT,signal.SIGTERM)}
    out=args.output_dir/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_ros_latency');out.mkdir(parents=True)
    dump(out/'metadata.json',{'source':args.source,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                             'same_pi_required':True,'consumer_hz':50})
    log=CsvLog(out/'ros.csv');node=None
    try:
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node=Node('trackstand_latency_observer')
        latest={};sequence=0
        def callback(message):
            nonlocal sequence
            callback_at=time.monotonic();sequence+=1
            if args.source=='probe':
                seq,published,source=decode_probe(list(message.data))
                detail='probe kernel timestamps; ROS callback lag includes publisher serialization/DDS/executor'
            else:
                if len(message.data)<5:return
                seq,published=sequence,None
                # Production motor_state[4] is node publication wall time, NOT
                # kernel feedback time. Do not mislabel it as sensor acquisition.
                wall_age=time.time()-message.data[4]
                source=callback_at-wall_age
                detail='production publication wall timestamp; not CAN sample timestamp'
            latest.update(seq=seq,publish_s=published,callback_s=callback_at,source=source)
            log.emit('ros_callback',seq=seq,publish_s=published,callback_s=callback_at,
                     source_age_ms=None if source is None else (callback_at-source)*1000,detail=detail)
        def imu_callback(message):
            wall=node.get_clock().now().nanoseconds*1e-9
            stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
            log.emit('imu_callback',callback_s=time.monotonic(),source_age_ms=(wall-stamp)*1000)
        def tick():
            now=time.monotonic()
            if latest:
                log.emit('consumer_tick',seq=latest['seq'],publish_s=latest['publish_s'],
                         callback_s=latest['callback_s'],tick_s=now,
                         source_age_ms=None if latest['source'] is None else (now-latest['source'])*1000)
        topic='/trackstand/latency_probe/state' if args.source=='probe' else '/trackstand/motor_state'
        node.create_subscription(Float64MultiArray,topic,callback,1)
        node.create_subscription(Imu,'/trackstand/imu',imu_callback,1)
        node.create_timer(.02,tick)
        print(f'Observing {topic}; 50-Hz mock consumer, no inference/commands. Logs: {out}',flush=True)
        end=time.monotonic()+args.duration
        while not stop.is_set() and rclpy.ok() and time.monotonic()<end:rclpy.spin_once(node,timeout_sec=.02)
    finally:
        if node is not None:node.destroy_node()
        rclpy.try_shutdown();log.close()
        for s,h in handlers.items():signal.signal(s,h)
    return 0


if __name__=='__main__':raise SystemExit(main())
