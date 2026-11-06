#!/usr/bin/env python3
"""Summarize diagnostic CSVs without ROS or CAN; outputs JSON and plain text."""
import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def number(row,key):
    try:
        v=float(row[key]);return v if math.isfinite(v) else None
    except (ValueError,KeyError,TypeError):return None


def percentile(x,p):
    x=sorted(x);i=(len(x)-1)*p;lo=int(i);hi=min(lo+1,len(x)-1)
    return x[lo]+(x[hi]-x[lo])*(i-lo)


def stats(values):
    x=[v for v in values if v is not None and math.isfinite(v)]
    return {'count':len(x),'p50':percentile(x,.5),'p95':percentile(x,.95),'p99':percentile(x,.99),'max':max(x)} if x else {'count':0}


def analyze(folder):
    folder=Path(folder);can=folder/'can.csv';ros=folder/'ros.csv'
    path=can if can.exists() else ros
    rows=list(csv.DictReader(path.open()))
    meta=json.loads((folder/'metadata.json').read_text())
    kinds=Counter(r['kind'] for r in rows);out={'folder':str(folder),'metadata':meta,'events':dict(kinds)}
    metrics=defaultdict(list)
    for r in rows:
        for name,key in [('clock_uncertainty_ms','clock_uncertainty_ms'),('receive_queue_ms','queue_ms')]:
            if r['kind']=='receive':metrics[name].append(number(r,key))
        if r['kind']=='receive':
            dq,pr=number(r,'dequeue_s'),number(r,'processed_s')
            if dq is not None and pr is not None:metrics['dequeue_to_processed_ms'].append((pr-dq)*1000)
        if r['kind']=='send':
            start,end,due=(number(r,k) for k in ('send_start_s','send_end_s','scheduled_s'))
            if start is not None and end is not None:metrics['send_syscall_ms'].append((end-start)*1000)
            if start is not None and due is not None:metrics['query_dispatch_lateness_ms'].append((start-due)*1000)
        if r['kind']=='receive' and r['pair_valid']=='1':metrics['query_to_kernel_reply_ms_trusted'].append(number(r,'rtt_ms'))
        if r['kind']=='ros_callback':
            pub,cb=number(r,'publish_s'),number(r,'callback_s')
            if pub is not None and cb is not None:metrics['publish_to_callback_ms'].append((cb-pub)*1000)
            metrics['callback_source_age_ms'].append(number(r,'source_age_ms'))
        if r['kind']=='consumer_tick':
            cb,tick=number(r,'callback_s'),number(r,'tick_s')
            if cb is not None and tick is not None:metrics['callback_to_50hz_consumer_ms'].append((tick-cb)*1000)
            metrics['consumer_source_age_ms'].append(number(r,'source_age_ms'))
    out['timing']= {k:stats(v) for k,v in metrics.items()}
    out['clock_samples_above_1ms']=sum((number(r,'clock_uncertainty_ms') or 0)>.001*1000 for r in rows)
    out['invalid_clock_samples']=sum(r['kind']=='receive' and r['clock_valid']=='0' for r in rows)
    out['max_socket_dropped']=max((number(r,'rx_dropped') or 0 for r in rows),default=0)
    out['can_error_frames']=sum(r['kind']=='receive' and bool(int(r['can_id'],16)&0x20000000) for r in rows if r['can_id'])
    if can.exists():
        result=json.loads((folder/'result.json').read_text())
        out['result']=result
        start=result.get('started_s');end=result.get('finished_s')
        if start is not None and end is not None and end>start:
            window=[r for r in rows if start<=float(r['mono_s'])<end and r['kind'] in ('send','receive')]
            bits=sum(47+8*int(r['dlc']) for r in window if r['dlc'])
            elapsed=end-start;rate=bits/elapsed
            out['traffic']={'duration_s':elapsed,'estimated_bits_per_s':rate,'frames_per_s':len(window)/elapsed,
                            'tx_frames_per_s':sum(r['kind']=='send' for r in window)/elapsed,
                            'rx_frames_per_s':sum(r['kind']=='receive' for r in window)/elapsed,
                            'by_kind_id_opcode':dict(Counter((r['kind']+':'+r['can_id']+':'+r['opcode']) for r in window))}
            if meta['mode']=='disabled':
                ref=meta['load_plan']['probe_expected_bits_per_s'];ratio=rate/ref
                out['traffic']['achieved_expected_ratio']=ratio
                out['traffic']['within_10_percent_of_planned_airtime']=.9<=ratio<=1.1
                out['traffic']['comparable_to_full_deployment_reference']=.9<=rate/meta['load_plan']['reference_bits_per_s']<=1.1
    (folder/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folders',nargs='+',type=Path);a=p.parse_args(argv)
    for f in a.folders:
        out=analyze(f)
        text=json.dumps(out,indent=2)
        (f/'analysis.txt').write_text(text+'\n')
        print(text)
    print('Timing is host-side, not firmware-loop timing. Compare same-Pi boot IDs for monotonic time correlation.\n'
          'Airtime excludes stuffing/retries; low achieved traffic invalidates an equal-load conclusion.\n'
          'Query RTT is trusted only before overlapping/time-out ambiguity for that motor/opcode.\n'
          'Production-source ROS age means publication age, not CAN feedback age.')


if __name__=='__main__':main()
