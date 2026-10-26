"""Audit four real replay recordings and plot unshifted motor responses; no CAN access."""

import csv
from dataclasses import dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import struct

os.environ.setdefault("MPLCONFIGDIR", "/tmp/trackstand_replay_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENTS = ROOT / "deploy/experiments"
SIM = ROOT / "isaaclab_train/reports/20261025_matched_target_replay"
FILES = (
    "20261026_162728_204031_scaled_50hz_policy_replay",
    "20261026_163127_842386_scaled_200hz_policy_replay",
    "20261026_163432_217938_scaled_50hz_drive_policy_replay",
    "20261026_163545_994664_scaled_200hz_drive_policy_replay",
)
RAD_PER_COUNT = 2 * math.pi / 16384
CORE = (2., 7.)
COMMON = (2., 5.36)  # Last complete 20-ms interval in the aborted steering run.
FIGURES = ("steering_tracking", "steering_zero_reference", "rear_tracking", "packet_timing",
           "simulator_comparison")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed/empty CSV: {path}")
    return rows


def held(time, values, at):
    """Zero-order hold on ACTUAL host send timestamps, never future commands."""
    index = np.searchsorted(time, at, side="right") - 1
    if np.any(index < 0):
        raise ValueError("requested reference before the first sent command")
    return values[index]


def describe(values):
    values = np.asarray(values)
    if not len(values):
        return None
    return dict(n=len(values), minimum=float(values.min()), mean=float(values.mean()),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)),
                p99=float(np.percentile(values, 99)), maximum=float(values.max()),
                rms=float(np.sqrt(np.mean(values ** 2))))


def error_stats(actual, reference):
    error = actual - reference
    return dict(n=len(error), bias=float(error.mean()), rms=float(np.sqrt(np.mean(error ** 2))),
                mae=float(abs(error).mean()), p95_abs=float(np.percentile(abs(error), 95)),
                maximum_abs=float(abs(error).max()),
                demeaned_rms=float(np.sqrt(np.mean((error - error.mean()) ** 2))))


@dataclass
class Run:
    stem: str

    def __post_init__(self):
        self.path = EXPERIMENTS / "logs" / (self.stem + ".csv")
        self.meta = json.loads(self.path.with_suffix(".json").read_text())
        self.rows = read_csv(self.path)
        self.plan = read_csv(self.path.with_name(self.stem + "_plan.csv"))
        self.drive = "drive_policy" in self.stem
        self.label = f"{'Rear C0' if self.drive else 'Steering MIT'} {self.meta['packet_hz']} Hz"
        kind = "drive_target" if self.drive else "mit_target"
        self.tx = [r for r in self.rows if r["record_type"] == kind]
        self.ct = np.array([float(r["replay_time_s"]) for r in self.tx])
        key = "encoded_command_torque_nm" if self.drive else "mit_target_position_rad"
        self.cv = np.array([float(r[key]) for r in self.tx])
        self.request = np.array([float(r["requested_torque_nm" if self.drive else "requested_position_rad"])
                                 for r in self.tx])
        self.end = min(float(r["replay_time_s"]) for r in self.rows
                       if r["phase"] == "shutdown" and r["can_direction"] == "local_tx")
        self.qtime, self.q = self.series("position_unwrapped_rad", 0xA3)
        self.vtime, self.v = self.series("velocity_rad_s", 0xA2)
        self.ttime, self.torque = self.series("torque_nm", 0xA1)
        self.ft, self.fq = self.series("mit_feedback_position_rad", 0xF1)
        self.metric_window = CORE if self.meta["completed"] else COMMON

    def series(self, key, opcode=None):
        rows = [r for r in self.rows if r["phase"] == "motion" and r["can_direction"] == "bus_rx"
                and r[key] and (opcode is None or r["can_opcode"] == str(opcode))]
        time = np.array([float(r["replay_time_s"]) for r in rows])
        values = np.array([float(r[key]) for r in rows])
        if not np.isfinite(time).all() or not np.isfinite(values).all() or np.any(np.diff(time) <= 0):
            raise ValueError(f"Invalid/duplicate/nonmonotonic measurements: {self.stem} {key}")
        return time, values

    def select(self, time, values, window):
        keep = (time >= window[0]) & (time < window[1])
        return time[keep], values[keep]

    def tracking(self, window):
        t, y = self.select(self.ttime if self.drive else self.qtime,
                           self.torque if self.drive else self.q, window)
        return error_stats(y, held(self.ct, self.cv, t))

    def alignment(self, window):
        """Descriptive time-shift fit; not a firmware/transport-latency measurement."""
        t, y = self.select(self.ttime if self.drive else self.qtime,
                           self.torque if self.drive else self.q, window)
        shifts = np.arange(0, .10001, .0001)
        losses = []
        for shift in shifts:
            error = y - held(self.ct, self.cv, t - shift)
            if not self.drive:
                error -= error.mean()  # Do not confuse the origin bias with response lag.
            losses.append(np.sqrt(np.mean(error ** 2)))
        losses = np.asarray(losses)
        best = int(losses.argmin())
        near = shifts[losses <= losses[best] * 1.01]
        return dict(best_shift_ms=float(shifts[best] * 1000),
                    within_one_percent_loss_shift_range_ms=[float(near.min()*1000), float(near.max()*1000)],
                    minimum_rms=float(losses[best]), zero_shift_rms=float(losses[0]),
                    offset_removed=not self.drive, window_s=list(window),
                    method="Grid 0..100 ms, 0.1-ms spacing; actual TX zero-order hold; fixed RX samples",
                    interpretation="Descriptive alignment only; no motor acquisition/wire timestamps; "
                                   "not a confidence interval, pure latency or internal loop rate")

    def validate(self):
        args = self.meta["arguments"]
        source = EXPERIMENTS / "replay_targets" / Path(args["targets"]).name
        assert digest(source) == self.meta["targets_sha256"]
        assert self.meta["disable_confirmed"] and self.meta["final_fault_code"] == 0
        assert self.meta["target_hz"] == 50 and args["query_rate"] == 100
        hashes = self.meta.get("source_sha256", {
            "steering_policy_replay.py": self.meta.get("script_sha256"),
            "steering_sine_mit_io.py": self.meta.get("transport_sha256"),
        })
        assert all(digest(HERE / "runtime_snapshot" / name) == value for name, value in hashes.items())
        assert np.array_equal([int(r["packet_index"]) for r in self.tx], np.arange(len(self.tx)))
        repeats = self.meta["packet_hz"] // 50
        assert all(int(row["target_index"]) == index // repeats for index, row in enumerate(self.tx))
        target_key = "requested_torque_nm" if self.drive else "target_rad"
        for index, row in enumerate(self.tx):
            expected = self.plan[index // repeats]
            assert float(expected[target_key]) == self.request[index]
            assert row["phase"] == expected["phase"]
            assert abs(float(row["target_time_s"]) - int(row["target_index"]) * .02) < 1e-9
            if index % repeats:
                assert row["mit_payload_hex"] == self.tx[index-1]["mit_payload_hex"]
            raw = bytes.fromhex(row["mit_payload_hex"])
            if self.drive:
                assert raw[0] == 0xC0 and len(raw) == 5
                code = struct.unpack("<i", raw[1:])[0]
                expected_current = args["torque_sign"] * self.request[index] / self.meta["torque_constant_nm_per_a"]
                assert code == round(expected_current * 1000)
                assert abs(self.cv[index]-self.request[index]) <= self.meta["torque_constant_nm_per_a"]*.000500001
            else:
                native = int.from_bytes(raw[:2], "big") * 6 / 65535 - 3
                assert abs(self.cv[index] - args["sign"] * (native-self.meta["native_zero_rad"])) < 1e-12
        for row in self.rows:
            if row["phase"] != "motion" or row["can_direction"] != "bus_rx":
                continue
            raw = bytes.fromhex(row["mit_payload_hex"])
            op = raw[0]
            if op == 0xA3:
                count = struct.unpack("<i", raw[3:7])[0]
                sign = args["encoder_sign"] if self.drive else args["sign"]
                expected = sign * (count-self.meta["encoder_zero_count"]) * RAD_PER_COUNT
                assert abs(float(row["position_unwrapped_rad"])-expected) < 1e-12
            elif op in (0xA1, 0xC0):
                current = struct.unpack("<i", raw[1:5])[0]*.001
                assert abs(float(row["current_a"])-current) < 1e-12
                expected = args["torque_sign"]*current*self.meta["torque_constant_nm_per_a"]
                assert abs(float(row["torque_nm"])-expected) < 1e-12
            elif op == 0xAE:
                assert raw[7] == 0 and raw[6] == (2 if self.drive else 5)
            elif op == 0xF1:
                assert raw[6] == 1
        if self.drive:
            assert self.meta["watchdog_restored"]
        if self.meta["completed"]:
            assert len(self.tx) == len(self.plan)*repeats
        else:
            assert not self.drive and "deadline" in self.meta["error"]
            assert len(self.tx) < len(self.plan)*repeats

    def metrics(self):
        intervals = np.diff(self.ct)*1000
        late = np.array([float(r["command_lateness_s"])*1000 for r in self.tx])
        queue_t, queue = self.series("receive_queue_delay_s")
        queries = [r for r in self.rows if r["record_type"] == "query_result" and r["phase"] == "motion"]
        output = dict(label=self.label, files=[self.path.name, self.path.with_suffix(".json").name,
                      self.stem+"_plan.csv"], metadata=self.meta, recording_rows=len(self.rows),
                      transmitted_packets=len(self.tx), planned_packets=len(self.plan)*self.meta["packet_hz"]//50,
                      last_sent_replay_s=float(self.ct[-1]), first_logged_shutdown_tx_s=self.end,
                      completed_core_window_s=list(self.metric_window),
                      tracking=self.tracking(self.metric_window), common_window_tracking=self.tracking(COMMON),
                      command_intervals_ms=describe(intervals), command_lateness_ms=describe(late),
                      effective_tx_hz=float((len(self.ct)-1)/(self.ct[-1]-self.ct[0])),
                      motion_receive_queue_age_ms=describe(queue*1000),
                      motion_query_results={status:sum(r["query_status"]==status for r in queries)
                                            for status in set(r["query_status"] for r in queries)},
                      alignment=self.alignment(self.metric_window),
                      common_window_alignment=self.alignment(COMMON),
                      speed_rad_s=describe(self.select(self.vtime,self.v,self.metric_window)[1]),
                      current_derived_torque_nm=describe(self.select(self.ttime,self.torque,self.metric_window)[1]))
        output["feedback_hz"] = {}
        for key, time in (("A3",self.qtime),("A2",self.vtime),("A1",self.ttime)):
            output["feedback_hz"][key] = float((len(time)-1)/(time[-1]-time[0]))
        for key in ("bus_voltage_v", "motor_temperature_c"):
            output[key] = describe(self.series(key,0xAE)[1])
        if self.drive:
            t,y = self.select(self.ttime,self.torque,CORE)
            speed = held(self.vtime,self.v,t)
            near = abs(speed)<.1
            _,zero = self.select(self.ttime,self.torque,(0.,1.9))
            output.update(zero_command_noise_nm=describe(zero),
                          near_zero_speed_core_fraction=float(near.mean()),
                          near_zero_speed_torque_abs_nm=describe(abs(y[near])),
                          near_zero_speed_abs_torque_above_moving_fit_fraction=float((abs(y[near])>.106).mean()),
                          observed_position_last_rad=float(self.q[-1]),
                          initial_encoder_single_turn_count=self.meta["encoder_zero_count"]%16384)
        else:
            a3_native_zero = self.meta["encoder_zero_count"]*RAD_PER_COUNT
            predicted_offset = a3_native_zero-self.meta["native_zero_rad"]
            t,fy = self.select(self.ft,self.fq,self.metric_window)
            observed_offset = np.interp(t,self.qtime,self.q)-fy
            first = next(r for r in self.rows if r["can_direction"]=="bus_rx" and r["can_opcode"]==str(0xF1))
            output["zero_reference"] = dict(initial_A3_native_rad=a3_native_zero,
                recorded_F1_zero_rad=self.meta["native_zero_rad"],
                initial_F1_position_code=int(first["mit_position_code"]),
                F0_position_max_rad=self.meta["mit_ranges"][0], predicted_offset_rad=predicted_offset,
                predicted_offset_deg=float(np.degrees(predicted_offset)),
                measured_A3_minus_F1_relative_rad=describe(observed_offset),
                residual_after_predicted_offset_rad=describe(observed_offset-predicted_offset),
                F1_based_tracking=error_stats(fy,held(self.ct,self.cv,t)))
            qt,q = self.select(self.qtime,self.q,self.metric_window)
            output["diagnostic_offset_removed_tracking"] = error_stats(q-predicted_offset,held(self.ct,self.cv,qt))
        return output


def compare_pairs(runs):
    results = {}
    for drive in (False,True):
        pair = [r for r in runs if r.drive==drive]
        assert pair[0].plan == pair[1].plan
        common = min(len(r.tx)//(r.meta["packet_hz"]//50) for r in pair)
        for index in range(common):
            assert pair[0].tx[index]["mit_payload_hex"] == pair[1].tx[4*index]["mit_payload_hex"]
        results["rear" if drive else "steering"] = dict(plans_exactly_equal=True,
            full_matched_50hz_intervals=common, encoded_payloads_equal_for_every_common_interval=True,
            note="Steering first A3 zeros differ although both F1 zeros equal +3 rad; "
                 "rear initial rotor/wheel angle differs between runs")
    return results


def simulator_data():
    result={}
    for tag in ("isaac400","mujoco400","mujoco800"):
        with np.load(SIM/tag/"airborne.npz") as data:
            sub=data["position_after_rad"].shape[1]
            result[tag]=dict(time=(np.arange(400*sub)+1)*float(data["physics_dt_s"]),
                             steering=data["position_after_rad"][:,:,4,0].ravel(),
                             rear_speed=data["velocity_after_rad_s"][:,:,4,1].ravel(),
                             rear_torque=data["submitted_nm"][:,:,4,1].ravel(),
                             target=np.repeat(data["target_rad"][:,4],sub))
    return result


def save(fig,name):
    for extension in ("png","svg","pdf"):
        fig.savefig(HERE/f"{name}.{extension}",dpi=160)
    plt.close(fig)


def panel(ax,window,unit,run=None):
    ax.set_xlim(*window)
    ax.set_ylabel(unit)
    ax.set_xlabel("Time from replay start (s)")
    ax.grid(alpha=.25)
    values=[]
    for line in ax.lines:
        if line.get_transform() is ax.transData:
            t=np.asarray(line.get_xdata());y=np.asarray(line.get_ydata())
            keep=(t>=window[0])&(t<=window[1])&np.isfinite(y)
            if keep.any():
                values.append(y[keep])
    if values:
        lo,hi=float(np.concatenate(values).min()),float(np.concatenate(values).max())
        pad=max((hi-lo)*.13,1e-4)
        ax.set_ylim(lo-pad,hi+pad)
        ax.text(.01,.98,f"min {lo:.5g} | max {hi:.5g}",va="top",transform=ax.transAxes,fontsize=8,
                bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
    for at in (2.,7.):
        if window[0]<=at<=window[1]:
            ax.axvline(at,color="gray",ls=":",lw=.9)
    if run and not run.meta["completed"] and window[0]<run.end<window[1]:
        ax.axvline(run.end,color="red",ls="--",label=f"Timing abort ~{run.end:.3f} s")
        ax.axvspan(run.end,window[1],color="gray",alpha=.1)
    ax.legend(loc="lower right",fontsize=8)


def target_line(ax,run):
    end=min(run.end, len(run.plan)*.02)
    ax.step(np.r_[run.ct,end],np.r_[run.cv,run.cv[-1]],where="post",color="black",lw=1.2,
            label="Encoded command (actual host sends)")


def steering_plots(runs):
    pair=[r for r in runs if not r.drive]
    fig,axes=plt.subplots(4,2,figsize=(18,16),layout="constrained")
    for col,run in enumerate(pair):
        for row,window in ((0,(0.,8.)),(1,CORE)):
            ax=axes[row,col]
            target_line(ax,run)
            ax.plot(run.qtime,run.q,color="tab:blue",label="A3 displacement from physical startup")
            ax.plot(run.ft,run.fq,color="tab:orange",alpha=.8,label="F1 relative to clipped startup value")
            ax.set_title(run.label+ (" | full run" if row==0 else " | recorded core"))
            panel(ax,window,"Steering position (rad)",run)
        axes[2,col].plot(run.vtime,run.v,color="tab:blue",label="A2 measured speed")
        panel(axes[2,col],CORE,"Steering speed (rad/s)",run)
        axes[3,col].plot(run.ttime,run.torque,color="tab:green",label="A1 measured current × B0, signed")
        panel(axes[3,col],CORE,"Current-derived torque (Nm)",run)
    fig.suptitle("Real steering replay: same Isaac-recorded 50 Hz targets; MIT Kp=10, Kd=0.2\n"
                 "True startup-relative A3 is blue; F1-relative orange uses an endpoint-biased zero",fontsize=13)
    save(fig,"steering_tracking")
    fig,axes=plt.subplots(2,2,figsize=(18,9),layout="constrained")
    for col,run in enumerate(pair):
        raw_time,raw_q=run.series("encoder_count",0xA3)
        native=raw_q*RAD_PER_COUNT
        ft,fp=run.series("mit_feedback_position_motor_rad",0xF1)
        axes[0,col].plot(raw_time,native,color="tab:blue",label="Native A3 count × 2π/16384")
        axes[0,col].plot(ft,fp,color="tab:orange",label="Native F1 position")
        axes[0,col].axhline(3.,color="red",ls="--",label="F0 upper encoding endpoint +3 rad")
        axes[0,col].set_title(run.label+" | initial range clipping")
        panel(axes[0,col],(0.,.35),"Native motor position (rad)")
        keep=(ft>=2)&(ft<min(7.,run.end))
        difference=np.interp(ft[keep],run.qtime,run.q)-run.fq[keep]
        axes[1,col].plot(ft[keep],difference,color="tab:purple",label="A3-relative − F1-relative")
        predicted=run.meta["encoder_zero_count"]*RAD_PER_COUNT-run.meta["native_zero_rad"]
        axes[1,col].plot([2.,7.],[predicted,predicted],color="red",ls="--",
                         label=f"Initial A3 minus F1: {predicted:.6f} rad ({np.degrees(predicted):.2f}°)")
        panel(axes[1,col],CORE,"Coordinate discrepancy (rad)",run)
    fig.suptitle("Steering initialization defect: F1 starts at code 65535 / +3 rad, while A3 lies above +3 rad\n"
                 "No curve has been shifted to make tracking look better",fontsize=13)
    save(fig,"steering_zero_reference")


def rear_plots(runs):
    pair=[r for r in runs if r.drive]
    fig,axes=plt.subplots(4,2,figsize=(18,15),layout="constrained")
    for col,run in enumerate(pair):
        ax=axes[0,col]
        target_line(ax,run)
        ct,cy=run.series("torque_nm",0xC0)
        ax.plot(ct,cy,color="tab:orange",alpha=.45,lw=.7,label="C0 reply current × B0")
        ax.plot(run.ttime,run.torque,color="tab:green",lw=.9,label="Independent A1 query current × B0")
        ax.set_title(run.label+" | current-derived torque, not shaft torque")
        panel(ax,(0.,7.),"Torque (Nm)")
        error=run.torque-held(run.ct,run.cv,run.ttime)
        axes[1,col].plot(run.ttime,error,color="tab:green",label="A1 torque − latest encoded request")
        axes[1,col].axhline(0,color="black",lw=.5)
        panel(axes[1,col],(0.,7.),"Torque tracking error (Nm)")
        axes[2,col].plot(run.vtime,run.v,color="tab:blue",label="A2 measured speed")
        panel(axes[2,col],(0.,7.),"Rear speed (rad/s)")
        axes[3,col].plot(run.qtime,run.q,color="tab:blue",label="A3 unwrapped displacement from startup")
        panel(axes[3,col],(0.,7.),"Rear displacement (rad)")
    for row in (2,3):
        limits=[axes[row,col].get_ylim() for col in (0,1)]
        lo,hi=min(x[0] for x in limits),max(x[1] for x in limits)
        for col in (0,1):
            axes[row,col].set_ylim(lo,hi)
    fig.suptitle("Rear replay: identical 50 Hz-held torque references, C0 packets at 50 versus 200 Hz\n"
                 "Measured torque is similar; wheel motion differs. One run per rate, different initial rotor angle.",
                 fontsize=13)
    save(fig,"rear_tracking")


def timing_plot(runs):
    fig,axes=plt.subplots(2,2,figsize=(18,9),layout="constrained")
    for run,ax in zip(runs,axes.flat):
        lateness=np.array([float(r["command_lateness_s"])*1000 for r in run.tx])
        t,queue=run.series("receive_queue_delay_s")
        ax.plot(t,queue*1000,color="tab:orange",alpha=.55,lw=.8,label="Kernel-RX to host-processing age")
        ax.plot(run.ct,lateness,color="tab:blue",lw=.8,label="Successful command send lateness")
        ax.axhline(3.,color="red",ls=":",label="Command abort threshold 3 ms")
        ax.set_title(run.label+f" | {'completed' if run.meta['completed'] else 'aborted'}")
        panel(ax,(0.,7. if run.drive else 8.),"Host timing (ms)",run)
        ax.set_ylim(-.15,max(4.6,float(queue.max()*1000)*1.1))
    fig.suptitle("Host timing only: an aborted command is absent from successful-TX lateness samples\n"
                 "These timestamps do not measure motor acquisition time, on-wire TX time or CPU scheduling cause",
                 fontsize=13)
    save(fig,"packet_timing")


def simulator_plot(runs,sims):
    fig,axes=plt.subplots(2,2,figsize=(18,10),layout="constrained")
    colors=("tab:green","tab:orange","tab:red")
    for row,drive in enumerate((False,True)):
        pair=[r for r in runs if r.drive==drive]
        for col,run in enumerate(pair):
            ax=axes[row,col]
            if not drive:
                target_line(ax,run)
                ax.plot(run.qtime,run.q,color="tab:blue",lw=1.3,label="Hardware A3 from true startup")
            else:
                ax.plot(run.vtime,run.v,color="tab:blue",lw=1.3,label="Hardware A2 speed")
            for (tag,sim),color in zip(sims.items(),colors):
                key="rear_speed" if drive else "steering"
                ax.plot(sim["time"],sim[key],color=color,alpha=.85,lw=.9,label=tag+" saved matched replay")
            ax.set_title(run.label+" versus the previous fixed-base simulator replay")
            panel(ax,CORE,"Rear speed (rad/s)" if drive else "Steering position (rad)",run)
    fig.suptitle("Same recorded target values, nominal supported simulator plants, real airborne hardware\n"
                 "Support fixtures are not independently metrologically matched; steering hardware zero is defective",
                 fontsize=13)
    save(fig,"simulator_comparison")


def export_samples(run):
    time,actual=(run.ttime,run.torque) if run.drive else (run.qtime,run.q)
    fields=("replay_time_s","actual","latest_encoded_reference","actual_minus_reference","in_recorded_core")
    path=HERE/(("rear" if run.drive else "steering")+f"_{run.meta['packet_hz']}hz_aligned.csv")
    reference=held(run.ct,run.cv,time)
    with path.open("w",newline="") as stream:
        writer=csv.writer(stream)
        writer.writerow(fields)
        writer.writerows((float(t),float(y),float(r),float(y-r),2<=t<7) for t,y,r in zip(time,actual,reference))


def write_report(runs,metrics,pairs,sims):
    s50,s200,d50,d200=(metrics[r.stem] for r in runs)
    def degrees(value):
        return float(np.degrees(value))
    rows=[]
    for run in runs:
        m=metrics[run.stem]
        rows.append(f"| {run.label} | {'Yes' if run.meta['completed'] else 'No: deadline at ~5.373 s'} | "
                    f"{m['transmitted_packets']}/{m['planned_packets']} | "
                    f"{m['effective_tx_hz']:.3f} | {m['command_intervals_ms']['p99']:.3f} |")
    steering=[]
    for m in (s50,s200):
        e=m["tracking"];c=m["common_window_tracking"];z=m["zero_reference"]
        steering.append(f"| {m['label']} | {degrees(e['rms']):.3f} | {degrees(e['bias']):+.3f} | "
                        f"{degrees(c['rms']):.3f} | {z['predicted_offset_deg']:.3f} | "
                        f"{degrees(e['demeaned_rms']):.3f} |")
    rear=[]
    for m in (d50,d200):
        e=m['tracking']
        rear.append(f"| {m['label']} | {e['rms']:.6f} | {e['bias']:+.6f} | {e['p95_abs']:.6f} | "
                    f"{m['speed_rad_s']['minimum']:.4f} | {m['observed_position_last_rad']:.6f} |")
    timing=[]
    for m in (s50,s200,d50,d200):
        timing.append(f"| {m['label']} | {m['command_intervals_ms']['median']:.3f} | "
                      f"{m['command_intervals_ms']['maximum']:.3f} | {m['command_lateness_ms']['maximum']:.3f} | "
                      f"{m['motion_receive_queue_age_ms']['maximum']:.3f} | "
                      f"{m['feedback_hz']['A1']:.2f} |")
    starting_angle=(d200['metadata']['encoder_zero_count']-d50['metadata']['encoder_zero_count'])*RAD_PER_COUNT
    steering_header=("| Run | Available-core A3 RMS error ° | Mean error ° | Common-window RMS ° | "
                     "Predicted zero bias ° | Demeaned error RMS ° |")
    rear_header=("| Run | A1 torque RMS error Nm | Mean error Nm | p95 absolute error Nm | "
                 "Minimum speed rad/s | Last A3 displacement rad |")
    timing_header=("| Run | Median interval ms | Maximum interval ms | Maximum successful-TX lateness ms | "
                   "Max motion RX queue age ms | A1 Hz |")
    initial_rows=[]
    for hz,m in ((50,s50),(200,s200)):
        initial_rows.append(f"| {hz} Hz | {m['metadata']['encoder_zero_count']} | "
                            f"{m['zero_reference']['initial_A3_native_rad']:.9f} | 3.000000000 | 65535 |")
    sim_peaks={}
    for tag,sim in sims.items():
        keep=(sim['time']>2)&(sim['time']<=7)
        sim_peaks[tag]=abs(sim['rear_speed'][keep].min())
    text=f"""# Real identical-target replay: steering and rear drive, 50 versus 200 Hz

Analysis date: 2026-10-26. All 12 supplied files were found: four measurement CSVs,
four plans and four metadata JSONs. No new motor/CAN operation or simulator run
was performed. Open [the figure gallery](index.html) or [exact statistics](statistics.json).

These recordings predate the steering B1 initialization fix. The matching
[runtime snapshots](runtime_snapshot/) preserve their exact source versions;
hash validation uses those snapshots rather than today's updated replay code.

## Findings

1. **A replay initialization bug biases physical steering by about 10–12 degrees.**
   Both initial MIT F1 readings are at the +3-rad encoding endpoint, although A3
   puts the straight-start shaft above +3 rad. The script accepted this endpoint
   as a valid zero. F1-relative tracking therefore looks good while actual
   startup-relative A3 displacement is too far right. This is a defect in the
   replay I wrote, not proof the steering motor itself has a 10-degree tracking error.
2. **Rear current-derived torque tracks reasonably closely at both rates.**
   A1 RMS error is {d50['tracking']['rms']:.4f} / {d200['tracking']['rms']:.4f} Nm at
   50 / 200 Hz. No clear torque-accuracy improvement is established at 200 Hz.
   Current times Kt is an electromagnetic torque estimate, not a shaft sensor.
3. **Rear movement differs greatly despite identical target payloads.** Peak
   backward speed is {abs(d50['speed_rad_s']['minimum']):.3f} versus
   {abs(d200['speed_rad_s']['minimum']):.3f} rad/s; final displacement is
   {d50['observed_position_last_rad']:.3f} versus {d200['observed_position_last_rad']:.3f} rad.
   Near-zero-speed behavior is not adequately established by the old moving-drag fit.
   One run per rate, different initial rotor angles and undersampled current
   transients prevent assigning this difference uniquely to packet rate.
4. **The 200-Hz steering run is incomplete because of a host deadline.** It
   sent 1,074/1,600 planned packets, then disabled successfully. Rear 200 Hz
   completed. This does not establish a motor/CAN 200-Hz limit or a firmware rate.

## Provenance, fixture and comparison boundary

The user confirms both tested wheels airborne/free, no restraint or hand
contact, straight steering startup, actual steering right, and no configuration
or fixture changes between rates. Only one motor was tested at a time.

All four runs use the exact same IsaacLab-source CSV, model **2500**, training
run **2026-10-21_11-18-29**, DR evaluation seed **42**, env **0**, episode **1**.
Source file: `replay_targets/2500_isaac400_env0_episode1.csv`, SHA-256
`{s50['metadata']['targets_sha256']}`. The raw CSV/JSON bytes, captured runtime hashes,
exact CAN encoding and every transmitted target were checked. Both rate pairs
have byte-identical plans; encoded command payloads match for every common
50-Hz interval. At 200 Hz each reference is repeated four times, not recomputed.

Steering has 1 s ramp + 1 s hold + core 2–7 s + 1 s final hold. Rear has 2 s
zero torque + core 2–7 s, then CF. Core source timestamps are 12.50–17.50 s.
The aborted steering comparison uses common complete intervals **2–5.36 s**;
the remaining successful 50-Hz trace is separately reported. There are no
invented episode resets, interpolated target ramps or motion samples after abort.

| Run | Completed | Packets sent/planned | Achieved host TX Hz | p99 interval ms |
| --- | --- | ---: | ---: | ---: |
{chr(10).join(rows)}

All four JSONs confirm disabled, fault 0. Both rear runs confirm CD restored.
All recorded active steering AE modes are 5 and F1 statuses 1; rear modes are 2.
All finished motion query records say received; this is not proof of zero wire
loss or known motor acquisition age. Supply stays about 20.2 V, with no recorded
large sag; the CSVs do not establish the firmware's internal control frequency.

## Steering: actual position versus a clipped reference

{steering_header}
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(steering)}

Errors above use fresh A3 multi-turn displacement from the real startup count
against the latest actually sent encoded target, with no shifting or bias removal.
Positive error means too far right. The last column removes only the mean error
to describe shape variation; it does **not** make physical steering correct.

| Rate | Startup A3 count | A3 native rad | Recorded F1 zero rad | F1 code |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(initial_rows)}

F0 position range is ±3 rad in both records. The script's absolute MIT command
is `recorded_F1_zero - requested_right_angle`, with requested velocity and
feedforward torque zero. When the actual initial native angle exceeds the
recorded F1 value, the requested displacement includes that missing offset.
The initial differences are **{s50['zero_reference']['predicted_offset_rad']:.6f} rad**
and **{s200['zero_reference']['predicted_offset_rad']:.6f} rad**. In the core,
interpolating A3 only to cross-check asynchronous F1 gives mean A3-relative minus
F1-relative differences **{s50['zero_reference']['measured_A3_minus_F1_relative_rad']['mean']:.6f}**
and **{s200['zero_reference']['measured_A3_minus_F1_relative_rad']['mean']:.6f} rad**.
The close agreement, plus endpoint plateaus during startup, strongly supports
F1 clipping as the cause. No firmware source or independent angle transducer is
available, so internal saturation implementation is not claimed independently.

F1-based core RMS errors are only
{degrees(s50['zero_reference']['F1_based_tracking']['rms']):.3f}° and
{degrees(s200['zero_reference']['F1_based_tracking']['rms']):.3f}°: using that channel
alone hides the origin error. Native A3 counts decrease and signed A3 displacement
increases while the user observes right steering; this supports the configured
position sign, rather than a sign inversion. Reported torque uses sign −1 too.

The corrected-origin *diagnostic* tracking errors and descriptive lag scans are
saved in JSON. They are not silently substituted into any physical-response plot.
Tracking dynamics appear much closer than the 10–12° absolute errors suggest,
but clean repeat tests require a valid, unsaturated startup reference first.
This replay defect does not establish the cause of earlier policy deployment
failures: normal `motor_mit_io` has a different startup/origin procedure.

## Rear: current tracks, motion is not repeatable in this pair

{rear_header}
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rear)}

The shared recorded request spans −0.233069 to +0.009307 Nm; encoded C0 spans
approximately −0.233100 to +0.009300 Nm. The negative request means backward
torque under the verified signs; negative measured wheel rotation agrees.
The trace has little positive demand and cannot independently validate strong
forward-torque behavior. Commands are `I_native=-torque/B0`, not a velocity target.

Quantization is at most half a milliamp times B0 (~0.00015 Nm). Error metrics use
the independent A1 query stream (~99 Hz) for both rates, not differently sampled
C0 replies. C0 replies are also plotted as actual measured-current reports.
During initial zero commands, A1 RMS is
{d50['zero_command_noise_nm']['rms']:.4f}/{d200['zero_command_noise_nm']['rms']:.4f} Nm.
Thus current-report/current-loop noise already accounts for a substantial scale
of the core error; the CSV cannot separate true torque ripple from reporting noise.

Wheel speed is within ±0.1 rad/s for
{d50['near_zero_speed_core_fraction']*100:.1f}% / {d200['near_zero_speed_core_fraction']*100:.1f}%
of core A1 samples using the latest A2 reading. At these near-zero speeds,
95th-percentile absolute current-derived torque is
{d50['near_zero_speed_torque_abs_nm']['p95']:.4f}/
{d200['near_zero_speed_torque_abs_nm']['p95']:.4f} Nm, above the prior 0.106 Nm
**moving** drag fit. The old speed sweep explicitly did not identify static
breakaway friction. This supports investigating static/angle-dependent drag,
cogging and small-command current dynamics; it does not measure a new universal
friction coefficient or justify adding an arbitrary viscous term.

The starting raw A3 counts differ by {d200['metadata']['encoder_zero_count']-d50['metadata']['encoder_zero_count']}
counts ({starting_angle:.6f} rad, {degrees(starting_angle):.2f}°), so wheel/rotor
phase was not controlled even though firmware settings were unchanged. The mean
torque errors are small, but sub-query transients can differ and are not fully
observed. Rate changes, initial angle and stick-slip effects are confounded.
There is no defensible conclusion that smaller wheel motion means better
200-Hz torque control, or that the motor cannot produce the requested current.

## Timing and delay

{timing_header}
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(timing)}

The aborted 200-Hz steering command was scheduled at **5.370 s**; the first
logged shutdown TX is at **{s200['first_logged_shutdown_tx_s']:.6f} s**. It had
exceeded the 3-ms host lateness threshold. Earlier successful intervals can be
shorter/longer than 5 ms as the scheduler returns toward its absolute grid.
The rejected packet has no successful-TX row. Kernel queue delays demonstrate
host-processing stalls, but do not identify Linux scheduling versus Python
logging contention, CPU load or another specific cause. No profiler was recorded.

Best time-shift fits are descriptive. A 0–100 ms scan compares measured values
to the actual zero-order-held sent targets; steering fits remove a constant
bias, rear fits do not. In the common steering window, best shifts are
{s50['common_window_alignment']['best_shift_ms']:.1f}/
{s200['common_window_alignment']['best_shift_ms']:.1f} ms. Rear best shifts over
the full core are {d50['alignment']['best_shift_ms']:.1f}/
{d200['alignment']['best_shift_ms']:.1f} ms. Interpret these as approximately
**tens of milliseconds for steering response** and **a few milliseconds for
sampled current response**, not fixed pure latency. A1/A3 sampling is about
10 ms, waveforms are not broadband identification inputs, acquisition time is
unknown and timing/noise biases remain. The scan grid is not measurement precision;
the saved near-minimum intervals are not statistical confidence intervals.

## Comparison with the saved simulator replay

The previous supported nominal replay uses exactly the same 50-Hz targets
(Isaac-source case 4, steering losses ON) in Isaac400, MuJoCo400 and MuJoCo800.
No new simulation was run. Steering true-startup A3 is offset from the simulated
response because the hardware reference is biased. Simulated rear peak backward
speeds in the core are **{sim_peaks['isaac400']:.2f}**, **{sim_peaks['mujoco400']:.2f}**
and **{sim_peaks['mujoco800']:.2f} rad/s**,
well above the hardware pair's 8.05/2.83 rad/s. Motor torque submissions in those
simulators are ideal; current dynamics/static drag/initial phase are not faithfully
identified. Real fixture versus fixed-root support, nominal parameters and the
steering-origin defect also limit a strict physical-equivalence claim.

## Next steps, in priority order

1. **Fix steering startup-reference validation**, then repeat. Reject endpoint
   F1 zeros and verify A3/F1 consistency. Establish an unsaturated native MIT
   reference with sufficient range for the straight position and target path;
   verify any protocol/range adjustment before actuation. Do not hide the error
   by subtracting 10° from the plotted physical position or change the sign.
2. **Make the timing experiment finish and record the missed deadline itself.**
   Measure host workload and queue age; reduce hot-loop/logging contention or
   revise the explicitly documented abort tolerance from measured Pi jitter.
   Keep actual send timestamps and avoid dropped/burst targets. The present pair
   supports neither a universal 200-Hz failure nor a need for 800-Hz Pi commands.
3. **Identify rear near-zero/breakaway behavior.** Repeat both rates in alternating
   order from the same marked rear-wheel angle, at least several times, with the
   same supply/temperature. Add controlled small-torque plateaus/ramp tests and
   current logging fast enough to resolve transients. Measure static threshold
   versus angle separately from moving drag; independently calibrate torque if
   a mechanical accuracy claim is needed. A 0.106-Nm moving fit is not that test.
4. **Replay faster MuJoCo target sequences only after these checks.** The present
   Isaac core has small steering variation around a large rightward offset and
   small rear torque. It does not validate 10–25 Hz steering reversal tracking,
   loaded tyre-ground behavior, full motor envelopes or unsupported balance.

## Reproduction and artifacts

From repository root:

```bash
python3 -B deploy/experiments/reports/20261026_policy_replay_hardware/analyze.py
python3 -B -m unittest discover -s deploy/experiments/reports/20261026_policy_replay_hardware -v
```

Figures: [steering](steering_tracking.png), [zero-reference defect](steering_zero_reference.png),
[rear torque/motion](rear_tracking.png), [host timing](packet_timing.png),
[saved simulator comparison](simulator_comparison.png). Each also has SVG/PDF.
Legends are at bottom right; physical min/max annotations exclude vertical markers.
Plots show raw, unshifted samples. Only the explicitly labelled cross-check/lag
calculations use interpolation or a shifted reference; no plotted response is altered.
Four `*_aligned.csv` files expose fresh A3 (rad) or A1 (Nm) against the latest
actually sent encoded reference. [statistics.json](statistics.json) includes
full metrics/provenance, and [verification.json](verification.json) records tests
and input/production preservation. Original logs, runtime code, models, training
data, earlier reports and progress ledgers remain unchanged.
"""
    (HERE/"README.md").write_text(text)
    cards="\n".join(f'<section><h2>{html.escape(name.replace("_"," "))}</h2>'
                     f'<a href="{name}.png"><img src="{name}.png" alt="{name}"></a>'
                     f'<p><a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p></section>'
                     for name in FIGURES)
    (HERE/"index.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Real recorded-target replay — 50/200 Hz</title>
<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#20252b}
img{width:100%;height:auto}section{margin-top:40px}a{color:#155bb5}.note{background:#fff3d8;padding:18px}</style>
<h1>Real recorded-target replay: model 2500, 50 versus 200 Hz</h1>
<p class="note">Steering has a clipped startup MIT zero; use A3 startup-relative displacement to assess
physical accuracy. Rear current tracking is similar, while wheel motion differs. Steering 200 Hz aborted.
All tests were airborne/free; no plotted curve is shifted or bias-corrected.</p>
<p><a href="README.md">Full report</a> · <a href="statistics.json">Exact statistics</a> ·
<a href="verification.json">Verification</a></p>'''+cards+'</html>')


def main():
    runs=[Run(stem) for stem in FILES]
    for run in runs:
        run.validate()
    pairs=compare_pairs(runs)
    metrics={run.stem:run.metrics() for run in runs}
    provenance={str((EXPERIMENTS/"logs"/(stem+suffix)).relative_to(ROOT)):
                digest(EXPERIMENTS/"logs"/(stem+suffix))
                for stem in FILES for suffix in (".csv",".json","_plan.csv")}
    sims=simulator_data()
    statistics=dict(runs=metrics,pair_validation=pairs,input_sha256=provenance,
        user_confirmed_conditions=dict(wheels_airborne_free=True,no_restraint=True,
            steering_initially_straight=True,steering_physically_right=True,configurations_unchanged=True),
        sampling_method="Errors against latest actual sent command, A3 steering / independent A1 rear; no shifting",
        units=dict(steering="rad",rear_torque="Nm",velocity="rad/s",timing="ms"))
    (HERE/"statistics.json").write_text(json.dumps(statistics,indent=2,allow_nan=False)+'\n')
    for run in runs:
        export_samples(run)
    steering_plots(runs)
    rear_plots(runs)
    timing_plot(runs)
    simulator_plot(runs,sims)
    write_report(runs,metrics,pairs,sims)
    for run in runs:
        m=metrics[run.stem]
        print(run.label, 'completed',run.meta['completed'],'error RMS',m['tracking']['rms'])
    print('Output:',HERE)


if __name__=="__main__":
    main()
