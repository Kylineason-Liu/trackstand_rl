"""Six figure sets for rear MIT PI, P-only and C1; native measurements retained."""

import argparse
import csv
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

from analyze import HERE, MODES, captures, integral, write_csv

COLORS = {"C1":"#2166ac","MIT":"#dd7c20","PI":"#358b52"}
LABELS = {"C1":"C1 PI","MIT":"MIT P","PI":"MIT PI"}
SELECTIONS = [("steady",.1,0.,0.),("steady",.4,0.,0.),("sine",0.,.25,.5),("sine",0.,1.,.5)]


def rows(name):
    with (HERE/name).open() as f:
        return [{k:v if k in ("mode","name","kind") else float(v) if v else None for k,v in r.items()}
                for r in csv.DictReader(f)]


def phase_for(phases,kind,base,amp,f,repeat=3):
    return next(p for p in phases if p["segment"]["kind"]==kind and p["segment"]["base_rad_s"]==base
                and p["segment"]["amplitude_rad_s"]==amp and p["segment"]["frequency_hz"]==f
                and p["segment"]["repeat"]==repeat)


def finish(ax):
    ax.grid(alpha=.24)
    ax.legend(loc="lower right",fontsize=7)


def extrema(ax):
    arrays = [np.asarray(l.get_ydata()) for l in ax.lines if len(l.get_ydata())>2]
    if arrays:
        y = np.concatenate(arrays)
        ax.text(.015,.97,f"Shown min/max: {y.min():.3g} / {y.max():.3g}",transform=ax.transAxes,
                va="top",fontsize=7,bbox=dict(facecolor="white",edgecolor="none",alpha=.8))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache",type=Path,required=True)
    args = ap.parse_args()
    data = captures(args.cache)
    steady,sine = rows("steady_summary.csv"),rows("sine_summary.csv")
    law = json.loads((HERE/"current_response_fit.json").read_text())["pi_current_response"]
    plt.rcParams.update({"font.size":10,"legend.fontsize":8,"savefig.dpi":175})
    gallery,exported = [],[]
    with PdfPages(HERE/"all_figures.pdf") as pdf:

        def save(fig,name,caption):
            fig.tight_layout(rect=(0,.045,1,.94))
            for suffix in ("png","svg","pdf"):
                fig.savefig(HERE/f"{name}.{suffix}")
            pdf.savefig(fig)
            gallery.append((name,caption))
            plt.close(fig)

        fig,axes = plt.subplots(2,2,figsize=(13,8))
        for mode in MODES:
            rr = sorted([r for r in steady if r["mode"]==mode],key=lambda r:r["base_rad_s"])
            for ax,narrow in ((axes[0,0],False),(axes[0,1],True)):
                part = [r for r in rr if not narrow or abs(r["base_rad_s"])<=.8]
                x = [r["base_rad_s"] for r in part]
                ax.plot(x,[r["velocity_mean_rad_s_mean"] for r in part],"o-",color=COLORS[mode],label=LABELS[mode])
                ax.fill_between(x,[r["velocity_mean_rad_s_min"] for r in part],
                                [r["velocity_mean_rad_s_max"] for r in part],color=COLORS[mode],alpha=.13)
            x = [r["base_rad_s"] for r in rr]
            axes[1,0].plot(x,[r["velocity_error_mean_rad_s_mean"] for r in rr],"o-",
                           color=COLORS[mode],label=LABELS[mode])
            axes[1,1].plot(x,[r["torque_mean_nm_mean"] for r in rr],"o-",color=COLORS[mode],label=LABELS[mode])
        for ax,limit in ((axes[0,0],10.5),(axes[0,1],.85)):
            ax.plot([-limit,limit],[-limit,limit],"k--",linewidth=.8,label="Perfect tracking")
            ax.set(xlabel="Intended target [rad/s]",ylabel="Mean fresh A2 speed [rad/s]")
        axes[0,0].set_title("Steady response: all levels")
        axes[0,1].set_title("Near zero: integral restores average movement")
        axes[1,0].set(xlabel="Intended target [rad/s]",ylabel="Mean A2 − effective target [rad/s]",
                       title="Steady velocity bias")
        axes[1,1].set(xlabel="Intended target [rad/s]",ylabel="Mean A1 × B0 torque [Nm]",
                       title="Integral maintains load torque at small mean error")
        for ax in axes.flat:
            finish(ax)
        fig.suptitle("Rear velocity control: MIT PI (P=0.30, I=3.3), MIT P and C1 PI")
        fig.text(.5,.01,"Three repeats; lines = means, shading = range. Window 2–5 s. "
                 "Airborne/free; measured current-derived torque, not independent shaft torque.",ha="center",fontsize=9)
        save(fig,"steady_response","Integral removes the P-only average-speed deficit.")

        for metric,name in (("fundamental_gain","sine_amplitude"),("velocity_error_rms_rad_s","sine_error")):
            fig,axes = plt.subplots(2,3,figsize=(16,8),sharex=True)
            for row,amp in enumerate((.25,1.)):
                for col,base in enumerate((0.,-3.,3.)):
                    ax = axes[row,col]
                    for mode in MODES:
                        part = sorted([r for r in sine if r["mode"]==mode and r["base_rad_s"]==base
                                       and r["amplitude_rad_s"]==amp],key=lambda r:r["frequency_hz"])
                        x = [r["frequency_hz"] for r in part]
                        ax.plot(x,[r[metric+"_mean"] for r in part],"o-",color=COLORS[mode],label=LABELS[mode])
                        ax.fill_between(x,[r[metric+"_min"] for r in part],[r[metric+"_max"] for r in part],
                                        color=COLORS[mode],alpha=.13)
                        if name=="sine_error":
                            ax.plot(x,[abs(r["velocity_error_mean_rad_s_mean"]) for r in part],"--",
                                    color=COLORS[mode],linewidth=.8)
                    ax.set_xscale("log")
                    ax.set_xticks([.5,1,2,5,10,20],["0.5","1","2","5","10","20"])
                    ax.set(xlabel="Frequency [Hz]",title=f"Baseline {base:g}, sine ±{amp:g} rad/s",
                           ylabel="Response fundamental / effective target" if name=="sine_amplitude"
                           else "Velocity RMS / |mean bias| [rad/s]")
                    if name=="sine_amplitude":
                        ax.axhline(1,color="black",linestyle="--",linewidth=.8)
                        top = max(r[metric+"_max"] for r in sine if r["base_rad_s"]==base
                                  and r["amplitude_rad_s"]==amp)
                        ax.set_ylim(-.03,max(1.2,top*1.06))
                    finish(ax)
            fig.suptitle("MIT PI improves slow tracking; tiny rapid reversals remain weak")
            fig.text(.5,.01,"Three-repeat means/ranges. Actual held P targets, with FF codec residual removed. "
                     "RMS is unshifted; dashed error-panel curves show |mean bias|.",ha="center",fontsize=9)
            save(fig,name,"Sine amplitude across all matched conditions" if name=="sine_amplitude"
                 else "Sine error including residual bias")

        fig,axes = plt.subplots(3,4,figsize=(20,10))
        for col,(kind,base,amp,f) in enumerate(SELECTIONS):
            start,stop = (0.,5.) if kind=="steady" else (1.,9.)
            for mode,(meta,phases,_) in data.items():
                p = phase_for(phases,kind,base,amp,f)
                ct,command = p["streams"]["command"]
                for row,(key,field) in enumerate((("velocity","a2_velocity_rad_s"),
                                                   ("current","a1_torque_nm"),("position","a3_relative_position_rad"))):
                    t,y = p["streams"][key]
                    keep = (t>=start) & (t<stop)
                    t,y = t[keep],y[keep].copy()
                    if row==1:
                        y *= meta["torque_constant_nm_per_a"]
                    elif row==2:
                        y -= y[0]
                    axes[row,col].plot(t,y,color=COLORS[mode],linewidth=.8,label=LABELS[mode])
                    exported.extend(dict(panel=f"r{row}_c{col}",mode=mode,field=field,time_s=float(a),value=float(b))
                                    for a,b in zip(t,y))
                if mode=="PI":
                    keep = (ct>=start) & (ct<stop)
                    axes[0,col].step(ct[keep],command[keep],where="post",color="black",linestyle="--",
                                     linewidth=1,label="PI effective P target")
                    pt = p["streams"]["position"][0]
                    origin = pt[(pt>=start) & (pt<stop)][0]
                    rt = np.linspace(origin,stop,1000)
                    rq = integral(ct,command,rt)-integral(ct,command,np.array([origin]))[0]
                    axes[2,col].plot(rt,rq,color="black",linestyle="--",linewidth=.8,label="PI target integral")
            axes[0,col].set_title(f"Steady +{base:g} rad/s" if kind=="steady" else f"Zero-base sine ±{amp:g}, {f:g} Hz")
            for row in range(3):
                ax = axes[row,col]
                ax.set(xlim=(start,stop),xlabel="Time from segment start [s]",
                       ylabel=("Velocity [rad/s]","Current-derived torque [Nm]","Relative displacement [rad]")[row])
                finish(ax); extrema(ax)
        fig.suptitle("Matched targets, repeat 3: native speed, current-derived torque and encoder travel")
        fig.text(.5,.01,"Separate runs; no smoothing or fitted time/bias shift. Each A3 subtracts its own first "
                 "shown sample. Same intended targets; the plotted reference is PI's codec-corrected P target.",
                 ha="center",fontsize=9)
        save(fig,"representative_traces","Matched near-zero and sine responses, with encoder travel")
        write_csv(HERE/"plotted_response_samples.csv",exported)

        fig,axes = plt.subplots(2,3,figsize=(16,7))
        for col,(base,amp) in enumerate(((0.,.1),(0.,1.),(2.,.25))):
            for mode,(meta,phases,_) in data.items():
                p = phase_for(phases,"step",base,amp,0.)
                for row,key in enumerate(("velocity","current")):
                    t,y = p["streams"][key]
                    keep = (t>=.4) & (t<1.3)
                    scale = meta["torque_constant_nm_per_a"] if row else 1.
                    axes[row,col].plot(t[keep],y[keep]*scale,color=COLORS[mode],linewidth=.8,label=LABELS[mode])
                if mode=="PI":
                    ct,c = p["streams"]["command"]
                    keep = (ct>=.4) & (ct<1.3)
                    axes[0,col].step(ct[keep],c[keep],where="post",color="black",linestyle="--",
                                     linewidth=.8,label="Target")
            axes[0,col].set_title(f"Baseline {base:g}, pulse +{amp:g} rad/s")
            for row in range(2):
                ax = axes[row,col]
                for t in (.6,1.):
                    ax.axvline(t,color="red",linewidth=.8,alpha=.7)
                ax.set(xlim=(.4,1.3),xlabel="Time from segment start [s]",
                       ylabel="Current-derived torque [Nm]" if row else "Velocity [rad/s]")
                finish(ax); extrema(ax)
        fig.suptitle("Matched step responses, repeat 3; red lines bound the 0.4-s pulse")
        fig.text(.5,.01,"Shown 0.4–1.3 s to resolve the pulse after the initial baseline transient. "
                 "Integral resets once per segment. Native A2/A1; no smoothing/time shift.",ha="center",fontsize=9)
        save(fig,"step_responses","Small and larger velocity steps")

        meta,phases,_ = data["PI"]
        fig,axes = plt.subplots(2,3,figsize=(17,7.5))
        for col,(kind,base,amp,f,start,stop) in enumerate((("steady",.1,0.,0.,0.,5.),
                                                        ("sine",0.,.25,.5,1.,5.),("sine",0.,.25,1.,1.,5.))):
            p = phase_for(phases,kind,base,amp,f)
            ct,target = p["streams"]["command"]
            vt,v = p["streams"]["velocity"]
            it,current = p["streams"]["current"]
            keep = (ct>=start) & (ct<stop)
            vk = (vt>=start) & (vt<stop)
            ik = (it>=start) & (it<stop)
            axes[0,col].step(ct[keep],target[keep],where="post",color="black",linestyle="--",label="P target")
            axes[0,col].plot(vt[vk],v[vk],color=COLORS["PI"],linewidth=.8,label="Fresh A2")
            axes[1,col].plot(it[ik],current[ik]*meta["torque_constant_nm_per_a"],color="#2166ac",
                             linewidth=.8,label="Fresh A1 × B0")
            i_tau = p["streams"]["integral"][1]
            total = p["streams"]["estimated_torque"][1]
            for y,color,label in ((total,"#c73b3b","Sent P+I estimate"),(i_tau,"#9659a7","Host I torque"),
                                  (total-i_tau,"#dd7c20","P component")):
                axes[1,col].step(ct[keep],y[keep],where="post",color=color,linewidth=.8,label=label)
            axes[0,col].set_title(f"Steady +{base:g} rad/s" if kind=="steady" else f"Sine ±{amp:g}, {f:g} Hz")
            for row in range(2):
                ax = axes[row,col]
                ax.set(xlim=(start,stop),xlabel="Time from segment start [s]",
                       ylabel="Velocity [rad/s]" if row==0 else "Controller/current torque [Nm]")
                finish(ax); extrema(ax)
        fig.suptitle("MIT PI torque decomposition: integral supports movement but cannot cure every reversal")
        fig.text(.5,.01,f"Command estimates use latest A2 at send; A1 is independently queried, unsmoothed. "
                 f"Known-law held-out current R²={law['heldout_repeat3_known_pi']['r_squared']:.3f}; "
                 "this is not forward plant prediction.",ha="center",fontsize=9)
        save(fig,"pi_controller","P and host I contributions compared with current telemetry")

    html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Rear MIT PI comparison</title>
<style>body{font:16px system-ui;max-width:1550px;margin:30px auto;padding:0 20px;line-height:1.5}
img{max-width:100%}</style></head><body><h1>Rear MIT PI versus P-only and C1</h1>
<p>Host integral restores steady tracking. Tiny rapid reversals remain weak.</p>
<p><a href="README.md">Report</a> · <a href="all_figures.pdf">Six-page PDF</a> ·
<a href="steady_summary.csv">Steady table</a> · <a href="sine_summary.csv">Sine table</a> ·
<a href="raw_audit.json">Raw audit</a> · <a href="current_response_fit.json">Current-law check</a></p>'''
    for name,caption in gallery:
        html += f'<h2>{escape(caption)}</h2><a href="{name}.png"><img src="{name}.png" '
        html += f'alt="{escape(caption)}"></a><p><a href="{name}.svg">SVG</a> · '
        html += f'<a href="{name}.pdf">PDF</a></p>'
    (HERE/"index.html").write_text(html+"</body></html>")


if __name__ == "__main__":
    main()
