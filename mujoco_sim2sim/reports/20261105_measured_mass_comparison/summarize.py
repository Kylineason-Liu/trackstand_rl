from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;R=H.parents[2]
rows=[];paired=[]
fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for ax,(v,hz) in zip(axs,[('v1',400),('v1',800),('v2',1200)]):
 lengths=[];flags=[]
 for label,color in [('current','tab:blue'),('measured','tab:orange')]:
  folder=H/f'{v}_{hz}hz_{label}';s=json.loads((folder/'summary.json').read_text());e=json.loads((folder/'episodes.json').read_text());rows.append(s)
  assert len(e)==64 and all(r['steps']<=1500 for r in e)
  assert all(not any(value for name,value in r['flags'].items() if name not in ('time_out','fallen')) for r in e)
  x=np.array([r['seconds'] for r in e]);t=np.array([r['timeout'] for r in e]);lengths.append(x);flags.append(t)
  ax.stairs(np.histogram(x,bins=np.arange(0,33,3))[0],np.arange(0,33,3),label=f'{label}: {sum(t)}/64 survived',color=color,linewidth=2)
  ax.set(title=f'{v} DR | {hz} Hz | pure survival',xlabel='Episode length (s)',ylabel='Episodes');ax.legend(fontsize=8);ax.grid(alpha=.2)
 old,new=lengths;diff=new-old;rng=np.random.default_rng(42);boot=diff[rng.integers(0,64,(10000,64))].mean(axis=1)
 p={'version':v,'hz':hz,'mean_delta_seconds':float(diff.mean()),'paired_bootstrap_mean_delta_95ci':np.percentile(boot,[2.5,97.5]).tolist(),'improved':int(sum(diff>0)),'worse':int(sum(diff<0)),'equal':int(sum(diff==0)),'survival_gained':int(sum(~flags[0]&flags[1])),'survival_lost':int(sum(flags[0]&~flags[1]))};paired.append(p)
 # exact DR sampling parity, including all saved coefficients.
 assert (H/f'{v}_{hz}hz_current/dr.json').read_bytes()==(H/f'{v}_{hz}hz_measured/dr.json').read_bytes()
fig.savefig(H/'episode_length_comparison.png',dpi=180);fig.savefig(H/'episode_length_comparison.svg');plt.close(fig)
with (H/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(H/'paired_comparison.json').write_text(json.dumps(paired,indent=2)+'\n')
man=json.loads((H/'manifest.json').read_text())
for value in [*man['models'].values(),*man['policies'].values()]:assert hashlib.sha256((R/value['path']).read_bytes()).hexdigest()==value['sha256']
report='Measured mass comparison, 2026-11-05\n\n'
report+='V1 DR model_2500 (2026-10-21_11-18-29), V2 DR model_50100 (2026-11-02_22-40-01).\n64 first episodes per case, environment seeds 42..105, DR ON, pure survival\n(time_out or fallen), 30-s horizon, policy 50 Hz. No first-N-completion selection:\nevery environment finishes exactly once. No retraining. Within each version/rate,\noriginal and measured models use exactly identical saved DR samples. DR is\ncentered on each model nominal mass, not fixed to the old mass distribution.\nCross-version DR contracts differ, so compare each version with its own baseline.\n\n'
report+='Only four XML mass attributes changed. Original XML/model code/policies retained.\nNew model is beside original and shares its meshes. Front fork maps to steering_link.\nTotal nominal old=2.4243294124 kg, new=2.44410 kg (+0.8155%).\nInertias and centers of mass retained: this is an isolated mass sensitivity test,\nnot a claim that the measured assembly inertia has been calibrated. Existing DR\nstill scales inertia by its per-body random mass multiplier in BOTH models.\n\n'
for s in rows:report+=f"{s['version']} {s['physics_hz']} Hz {s['mass_configuration']}: survived {s['timeouts']}/64 ({s['timeout_rate']*100:.2f}%), mean {s['mean_seconds']:.3f} s, median {s['median_seconds']:.3f} s.\n"
report+='\nPaired differences (measured minus current), 10000 paired bootstrap resamples:\n'
for p in paired:report+=json.dumps(p)+'\n'
report+='\nInterpret small changes cautiously with only 64 paired seeds; CIs quantify\nseed sensitivity, not model correctness. Results do not predict hardware success.\n\nReproduce from repository root:\nOPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -u mujoco_sim2sim/reports/20261105_measured_mass_comparison/evaluate.py\nMPLCONFIGDIR=/tmp/trackstand_mass_plot python3 mujoco_sim2sim/reports/20261105_measured_mass_comparison/summarize.py\n\nEvidence: manifest.json with hashes; per-case episodes.json, summary.json,\ndr.json, sampled_masses.json; run.log; summary.csv and paired_comparison.json.\nSingle-thread ONNX CPU inference; same production env/model/policy classes as\nviewer, with fixed first-episode budget in this report helper. Model/policy hash checks\nand exact DR equality checks passed; 384 episodes completed across 6 cases.\n'
(H/'report.txt').write_text(report)
print(report)
