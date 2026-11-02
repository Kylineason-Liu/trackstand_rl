"""Report-local plots; executed by analyze.py after its contract assertions."""

from html import escape


plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .22,
                     "savefig.dpi": 160, "figure.dpi": 110})
BLUE, ORANGE, GREEN, RED, GREY = "#2463b4", "#ed8a19", "#18825d", "#c83333", "#666666"
FIGURES = []


def savefig(fig, name, title, directory=HERE):
    directory.mkdir(exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{name}.{ext}", bbox_inches="tight")
    FIGURES.append((str((directory / name).relative_to(HERE)), title))


def decorate(ax, ylabel=None):
    if ylabel:
        ax.set_ylabel(ylabel)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="lower right", fontsize=7, framealpha=.88)
    ax.margins(x=.005)


def extrema(ax, values, unit=""):
    x = np.concatenate([np.ravel(v) for v in values])
    x = x[np.isfinite(x)]
    if len(x):
        ax.text(.01, .98, f"range: {x.min():.3g} to {x.max():.3g} {unit}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(facecolor="white", alpha=.75, edgecolor="none"))


def shade_attempts(axes):
    for ep in EPISODES:
        for ax in np.ravel(axes):
            ax.axvspan(ep["start_s"], ep["stop_s"], color=GREEN, alpha=.10, lw=0)
            ax.axvline(ep["start_s"], color=RED, alpha=.4, lw=.6)


fig, axes = plt.subplots(6, 1, figsize=(16, 14), sharex=True, constrained_layout=True)
fig.suptitle("October 29 full deployment | Trackstand-Balance-v1 model_2500 | 21 policy-enabled attempts\n"
             "Green intervals: policy enabled; red lines: first successful policy tick; MCAP reception time")
axes[0].plot(IMU[:, 0], ROLL, color=BLUE, lw=.8, label="IMU-quaternion roll")
for value in (-30, 30):
    axes[0].axhline(value, color=RED, ls="--", lw=.7)
axes[1].plot(ST[:, 0], ST[:, 1], color=GREEN, lw=.7, label="A3 steering feedback")
axes[1].scatter(ENABLED[:, 0], ENABLED[:, 2], s=2, color=ORANGE, label="50 Hz held steering request")
axes[2].plot(ST[:, 0], ST[:, 7], color=GREEN, lw=.7, label="A1 × Kt rear torque estimate")
axes[2].scatter(ENABLED[:, 0], ENABLED[:, 3], s=2, color=ORANGE, label="50 Hz rear torque request")
axes[3].plot(ST[:, 0], ST[:, 4], color=BLUE, lw=.7, label="A2 rear speed")
for value in (-61.6, 61.6):
    axes[3].axhline(value, color=RED, ls="--", lw=.7)
axes[4].plot(MC[:, 0], col("motor_control", "steering_pair_age_ms"), color=BLUE, lw=.65,
             label="Steering encoder-pair age")
axes[4].plot(MC[:, 0], col("motor_control", "drive_pair_age_ms"), color=ORANGE, lw=.65,
             label="Rear encoder-pair age")
axes[4].axhline(50, color=RED, ls="--", label="Motor I/O 50 ms freshness gate")
axes[4].set_ylim(0, 190)
axes[5].step(PT[:, 0], col("policy_timing", "policy_enabled"), where="post", color=GREEN, label="Policy enabled")
axes[5].step(A["motor_fault"][:, 0], A["motor_fault"][:, 1], where="post", color=RED, label="Motor fault")
axes[5].set_ylim(-.1, 1.4)
for ax, ylabel in zip(axes, ("Roll (deg)", "Steering (rad)", "Rear torque (Nm)", "Rear speed (rad/s)", "Age (ms)", "Flag")):
    decorate(ax, ylabel)
    ax.axvspan(4.008, 17.368, color=GREY, alpha=.13)
shade_attempts(axes)
for ep in EPISODES:
    axes[0].text(ep["start_s"], 64, str(ep["episode"]), fontsize=7, ha="center")
axes[0].set_ylim(-73, 73)
axes[-1].set_xlabel("Seconds since first bag message")
savefig(fig, "overview", "Full recording and attempt timeline")
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
colors = [GREEN if ep["reason"] == "RC_down" else RED if ep["reason"].startswith("roll") else ORANGE
          for ep in EPISODES]
ax.bar([ep["episode"] for ep in EPISODES], [ep["duration_s"] for ep in EPISODES], color=colors)
for ep in EPISODES:
    ax.text(ep["episode"], ep["duration_s"] + .03, f"{ep['duration_s']:.2f}", ha="center", fontsize=8)
ax.set(xticks=range(1, 22), xlabel="Policy episode", ylabel="Enabled duration (s)", ylim=(0, 2.65))
ax.set_title("17 roll-threshold stops (red), 3 stale-feedback stops (orange), 1 RC-down stop (green)\n"
             "Source monotonic duration from first policy tick to state transition; not measured unsupported survival")
savefig(fig, "attempt_durations", "Durations and stop reasons")
plt.close(fig)

attempt_directory = HERE / "attempts"
with PdfPages(HERE / "all_attempts.pdf") as pdf:
    for ep in EPISODES:
        b, e = ep["start_s"], ep["stop_s"]
        oi = ep["_oi"]
        # Preserve shutdown response, without extending stale cached requests.
        im = (IMU[:, 0] >= b - .12) & (IMU[:, 0] <= e + .22)
        st = (ST[:, 0] >= b - .12) & (ST[:, 0] <= e + .22)
        mc = (MC[:, 0] >= b - .12) & (MC[:, 0] <= e + .22)
        sm = mc & (col("motor_control", "steering_send_success") > .5)
        dm = mc & (col("motor_control", "drive_send_success") > .5)
        fig, axes = plt.subplots(6, 1, figsize=(13.2, 14), sharex=True, constrained_layout=True)
        fig.suptitle(f"Attempt {ep['episode']:02d} | {ep['duration_s']:.3f} s | {ep['reason']}\n"
                     "model_2500; wheels grounded, bike unsupported; native unsmoothed recorded samples")
        axes[0].plot(IMU[im, 0] - b, ROLL[im], color=BLUE, label="Quaternion-derived roll")
        for limit in (-30, 30):
            axes[0].axhline(limit, color=RED, ls="--", lw=.7)
        axes[0].set_ylim(-40, 40)
        extrema(axes[0], [ROLL[im]], "deg")
        axes[1].step(RAW[oi, 0] - b, RAW[oi, 1] * 1.57, where="post", color=BLUE, lw=.9,
                     alpha=.7, label="Raw action × 1.57")
        axes[1].step(ENABLED[oi, 0] - b, ENABLED[oi, 2], where="post", color=ORANGE, lw=1,
                     label="ROS request after 50 Hz slew")
        axes[1].scatter(MC[sm, 0] - b, col("motor_control", "mit_encoded_position_rad")[sm],
                        s=5, marker=".", color=RED, label="Successful MIT send: encoded target")
        axes[1].plot(ST[st, 0] - b, ST[st, 1], color=GREEN, label="Actual A3 position")
        extrema(axes[1], [RAW[oi, 1] * 1.57, ST[st, 1]], "rad")
        axes[2].plot(ST[st, 0] - b, ST[st, 3], color=GREEN, label="Actual A2 steering speed")
        for limit in (-23.9, 23.9):
            axes[2].axhline(limit, color=RED, ls="--", lw=.7)
        extrema(axes[2], [ST[st, 3]], "rad/s")
        axes[3].step(RAW[oi, 0] - b, RAW[oi, 2] * 6.28, where="post", color=BLUE, lw=1.2,
                     label="Raw action × 6.28 = ROS torque request")
        axes[3].scatter(MC[dm, 0] - b, col("motor_control", "drive_encoded_torque_nm")[dm],
                        s=6, marker=".", color=ORANGE, label="Successful C0 send: encoded torque")
        axes[3].plot(ST[st, 0] - b, ST[st, 7], color=GREEN, lw=1,
                     label="Actual A1-current × Kt torque estimate")
        extrema(axes[3], [RAW[oi, 2] * 6.28, ST[st, 7]], "Nm")
        axes[4].plot(ST[st, 0] - b, ST[st, 4], color=GREEN, label="Actual A2 rear speed")
        for limit in (-61.6, 61.6):
            axes[4].axhline(limit, color=RED, ls="--", lw=.7)
        extrema(axes[4], [ST[st, 4]], "rad/s")
        axes[5].plot(MC[mc, 0] - b, col("motor_control", "steering_pair_age_ms")[mc],
                     color=BLUE, label="Steering encoder age")
        axes[5].plot(MC[mc, 0] - b, col("motor_control", "drive_pair_age_ms")[mc],
                     color=ORANGE, label="Rear encoder age")
        axes[5].axhline(50, color=RED, ls="--", lw=.8, label="50 ms motor feedback gate")
        axes[5].set_ylim(0, max(65, np.nanmax(col("motor_control", "steering_pair_age_ms")[mc]) * 1.1))
        for ax, ylabel in zip(axes, ("Roll (deg)", "Steering (rad)", "Steer speed (rad/s)",
                                    "Rear torque (Nm)", "Rear speed (rad/s)", "Feedback age (ms)")):
            ax.axvline(0, color=RED, lw=1.1)
            ax.axvline(e - b, color=GREY, ls="--", lw=1)
            ax.axvspan(e - b, e - b + .22, color=GREY, alpha=.08)
            decorate(ax, ylabel)
        axes[-1].set(xlabel="Seconds from first successful policy tick (red); dashed grey = policy stop",
                     xlim=(-.1, e - b + .2))
        savefig(fig, f"attempt_{ep['episode']:02d}", f"Attempt {ep['episode']:02d}: {ep['reason']}", attempt_directory)
        pdf.savefig(fig)
        plt.close(fig)

# Observe representative starts and the longest roll-terminated attempt.
fig, axes = plt.subplots(11, 3, figsize=(18, 22), constrained_layout=True)
for column, ep_number in enumerate((1, 17, 19)):
    ep = EPISODES[ep_number - 1]
    oi = ep["_oi"]
    for j, name in enumerate(SCHEMAS["observation"]):
        ax = axes[j, column]
        ax.plot(OBS[oi, 0] - ep["start_s"], OBS[oi, j + 1], color=BLUE, lw=.9)
        ax.axvline(0, color=RED, lw=.8)
        extrema(ax, [OBS[oi, j + 1]])
        if column == 0:
            ax.set_ylabel(name)
        if j == 0:
            ax.set_title(f"Attempt {ep_number}: {ep['reason']}")
        if j == 10:
            ax.set_xlabel("Seconds from policy start")
fig.suptitle("All 11 recorded policy inputs | model_2500 | rad, rad/s, dimensionless gravity and trigonometric terms\n"
             "Rear position uses episode-rebased 4π branch; no linear velocity estimate")
savefig(fig, "observations", "Eleven observations for attempts 1, 17 and 19")
plt.close(fig)

fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
for column, ep_number in enumerate((13, 14, 21)):
    ep = EPISODES[ep_number - 1]
    f = next(f for f in faults if f["episode"] == ep_number)
    zero = f["event"]["t"]
    mc = abs(MC[:, 0] - zero) <= .18
    pt = abs(PT[:, 0] - zero) <= .18
    for name, color in (("steering_pair_age_ms", BLUE), ("drive_pair_age_ms", ORANGE)):
        axes[0, column].plot(MC[mc, 0] - zero, col("motor_control", name)[mc], color=color, label=name)
    axes[0, column].axhline(50, color=RED, ls="--", label="50 ms I/O freshness gate")
    axes[0, column].set_title(f"Attempt {ep_number}: {ep['reason']}")
    axes[1, column].plot(PT[pt, 0] - zero, col("policy_timing", "motor_age_ms")[pt], color=BLUE,
                        label="Policy age of motor_state publication")
    axes[1, column].axhline(60, color=RED, ls="--", label="60 ms policy freshness gate")
    for name, color in (("outputs_active", GREEN), ("fault_latched", RED), ("disable_confirmed", BLUE)):
        axes[2, column].step(MC[mc, 0] - zero, col("motor_control", name)[mc], where="post", color=color, label=name)
    for row in range(3):
        axes[row, column].axvline(0, color=RED, lw=.9)
        decorate(axes[row, column], ("Encoder pair age (ms)", "Motor publication age (ms)", "Flag")[row])
    axes[-1, column].set_xlabel("Seconds relative to motor fault event")
fig.suptitle("Three stale-feedback interruptions | clock guards, packet freshness and acknowledgements are separate\n"
             "Rows retain source telemetry; a flag change does not prove the exact physical torque-off instant")
savefig(fig, "feedback_faults", "Evidence for three feedback-related stops")
plt.close(fig)

# Distribution context, not a matched-plant or paired-policy comparison.
sim_path = ROOT / "isaaclab_train/reports/20261025_2500_dr_comparison/isaac400/motor_samples.npz"
sim = np.load(sim_path)
sim_raw, sim_obs = sim["raw_action"], sim["observation"]
flat_obs, flat_actions = sim_obs.reshape(-1, 11), sim_raw.reshape(-1, 2)
validation_indices = np.linspace(0, len(flat_obs) - 1, 2048, dtype=int)
sim_onnx_action = np.array([session.run(None, {"observation": flat_obs[i:i + 1]})[0][0]
                            for i in validation_indices])
sim_onnx_error = np.max(abs(sim_onnx_action - flat_actions[validation_indices]))
assert sim_onnx_error < 1e-5
age = np.zeros(sim_raw.shape[1], dtype=int)
early = np.zeros(sim_raw.shape[:2], dtype=bool)
for i in range(len(sim_raw)):
    early[i] = age < 100
    age += 1
    age[sim["done"][i]] = 0
sim_target = sim["held_steering_target_rad"]
sim_pos = sim["position_after_rad"]
sim_err = sim_pos[..., 0] - sim_target
comparisons = []
for j, name in enumerate(SCHEMAS["observation"]):
    for label, x in (("hardware", OBS[:, j + 1]), ("Isaac_all", sim_obs[..., j]),
                     ("Isaac_first_2s", sim_obs[..., j][early])):
        comparisons.append(dict(channel=name, dataset=label, **stats(x), rms=rms(x)))
save_csv("observation_distribution_comparison.csv", comparisons)
summary["saved_isaac_context"] = {
    "path": str(sim_path.relative_to(ROOT)),
    "interpretation": "Same model, saved DR 64-env evaluation; different trajectories/loads, not a paired transfer test.",
    "policy_samples_all": int(np.prod(sim_raw.shape[:2])), "policy_samples_first_2s": int(early.sum()),
    "steering_raw_endpoint_fraction_all": np.mean(abs(sim_raw[..., 0]) >= .95),
    "steering_raw_endpoint_fraction_first2s": np.mean(abs(sim_raw[..., 0][early]) >= .95),
    "rear_raw_endpoint_fraction_all": np.mean(abs(sim_raw[..., 1]) >= .95),
    "rear_raw_endpoint_fraction_first2s": np.mean(abs(sim_raw[..., 1][early]) >= .95),
    "steering_substep_end_tracking_rms_deg_all": np.rad2deg(rms(sim_err)),
    "steering_substep_end_tracking_rms_deg_first2s": np.rad2deg(rms(sim_err[early[:, None, :].repeat(8, axis=1)])),
    "onnx_validation_samples": len(validation_indices),
    "onnx_vs_recorded_isaac_action_max_abs_error": sim_onnx_error,
}
np.savez_compressed(HERE / "saved_simulation_actor_check.npz", flat_policy_sample_index=validation_indices,
                    onnx_action=sim_onnx_action, recorded_isaac_action=flat_actions[validation_indices])
save_json("statistics.json", summary)
fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
specs = (("Steering raw action", RAW[:, 1], sim_raw[..., 0][early], "Dimensionless"),
         ("Rear raw action", RAW[:, 2], sim_raw[..., 1][early], "Dimensionless"),
         ("Steering speed observation", OBS[:, 8], sim_obs[..., 7][early], "rad/s"),
         ("Rear speed observation", OBS[:, 9], sim_obs[..., 8][early], "rad/s"),
         ("Roll-rate observation", OBS[:, 3], sim_obs[..., 2][early], "rad/s"),
         ("Steering-position observation", OBS[:, 6], sim_obs[..., 5][early], "rad"))
for ax, (name, real, reference, units) in zip(axes.flat, specs):
    lo, hi = min(np.min(real), np.min(reference)), max(np.max(real), np.max(reference))
    bins = np.linspace(lo - 1e-5, hi + 1e-5, 46)
    ax.hist(reference.ravel(), bins=bins, density=True, histtype="step", color=BLUE, lw=1.3,
            label="Saved Isaac400 DR: first 2 s of episodes")
    ax.hist(real, bins=bins, density=True, histtype="step", color=ORANGE, lw=1.3,
            label="This deployment: all active inputs")
    ax.set(title=name, xlabel=units, ylabel="Probability density")
    decorate(ax)
fig.suptitle("Same model_2500, different closed-loop trajectories | distributions diagnose a gap, not its cause\n"
             "Isaac first 2 s per episode reduces stable-versus-startup sampling bias; samples are not paired")
savefig(fig, "simulation_context", "Policy and observation distributions versus saved Isaac DR evaluation")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)
rq = np.array([row["held_encoded_request_nm"] for row in rear_response])
fb = np.array([row["a1_current_torque_nm"] for row in rear_response])
sp = np.array([row["rear_speed_rad_s"] for row in rear_response])
image = axes[0].scatter(abs(rq), abs(fb), c=abs(sp), cmap="viridis", s=13, alpha=.65, vmin=0, vmax=61.6)
axes[0].plot([0, 6.6], [0, 6.6], color=RED, ls="--", lw=.9, label="Equal magnitude")
axes[0].set(xlabel="Held encoded C0 request magnitude (Nm)", ylabel="A1 × Kt torque magnitude (Nm)",
            title="Every active paired sample, no fitted time shift")
fig.colorbar(image, ax=axes[0], label="Rear encoder speed magnitude (rad/s)")
large = abs(rq) > 3
motoring = large & (rq * sp > 0)
braking = large & (rq * sp < 0)
axes[1].scatter(abs(sp[motoring]), abs(rq[motoring]) - abs(fb[motoring]), s=18, color=ORANGE,
                label="Request >3 Nm, torque same direction as speed")
axes[1].scatter(abs(sp[braking]), abs(rq[braking]) - abs(fb[braking]), s=18, color=BLUE,
                label="Request >3 Nm, torque opposite to speed")
axes[1].axhline(0, color=RED, ls="--", lw=.9)
axes[1].set(xlabel="Rear encoder speed magnitude (rad/s)", ylabel="Request magnitude − A1 torque magnitude (Nm)",
            title="Large requests: speed and direction context")
for ax in axes:
    decorate(ax)
fig.suptitle("Rear current-derived torque under vigorous policy commands\n"
             "No shaft sensor, bus voltage or exact A1 acquisition timestamp: this is evidence of a response gap, not an identified envelope")
savefig(fig, "rear_torque_response", "Rear torque request versus current feedback and wheel speed")
plt.close(fig)

fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
for ax, x, title, units in (
    (axes[0, 0], col("policy_timing", "cycle_interval_ms")[SUCCESS], "Active policy interval", "ms"),
    (axes[0, 1], np.array(true_mit_intervals), "Motor-reported actual MIT send interval", "ms"),
    (axes[1, 0], col("policy_timing", "imu_age_ms")[SUCCESS], "Policy-reported IMU input age", "ms"),
    (axes[1, 1], col("policy_timing", "motor_age_ms")[SUCCESS], "Policy-reported motor publication age", "ms"),
):
    ax.hist(x[np.isfinite(x)], bins=35, color=BLUE, alpha=.8)
    s = stats(x)
    ax.set(title=title, xlabel=units, ylabel="Count")
    ax.text(.98, .96, f"median {s['median']:.2f}\np95 {s['p95']:.2f}\nmax {s['max']:.2f}",
            transform=ax.transAxes, ha="right", va="top")
fig.suptitle("Active-attempt timing | intervals exclude disabled gaps; ages are host freshness estimates\n"
             "MIT intervals are observable successful sends in recorded telemetry, not an independent CAN capture")
savefig(fig, "timing", "Policy, send cadence and input age distributions")
plt.close(fig)

links = []
for name, title in FIGURES:
    links.append(f'<section><h2>{escape(title)}</h2><p><a href="{name}.png">PNG</a> · '
                 f'<a href="{name}.svg">SVG</a> · <a href="{name}.pdf">PDF</a></p>'
                 f'<a href="{name}.png"><img loading="lazy" src="{name}.png" alt="{escape(title)}"></a></section>')
(HERE / "index.html").write_text(
    '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
    '<title>20261029 deployment: model_2500</title><style>body{max-width:1500px;margin:2rem auto;'
    'padding:0 1rem;font:16px system-ui;color:#172235;background:#f7f9fb}img{max-width:100%;height:auto}'
    'section{background:white;padding:1rem;margin:2rem 0;border:1px solid #dde3e9}a{color:#2463b4}</style>'
    '<h1>October 29 deployment — Trackstand-Balance-v1 model_2500</h1>'
    '<p>Both wheels grounded, bike unsupported. Red vertical lines mark policy starts, not measured '
    'physical support release. A1 × Kt is current-derived torque, not independent shaft torque.</p>'
    '<p><a href="README.md">Analysis report</a> · <a href="episodes.csv">Attempt table</a> · '
    '<a href="statistics.json">Full statistics</a> · <a href="all_attempts.pdf">21-page attempt PDF</a></p>'
    + ''.join(links) + '</html>')
save_json("figure_manifest.json", [{"stem": name, "title": title} for name, title in FIGURES])
print(f"Wrote {len(FIGURES)} figures (PNG/SVG/PDF), 21-page attempt PDF and HTML gallery.")
