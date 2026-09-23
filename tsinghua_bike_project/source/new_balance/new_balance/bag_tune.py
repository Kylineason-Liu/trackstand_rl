#!/usr/bin/env python3
"""
Tune Q and R against a recorded run instead of against the nominal model.

One job: read ``/new_balance/debug`` out of a rosbag, extract the four things
the bag knows and the model does not, and then sweep candidate weights through
a simulation that contains those four things.  Nothing is written and no node
is touched.

What a bag can and cannot tell you
----------------------------------
Q and R are not identifiable from data -- they are a statement of what you
want, so no amount of logging produces them.  What the bag does supply is the
four numbers that decide whether a given Q/R is *achievable*:

  1. actuator lag and dc gain.  Cross-correlate the commanded u against
     d(delta)/dt from the encoder.  Since the inner PD was removed the steer
     motor closes its own speed loop, and that loop's dead time is the single
     biggest thing missing from A / Bu.
  2. disturbance intensity.  One scalar w, fitted so the Lyapunov steady-state
     covariance of the *flown* gain reproduces the roll RMS in the bag.  With
     it, every candidate gain can be turned into a predicted RMS.
  3. which constraint actually binds.  Usually not the one you were tuning
     for; see stage A, which classifies how each episode ended.
  4. the realised RMS of delta, phi, phi-dot and u -- the honest Bryson
     denominators, as opposed to the ones you guessed.

Two different failures, kept apart
----------------------------------
Stage E answers "is the linear loop stable", exactly, by lifting the delay into
the state and taking a spectral radius.  Stage F answers "does it survive the
rate clamp under the measured disturbance", by simulation.  These are not the
same question and they have opposite fixes: a linearly unstable loop wants
different gains or less dead time, while a linearly stable loop that still
falls over wants a *higher* max_steer_velocity -- softening Q/R there asks for
less control authority and makes it worse.  Reading a clamp failure as a tuning
problem is the easy mistake and it sends you the wrong way.

Why stage F and G are simulations rather than a phase margin
------------------------------------------------------------
A frequency-domain phase margin is cheap but gameable: a grid search over Q
will happily find a huge Q_phidot whose loop gain dips under 1 early, scores a
large "phase margin", and then sits in the rate clamp 84 % of the time and
diverges when you actually run it.  So candidates are scored by simulating the
plant at 1 kHz with the control held at control_hz, the command delayed and
scaled by the measured actuator response, and u clipped at max_steer_velocity.

Usage
-----
    # everything, on the newest bag
    python3 bag_tune.py ../../../rosbag2_2026_08_14-06_22_34

    # diagnosis only, no sweep (fast)
    python3 bag_tune.py <bag> --no-sweep

    # score a specific candidate instead of the R multiples
    python3 bag_tune.py <bag> --Q 4 100 2 --R 10

    # the bag was flown at 50 Hz but you want to know about 200 Hz
    python3 bag_tune.py <bag> --control-hz 200

``--params-file`` supplies bike_*, min_scheduling_speed and the baseline
lqr_Q_diag / lqr_R_diag; without it the built-in defaults are used and the
header says so.

Needs the ``mcap`` package (``pip install mcap``); no ROS installation is
required, the Float64MultiArray CDR payload is decoded directly.

Runnable as a script or as ``python3 -m new_balance.bag_tune``.
"""

from __future__ import annotations

import argparse
import glob
import os
import struct

import numpy as np

try:  # installed as part of the package
    from new_balance.k_to_poles import (BikeParams, lqr_gain,
                                        params_from_yaml, state_matrices)
except ImportError:  # run as a plain script from this directory
    from k_to_poles import (BikeParams, lqr_gain, params_from_yaml,
                            state_matrices)


DEG = np.degrees(1.0)

# /new_balance/debug column indices, per the layout in new_main_balance.py
D_ROLL, D_ROLL_RATE, D_STEER, D_SPEED = 0, 1, 2, 3
D_U, D_STEER_C1, D_SAT, D_ENABLED = 4, 5, 6, 7


# ══════════════════════════════════════════════════════════════════════
# Reading the bag
# ══════════════════════════════════════════════════════════════════════


def _decode_f64ma(buf: bytes) -> np.ndarray:
    """std_msgs/Float64MultiArray out of a CDR payload.

    Layout: 4-byte encapsulation header, then the MultiArrayLayout (a
    sequence of MultiArrayDimension plus data_offset), then the double
    sequence.  CDR alignment is counted from the end of the header.
    """
    def align(off: int, n: int) -> int:
        pad = (off - 4) % n
        return off + (n - pad) % n

    off = 4
    (ndim,) = struct.unpack_from("<I", buf, off)
    off += 4
    for _ in range(ndim):
        (slen,) = struct.unpack_from("<I", buf, off)
        off = align(off + 4 + slen, 4) + 8      # label, then size + stride
    off = align(off + 4, 4)                     # data_offset
    (count,) = struct.unpack_from("<I", buf, off)
    off = align(off + 4, 8)
    return np.frombuffer(buf, dtype="<f8", count=count, offset=off).copy()


def read_debug(bag: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (log times [s], debug array [n, 14]) for one bag directory or file."""
    try:
        from mcap.reader import make_reader
    except ImportError:  # pragma: no cover - depends on the environment
        raise SystemExit("this tool needs the mcap reader:  pip install mcap")

    path = bag
    if os.path.isdir(bag):
        hits = sorted(glob.glob(os.path.join(bag, "*.mcap")))
        if not hits:
            raise SystemExit(f"no .mcap file in {bag}")
        path = hits[0]

    times, rows = [], []
    with open(path, "rb") as handle:
        for _schema, _chan, msg in make_reader(handle).iter_messages(
                topics=["/new_balance/debug"]):
            times.append(msg.log_time * 1e-9)
            rows.append(_decode_f64ma(msg.data))
    if not rows:
        raise SystemExit(f"{path} has no /new_balance/debug messages -- "
                         "was the balance node running when it was recorded?")
    width = min(len(r) for r in rows)
    return np.asarray(times), np.array([r[:width] for r in rows])


def split_episodes(times: np.ndarray, dbg: np.ndarray,
                   min_speed: float) -> tuple[float, list[np.ndarray]]:
    """Contiguous stretches with the controller armed and the bike rolling.

    A tick that hits a fault returns before publishing debug, so gaps in the
    log time are exactly the faults; anything longer than five ticks starts a
    new episode.
    """
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.02
    live = (dbg[:, D_ENABLED] > 0.5) & (dbg[:, D_SPEED] > min_speed)
    seg = np.cumsum(np.isin(np.arange(len(times)),
                            np.flatnonzero(np.diff(times) > 5 * dt) + 1))
    groups = [np.flatnonzero((seg == s) & live) for s in np.unique(seg)]
    return dt, [g for g in groups if len(g) > int(1.0 / dt)]


# ══════════════════════════════════════════════════════════════════════
# The four things the bag knows
# ══════════════════════════════════════════════════════════════════════


def actuator_response(dbg, groups, dt) -> tuple[float, float, float]:
    """Lag [s], dc gain and r^2 of d(delta)/dt against the commanded u.

    Resolution is one tick, refined by fitting a parabola to the correlation
    peak, so a 50 Hz bag cannot resolve better than a few ms either way.
    """
    lags, gains, r2s, weights = [], [], [], []
    smooth = max(3, int(round(0.02 / dt)) | 1)
    for g in groups:
        ddot = np.convolve(np.gradient(dbg[g, D_STEER], dt),
                           np.ones(smooth) / smooth, mode="same")
        cmd = dbg[g, D_U]
        corr = []
        for lag in range(int(0.12 / dt) + 1):
            a, b = (cmd[:len(cmd) - lag] if lag else cmd), ddot[lag:]
            corr.append(np.corrcoef(a, b)[0, 1] if len(a) > 10 else -1.0)
        corr = np.asarray(corr)
        peak = int(np.argmax(corr))
        sub = 0.0
        if 0 < peak < len(corr) - 1:
            lo, mid, hi = corr[peak - 1], corr[peak], corr[peak + 1]
            den = lo - 2 * mid + hi
            sub = 0.5 * (lo - hi) / den if den else 0.0
        a, b = (cmd[:len(cmd) - peak] if peak else cmd), ddot[peak:]
        lags.append((peak + sub) * dt)
        gains.append(float(np.dot(a, b) / np.dot(a, a)) if np.dot(a, a) else np.nan)
        r2s.append(corr[peak] ** 2)
        weights.append(len(g))
    w = np.asarray(weights, float)
    return (float(np.average(lags, weights=w)),
            float(np.average(gains, weights=w)),
            float(np.average(r2s, weights=w)))


def calibrate_disturbance(params, speed, min_sched, gain, roll_rms) -> float:
    """Roll-acceleration noise intensity that reproduces the measured roll RMS."""
    from scipy.linalg import solve_lyapunov

    A, Bu = state_matrices(params, speed, min_sched)
    P = solve_lyapunov(A - Bu @ np.atleast_2d(gain), -np.diag([0.0, 0.0, 1.0]))
    return float(roll_rms / np.sqrt(P[1, 1]))


# ══════════════════════════════════════════════════════════════════════
# Scoring a candidate: simulate what the bag says the loop really is
# ══════════════════════════════════════════════════════════════════════


_H = 1e-3          # plant integration step
_T_END = 30.0      # per-run horizon; divergences here show up in a few seconds


def spectral_radius(params, gain, speed, min_sched, ctrl_hz, lag, act_gain):
    """Exact linear stability of the sampled, delayed loop -- no simulation.

    Lift the ZOH-discretised plant together with the d = round(lag / T) queued
    commands into one matrix and take its spectral radius.  rho < 1 means the
    *linear* loop is stable, which is a different question from whether it
    survives the rate clamp; keeping the two apart is the whole point.  It also
    costs one eigenvalue solve instead of a 30 s simulation.
    """
    from scipy.linalg import expm

    T = 1.0 / ctrl_hz
    d = int(round(lag / T))
    A, Bu = state_matrices(params, speed, min_sched)
    M = expm(np.block([[A, Bu], [np.zeros((1, 4))]]) * T)
    Ad, Bd = M[:3, :3], M[:3, 3:] * act_gain
    K = np.atleast_2d(np.ravel(gain))
    if d == 0:
        return float(np.abs(np.linalg.eigvals(Ad - Bd @ K)).max())
    F = np.zeros((3 + d, 3 + d))
    F[:3, :3] = Ad
    F[:3, 3] = Bd.ravel()                       # oldest queued command applies
    for i in range(d - 1):                      # shift the queue along
        F[3 + i, 4 + i] = 1.0
    F[-1, :3] = -K                              # newest command enters at the back
    return float(np.abs(np.linalg.eigvals(F)).max())


def simulate(params, gain, speed, min_sched, ctrl_hz, lag, act_gain,
             noise_w, u_clip, seed=0):
    """-> (ok, p99|delta| deg, rms phi deg, rms u, clip fraction)."""
    from scipy.linalg import expm

    A, Bu = state_matrices(params, speed, min_sched)
    # A is singular (delta is a pure integrator), so discretise with the
    # augmented exponential rather than A^-1 (e^{AH} - I) Bu.
    M = expm(np.block([[A, Bu], [np.zeros((1, 4))]]) * _H)
    Ad, Bd = M[:3, :3], M[:3, 3:].ravel()

    n = int(_T_END / _H)
    every = max(1, int(round(1.0 / ctrl_hz / _H)))
    queue = [0.0] * max(1, int(round(lag / _H)))
    kick = np.random.default_rng(seed).normal(0.0, noise_w / np.sqrt(_H), n) * _H
    K = np.ravel(gain)

    x, held = np.zeros(3), 0.0
    steer, roll, cmd = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        if i % every == 0:
            held = float(np.clip(-np.dot(K, x), -u_clip, u_clip))
        queue.append(held * act_gain)
        x = Ad @ x + Bd * queue.pop(0) + np.array([0.0, 0.0, kick[i]])
        if not np.all(np.isfinite(x)) or abs(x[1]) > np.radians(90.0):
            return False, np.nan, np.nan, np.nan, np.nan
        steer[i], roll[i], cmd[i] = x[0], x[1], held
    return (True,
            float(np.percentile(np.abs(steer), 99) * DEG),
            float(np.sqrt(np.mean(roll ** 2)) * DEG),
            float(np.sqrt(np.mean(cmd ** 2))),
            float(np.mean(np.abs(cmd) > 0.999 * u_clip)))


def score(params, gain, speed, min_sched, cfg, seeds=(0, 1, 2)):
    """Mean over seeds; a single diverging seed fails the whole candidate."""
    runs = [simulate(params, gain, speed, min_sched, cfg["hz"], cfg["lag"],
                     cfg["act_gain"], cfg["w"], cfg["u_clip"], seed=s)
            for s in seeds]
    if not all(r[0] for r in runs):
        return None
    return tuple(float(np.mean([r[k] for r in runs])) for k in (1, 2, 3, 4))


# ══════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Tune LQR weights against a recorded run.")
    ap.add_argument("bag", help="rosbag2 directory (or a single .mcap file)")
    ap.add_argument("--params-file",
                    help="ROS params yaml for bike_* and the baseline lqr_*")
    ap.add_argument("--Q", nargs=3, type=float, metavar=("Q1", "Q2", "Q3"),
                    help="baseline state weights, overriding the params file")
    ap.add_argument("--R", nargs=1, type=float, metavar="R1",
                    help="baseline control weight")
    ap.add_argument("--control-hz", type=float, default=None,
                    help="tick rate to score against (default: as recorded)")
    ap.add_argument("--max-steer-velocity", type=float, default=10.0,
                    help="u clamp used in the simulation (default: 10)")
    ap.add_argument("--steer-limit-deg", type=float, default=30.0,
                    help="soft limit p99|delta| must stay under (default: 30)")
    ap.add_argument("--steer-fault-deg", type=float, default=60.0,
                    help="steer_angle_limit_deg that was in force during the "
                         "recording, for stage A only (default: the 60.0 "
                         "declare_parameter value, not the 30.0 in params.yaml)")
    ap.add_argument("--roll-limit-deg", type=float, default=30.0,
                    help="max_roll_angle_deg during the recording (stage A)")
    ap.add_argument("--live-speed", type=float, default=0.8,
                    help="speed above which a tick counts as riding")
    ap.add_argument("--speeds", nargs="*", type=float,
                    default=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4],
                    help="speeds to sweep")
    ap.add_argument("--r-multiples", nargs="*", type=float,
                    default=[1, 2, 4, 8, 16],
                    help="R multipliers to sweep (Q held fixed)")
    ap.add_argument("--u-clips", nargs="*", type=float,
                    default=[10, 15, 20, 30],
                    help="max_steer_velocity values to test in stage F")
    ap.add_argument("--no-sweep", action="store_true",
                    help="diagnosis only; skip the simulation sweep")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    params, min_sched = BikeParams(), 1.0
    Q = [1.2961771265077047, 840.8586455916102, 4.692318728430152]
    R = [1.1828982163197082]
    origin = "built-in defaults"
    if args.params_file:
        params, min_sched = params_from_yaml(args.params_file, params)
        import yaml
        with open(args.params_file, "r", encoding="utf-8") as handle:
            doc = yaml.safe_load(handle) or {}
        for key, value in doc.items():
            if (isinstance(value, dict) and "ros__parameters" in value
                    and key.endswith("balance_node")):
                node = value["ros__parameters"] or {}
                Q = node.get("lqr_Q_diag", Q)
                R = node.get("lqr_R_diag", R)
                origin = f"lqr_* in {args.params_file}"
    if args.Q and args.R:
        Q, R = args.Q, args.R
        origin = "given on the command line"

    times, dbg = read_debug(args.bag)
    dt, groups = split_episodes(times, dbg, args.live_speed)
    if not groups:
        raise SystemExit("no episode with the controller armed and v > "
                         f"{args.live_speed} m/s lasting over 1 s")
    idx = np.concatenate(groups)
    rec_hz = 1.0 / dt
    hz = args.control_hz if args.control_hz else rec_hz

    print()
    print("=" * 78)
    print(f"  Tuning Q and R against {os.path.basename(args.bag.rstrip('/\\'))}")
    print("=" * 78)
    print(f"  recorded at {rec_hz:.0f} Hz, {len(groups)} episode(s), "
          f"{len(idx) * dt:.1f} s armed and rolling")
    print(f"  baseline weights: {origin}")
    print(f"    Q = diag({[round(float(q), 4) for q in Q]})   "
          f"R = diag({[round(float(r), 4) for r in R]})")
    print()

    # ── A. how each episode ended ──────────────────────────────────────
    print("A. HOW EACH EPISODE ENDED -- which constraint actually binds")
    print(f"   latches in force when this was recorded: steer "
          f"{args.steer_fault_deg:g} deg, roll {args.roll_limit_deg:g} deg")
    print(f"   {'ep':>3} {'len_s':>6} {'|steer|max':>11} {'|roll|max':>10}   reason")
    reasons = {}
    for i, g in enumerate(groups):
        smax = float(np.abs(dbg[g, D_STEER]).max() * DEG)
        rmax = float(np.abs(dbg[g, D_ROLL]).max() * DEG)
        # whichever latch it got closest to, if it got close to either
        ratios = (smax / args.steer_fault_deg, rmax / args.roll_limit_deg)
        why = ("disabled or slowed" if max(ratios) < 0.9
               else "steer soft limit" if ratios[0] >= ratios[1]
               else "roll emergency")
        reasons[why] = reasons.get(why, 0) + 1
        print(f"   {i:3d} {len(g) * dt:6.1f} {smax:11.1f} {rmax:10.1f}   {why}")
    print("   -> " + ", ".join(f"{n}x {k}" for k, n in
                               sorted(reasons.items(), key=lambda kv: -kv[1])))
    print()

    # ── B. realised amplitudes ─────────────────────────────────────────
    print("B. REALISED AMPLITUDES -- the honest Bryson denominators")
    print(f"   {'signal':24} {'rms':>9} {'p95|.|':>9} {'max|.|':>9}")
    stats = {}
    for name, col, scale, unit in (("steer angle  delta", D_STEER, DEG, "deg"),
                                   ("roll angle   phi", D_ROLL, DEG, "deg"),
                                   ("roll rate    phidot", D_ROLL_RATE, DEG, "deg/s"),
                                   ("steer cmd    u", D_U, 1.0, "rad/s")):
        x = dbg[idx, col] * scale
        stats[name.split()[-1]] = float(np.sqrt(np.mean(x ** 2)))
        print(f"   {name:24} {np.sqrt(np.mean(x ** 2)):9.2f} "
              f"{np.percentile(np.abs(x), 95):9.2f} {np.abs(x).max():9.2f}  {unit}")
    clipped = float(np.mean(np.abs(dbg[idx, D_U]) > 0.999 * args.max_steer_velocity))
    print(f"   u at the {args.max_steer_velocity:g} rad/s clamp: {clipped * 100:.1f} % "
          f"of ticks")
    print()

    # ── C. the actuator ────────────────────────────────────────────────
    lag, act_gain, r2 = actuator_response(dbg, groups, dt)
    zoh = 0.5 / hz
    print("C. ACTUATOR -- d(delta)/dt against the commanded u")
    print(f"   lag      {lag * 1e3:6.1f} ms   (resolution is one tick, "
          f"{dt * 1e3:.0f} ms)")
    print(f"   dc gain  {act_gain:6.3f}      (1.0 would be perfect tracking)")
    print(f"   r^2      {r2:6.3f}")
    print(f"   + {zoh * 1e3:.1f} ms zero-order hold at {hz:.0f} Hz "
          f"=> {(lag + zoh) * 1e3:.1f} ms of loop dead time")
    print("   None of this is in A / Bu, which is why the nominal design is")
    print("   optimistic and why the sweep below is a simulation, not a Riccati")
    print("   solve.")
    print()

    # ── D. the disturbance ─────────────────────────────────────────────
    vbar = float(np.mean(dbg[idx, D_SPEED]))
    K_flown = lqr_gain(params, vbar, Q, R, min_sched)
    noise_w = calibrate_disturbance(params, vbar, min_sched, K_flown,
                                    float(np.sqrt(np.mean(dbg[idx, D_ROLL] ** 2))))
    print("D. DISTURBANCE -- one scalar, fitted to the measured roll RMS")
    print(f"   mean speed ridden        {vbar:.2f} m/s")
    print(f"   gain flown at that speed [{K_flown[0]:.2f}, {K_flown[1]:.2f}, "
          f"{K_flown[2]:.2f}]")
    print(f"   roll-accel noise         {noise_w:.2f} rad/s^2/sqrt(Hz)")
    print()

    # ── E. exact linear stability, before any simulation ───────────────
    print("E. LINEAR STABILITY of the flown gain, exact, delay and ZOH included")
    print("   Ask this first.  A loop can be linearly stable and still fall over")
    print("   because the rate clamp takes the authority away, and the fix for")
    print("   the two is not the same one.")
    print(f"   {'v':>5}   {'K':>26}  {'rho':>7} {'decay':>9}   verdict")
    lin_ok = []
    for speed in args.speeds:
        gain = lqr_gain(params, speed, Q, R, min_sched)
        rad = spectral_radius(params, gain, speed, min_sched, hz, lag, act_gain)
        decay = np.log(rad) * hz          # per second, comparable across rates
        if rad < 1.0:
            lin_ok.append(speed)
        print(f"   {speed:5.2f}   [{gain[0]:6.2f},{gain[1]:7.2f},{gain[2]:6.2f}]  "
              f"{rad:7.3f} {decay:+8.2f}/s   "
              f"{'stable' if rad < 1 else 'UNSTABLE'}")
    print()

    if args.no_sweep:
        return 0

    # ── F. is the rate clamp the thing that is actually killing it? ────
    cfg = {"hz": hz, "lag": lag, "act_gain": act_gain, "w": noise_w,
           "u_clip": args.max_steer_velocity}
    print(f"F. RATE CLAMP -- baseline Q/R, disturbance at the fitted level")
    print(f"   cell = seeds surviving {_T_END:.0f} s out of 3")
    print(f"   {'clamp':>8}" + "".join(f"{s:>8.2f}" for s in args.speeds))
    clamp_ok = {}
    for clip in args.u_clips:
        row = f"   {clip:6.0f}  "
        for speed in args.speeds:
            gain = lqr_gain(params, speed, Q, R, min_sched)
            runs = [simulate(params, gain, speed, min_sched, hz, lag, act_gain,
                             noise_w, clip, seed=s) for s in range(3)]
            n_ok = sum(r[0] for r in runs)
            clamp_ok[(clip, speed)] = n_ok
            row += f"{n_ok:>7}/3"
        print(row)
    print()

    # ── G. the sweep ───────────────────────────────────────────────────
    print(f"G. SWEEP -- Q fixed, R scaled; {hz:.0f} Hz tick, {lag * 1e3:.0f} ms lag, "
          f"gain {act_gain:.2f}, clamp {args.max_steer_velocity:g} rad/s")
    print(f"   cell = p99|delta| in deg over {_T_END:.0f} s x 3 seeds")
    print(f"   X = diverged      * = stays inside the {args.steer_limit_deg:g} deg "
          f"soft limit")
    print()
    print(f"   {'v \\ Rx':>8}" + "".join(f"{m:>10g}" for m in args.r_multiples))
    ok_cells = []
    for speed in args.speeds:
        row = f"   {speed:8.2f}"
        for mult in args.r_multiples:
            gain = lqr_gain(params, speed, Q, [R[0] * mult], min_sched)
            res = score(params, gain, speed, min_sched, cfg)
            if res is None:
                row += f"{'X':>10}"
                continue
            p99, rms_phi, rms_u, clip = res
            good = p99 <= args.steer_limit_deg
            row += f"{p99:>9.0f}{'*' if good else ' '}"
            if good:
                ok_cells.append((speed, mult, p99, rms_phi, rms_u, clip, gain))
        print(row)
    print()

    # ── the recommendation ─────────────────────────────────────────────
    print("   VERDICT")
    base = args.max_steer_velocity
    baseline_fails = [s for s in args.speeds
                      if clamp_ok.get((base, s), 3) < 3 and s in lin_ok]
    if baseline_fails:
        rescued = [c for c in args.u_clips if c > base
                   and all(clamp_ok.get((c, s), 0) == 3 for s in baseline_fails)]
        print(f"   The loop is LINEARLY STABLE at "
              f"{', '.join(f'{s:g}' for s in baseline_fails)} m/s and still falls")
        print(f"   over at max_steer_velocity = {base:g} rad/s.  That is the clamp "
              f"taking the")
        print("   authority away, not a tuning problem -- softening Q/R only asks "
              "for")
        print("   less control and makes the excursion worse.")
        if rescued:
            print(f"   Raising max_steer_velocity to {min(rescued):g} rad/s "
                  f"fixes every one of them.")
        print("   Do that before reading anything into the R sweep below.")
        print()
    if not ok_cells:
        print(f"   No cell keeps p99|delta| under {args.steer_limit_deg:g} deg at "
              f"this clamp.")
        if not baseline_fails:
            print("   Q/R is not the binding constraint -- cut the dead time "
                  "(raise")
            print("   control_hz, shrink the motor's own speed-loop lag) or ride "
                  "faster.")
        return 0
    ok_cells.sort(key=lambda c: (c[0], c[1]))
    slow = min(c[0] for c in ok_cells)
    print(f"   lowest speed that works: {slow:.2f} m/s")
    for speed, mult, p99, rms_phi, rms_u, clip, gain in ok_cells:
        if speed != slow:
            continue
        print(f"     R x {mult:<4g} -> Q = {[round(float(q), 4) for q in Q]}, "
              f"R = [{R[0] * mult:.4f}]")
        print(f"                K = [{gain[0]:.3f}, {gain[1]:.3f}, {gain[2]:.3f}]  "
              f"p99|delta| {p99:.0f} deg  rms phi {rms_phi:.2f} deg  "
              f"rms u {rms_u:.2f} rad/s  clip {clip * 100:.1f} %")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
