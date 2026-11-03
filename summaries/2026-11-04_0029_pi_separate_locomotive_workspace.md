# Correction: the Pi's locomotive stack now has its own workspace

- **Written:** 2026-11-04 00:29 HKT. **Changed on the Pi:** 2026-11-04 00:24–00:28 HKT.
- **Corrects:** §2 of [2026-11-04_0021_locomotive_pi_deploy](2026-11-04_0021_locomotive_pi_deploy.md). That deploy switched the Pi's `~/trackstand_rl` to the `locomotive` branch. At the user's request it is now split into two independent folders. The eval results in that summary are unchanged, so this note has no charts.

## Pi layout (`smallbike`)

| Folder | Branch / commit | Stack | Build | Tests on Pi |
|---|---|---|---|---|
| `~/trackstand_rl` | `main` @ `f1b1ff1` (restored to its pre-deploy state; the local `locomotive` branch was deleted) | trackstand only | `deploy/build`, `install` removed and rebuilt with `--symlink-install`; no locomotive executables | 650 passed |
| `~/trackstand_rl_locomotive` (new) | `locomotive` @ `9773e1d` | locomotive + trackstand | own `deploy/{build,install,log}`, `--symlink-install` | 702 passed; controller loads rank 0 `ec3ccae9…` and starts disabled |

- **Independent clone:** `git clone --no-hardlinks --single-branch --branch locomotive`. The two folders share no git objects. `origin` points at `https://github.com/Kylineason-Liu/trackstand_rl.git` and fetches only `locomotive`.
- **Kept as they were:** the user's local rosbag changes in `~/trackstand_rl`, which are a tracked-file deletion under `deploy/experiments/logs/` plus an untracked copy in `deploy/`. `~/.bashrc` sources only `/opt/ros/jazzy` and `~/ros2_ws`; neither contains `trackstand_deploy`.
- **Same package name in both:** each workspace builds a package called `trackstand_deploy`, so source exactly one workspace per shell:
  - locomotive: `source ~/trackstand_rl_locomotive/deploy/install/setup.bash`
  - trackstand: `source ~/trackstand_rl/deploy/install/setup.bash`
- **Session provenance:** `locomotive.launch.py` takes the git commit for `session.json` from the folder it runs from, so locomotive sessions record `~/trackstand_rl_locomotive`'s commit. Sessions still go to `~/trackstand_logs/locomotive_<ts>/`.
- **Not done:** no live launch, CAN down, no motor writes.
