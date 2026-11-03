# Codex workflow for Trackstand

The main CLI session coordinates five specialist roles. The project configuration is
`.codex/config.toml`; reusable role instructions are in `.codex/agents/*.toml`.
`AGENTS.md` assigns the root session its coordinator behavior. The
`main_coordinator.toml` file also defines that role for an explicit coordination
handoff; it does not automatically replace the root session or start another agent.

## Roles and ownership

| Role | Work | Owned writes |
| --- | --- | --- |
| `main_coordinator` | Scope, delegation, evidence review, resource scheduling and final answer | Assigned coordination documents; reviews other agents' changes |
| `implementer` | Production code and focused validation | Explicitly assigned source/test/config files |
| `isaaclab_evaluation_analyst` | Isaac Lab evaluations and result analysis | Assigned new `isaaclab_train/reports/<run_id>/` artifacts/helpers |
| `mujoco_evaluation_analyst` | MuJoCo evaluation, survival and sim-to-sim analysis | Assigned new `mujoco_sim2sim/reports/<run_id>/` artifacts/helpers |
| `deploy_evaluation_analyst` | Deployment/bench log analysis and motor telemetry | Assigned new `deploy/experiments/reports/<run_id>/` artifacts/helpers |
| `summarizer` | Evidence synthesis; dual-format recording when requested | Assigned summary; both root progress ledgers only for a recording task |

These are reusable instructions, not six always-running processes. Each spawned
agent has its own conversation thread. They share the checkout and inherit the
session's model, reasoning effort, tools and permissions. No model or permission
override is installed by this setup. File ownership is a workflow rule, not an
additional filesystem sandbox.

## Normal CLI use

Start a fresh session after installing/changing the configuration:

```bash
codex -C /path/to/trackstand_rl
```

From the repository directory, simply run `codex`. Project configuration is loaded
for trusted projects; use the CLI's trust prompt when first opening another clone.
There is no separate service to start. Root coordination comes from `AGENTS.md`,
so no role-selection flag is required.

Describe the actual task. For example:

> Investigate the rear-wheel discrepancy using the configured evaluation analysts
> in parallel. Start with existing evidence. Give each analyst a separate report
> directory and return a combined diagnosis before making production changes.

For a complete change:

> Use the project multi-agent workflow to implement [specific change]. Have the
> relevant analysts establish the current behavior, assign the code change to
> implementer, evaluate the changed behavior, then have summarizer summarize the
> verified result. Record completed progress in both ledgers.

For one component:

> Delegate analysis of [log stem] to deploy_evaluation_analyst, then have summarizer
> produce a short evidence-based summary. No new hardware run is needed.

Use `/subagents` to inspect/switch between spawned threads; steer or stop them by
asking the main coordinator. You can ask for a particular role explicitly or let
the repository guidance select the relevant specialists. Small questions and
simple edits can remain in the main session. You do not need to create multiple
independent CLI sessions.

The project sets `agents.enabled = true` and
`agents.max_concurrent_threads_per_session = 5`, excluding the primary session.
That is a ceiling, not a request to spawn everyone. Host/runtime limits still
apply; run independent assignments in waves if fewer slots are available. The
summarizer normally runs after the specialists, not alongside unfinished work.

## Assignment and handoff

The coordinator supplies each specialist with:

- Objective and completion criteria, including the requested task/version.
- Source revision or snapshot, relevant evidence paths, checkpoint/config and
  known physical conditions. Missing facts remain explicit.
- Owned output paths and allowed edits. Use a new run directory for each analysis.
- Validation and run scope, seed/sample budget, device/CPU allocation and any
  already authorized hardware operations.
- Dependencies and what must be returned before the next step starts.

Specialists return: **result, evidence paths, exact conditions/commands, completed
sample counts, checks performed, limitations and proposed next action**. Label
proposed, implemented, tested offline, evaluated and hardware-measured status.

A typical order is independent baseline analysis, coordinator review, scoped
implementation, relevant evaluation, then summarization. Not every task needs
every stage. Analysts can create analysis helpers in their new report directories;
runtime fixes belong to the implementer. If code changes during analysis, retain
the analyzed revision/hash and recheck affected conclusions against the new code.

## Shared resources and evidence

- Keep one writer per source file and one writer for each report directory.
  Parallel source changes require disjoint ownership or explicitly assigned Git
  worktrees; worktrees still share GPU and hardware resources.
- The coordinator schedules GPU evaluations and CPU workers. Do not start an
  extra training job or contend with existing jobs merely to use all agent slots.
- Deployment analysis is offline by default. A live operation requires an
  assignment within the user's existing authorization and exclusive ownership of
  the relevant command channel. Do not ask again for authorization already given.
  Configuring agents does not itself authorize motor movement.
- Preserve raw logs, checkpoints, training artifacts and prior reports. The
  current Isaac evaluator writes beside the checkpoint: inspect its behavior and
  use a hash-verified checkpoint copy in the assigned run directory. Verify the
  current MuJoCo policy contract before selecting a checkpoint.
- Inspect current source and matching experiment metadata before applying dated
  progress conclusions. Roles deliberately do not fix motor gains, rates,
  friction coefficients or checkpoint names as universal defaults.
- Distinguish nominal and observed timing; policy inference, target updates,
  controller calculations and physics stepping; commands, fresh feedback and
  physical execution; and simulated versus hardware evidence.

## Summaries and progress records

The summarizer receives completed specialist handoffs and verifies material claims
against disk. A summary-only task changes no ledger. When progress recording is
requested, the coordinator assigns both `progress_record.md` and
`progress_record.txt` to the summarizer as their sole writer and reviews its diff.

Read both latest ledgers, deduplicate, preserve previous content, append numbered
dated Markdown entries in the appropriate part and dated TXT paragraphs in its
existing format. Do not append Trackstand entries after the Markdown footer or
rewrite historical conclusions to look current. Record supersession explicitly.
Temporary backups/checks belong outside the repository, such as a task directory
under `/tmp`.

## Installation validation

Validate all TOML files and referenced paths, then use the installed CLI's strict
configuration check from the repository root:

```bash
codex --strict-config app-server --print-startup-config
```

This checks configuration startup; it does not run a model, simulation or hardware
test. Confirm role discovery in a fresh session before delegating real work.

References: [official subagent documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents)
and [project configuration/trust](https://learn.chatgpt.com/docs/config-file/config-basic).
