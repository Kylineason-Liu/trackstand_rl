# Trackstand diagnostic register with Doorstop

Doorstop **v3.2** is downloaded to `trackstand_rl/doorstop/`, pinned to upstream commit
`b4f7f623010f131f5fde37212dde61019ffcf720`:
[upstream release](https://github.com/doorstop-dev/doorstop/releases/tag/v3.2).
It runs in `doorstop/.venv/`, independently of Isaac Lab and ROS Python.
The downloaded tool, virtual environment and generated HTML are Git-ignored;
this guide, wrapper and YAML records can be version-controlled.

## Open the register

Run from `~/trackstand_rl`:

```bash
./docs/deployment_validation/ds
./docs/deployment_validation/ds publish all html
python3 -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765/docs/deployment_validation/html/index.html>.
Browse the documents and traceability table; use **Ctrl+C** to stop the server.
This browser view is for reading. Edit YAML with your editor, then republish.
Serving the repository root makes the existing report/procedure links work.

`ds` is a small launcher, not another requirements system: it selects the local
virtual environment, scans only `roadmap/`, and disables automatic review,
link stamping and reformatting during validation. Use it instead of bare
`doorstop`, which can scan the whole repository and change new records while
validating. Explicit `review` and `clear` still write their intended records.

## The prepared example

| Item | Edit this file | Meaning |
| --- | --- | --- |
| `REQ001` | `roadmap/requirements/REQ001.yml` | Requirement: attribute motor-feedback delay to its actual stage. |
| `TST001` | `roadmap/tests/TST001.yml` | Test: compare minimal/full recording load at matched CAN traffic. |
| `RES001` | `roadmap/results/RES001.yml` | Result placeholder; **no experiment run for this example**. |

The chain is `REQ001 ← TST001 ← RES001` (child links to parent).
All three have `status: untested`. Their document text was reviewed during setup;
**document review is not a passed hardware test**. Existing report links are
background evidence, not new results.

## Edit a result after an experiment

```bash
nano docs/deployment_validation/roadmap/results/RES001.yml
```

Fill these existing fields with actual values; retain units and scope:

```yaml
status: inconclusive
configuration: 'Actual Git commit, checkpoint, params and firmware identity'
environment: 'Actual support/load, power supply and CAN adapter conditions'
evidence: 'Actual run/report paths, counts and measured timing statistics'
uncertainty: 'What this run cannot distinguish'
next_action: 'One experiment that separates the remaining explanations'
```

This is an editing example, not measured data. Use `supported within tested
conditions`, `contradicted`, `inconclusive`, `untested`, or `needs revalidation`
as appropriate. Keep older runs as separate result items rather than overwriting
history. Add Markdown report links in `text`; current HTML document pages live
in `html/documents/`, so repository-root paths begin `../../../../` there.

```bash
./docs/deployment_validation/ds
./docs/deployment_validation/ds review RES001
./docs/deployment_validation/ds publish all html
```

Review only after checking the content. Doorstop checks structure and review
fingerprints, not the truth of the experiment or these custom status meanings.
Warnings do not make the default command exit nonzero; read its output (or use
`./docs/deployment_validation/ds --error-all` for stricter CI checks).
Custom fields are published and included in review fingerprints by each
`.doorstop.yml` configuration.

## Add another concern

For this untouched example, these commands create `REQ002`, `TST002`, `RES002`:

```bash
./docs/deployment_validation/ds add REQ
./docs/deployment_validation/ds add TST
./docs/deployment_validation/ds add RES
./docs/deployment_validation/ds link TST002 REQ002
./docs/deployment_validation/ds link RES002 TST002
```

Use the actual IDs printed if you already added items. Edit their YAML, for
example a steering loaded-replay requirement, procedure and pending result.
Choose `category: bike/environment`, `policy/training`, or `deployment/control`.
Fill `hypothesis`, `acceptance`, `configuration`, `environment`, `evidence`,
`uncertainty`, `next_action` and `status`; do not treat default placeholders as
completed evidence. Then review each new document item explicitly.

## When a requirement changes

Edit `REQ001`'s `text` or `acceptance`, then validate. You should see
`unreviewed changes` and `TST001: suspect link: REQ001`.
Inspect whether the test is still appropriate and whether its old results remain
applicable. Mark affected results `needs revalidation` when necessary. Only after
that review:

```bash
./docs/deployment_validation/ds review REQ001
./docs/deployment_validation/ds clear TST001
./docs/deployment_validation/ds review TST001
```

`clear` acknowledges the changed parent link; it does **not** repeat a test.
Changes to arbitrary source files or hardware are not automatically reflected
in this register. Update configuration identities and review dependent results
explicitly. Do not run `review all`/`clear all` merely to silence warnings.

Because the scan root is deliberately a subtree, write commands may print
`no working copy found at: .../roadmap`. Doorstop uses its no-op VCS adapter
there; normal Git tracking remains in the outer Trackstand repository. Nothing
is staged or committed by this setup.

## Recreate the local installation

Only if the ignored `doorstop/` directory is absent:

```bash
git clone --depth 1 --branch v3.2 https://github.com/doorstop-dev/doorstop.git doorstop
python3 -m venv doorstop/.venv
doorstop/.venv/bin/python -m pip install -c docs/deployment_validation/doorstop_requirements.txt ./doorstop
```

[Dependency pins](doorstop_requirements.txt) capture this installation.
[Setup verification](SETUP_VERIFICATION.md) distinguishes software checks from
physical tests. No existing deployment or simulation configuration was changed.
