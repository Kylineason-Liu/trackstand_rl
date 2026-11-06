# Doorstop setup verification

- Date: 2026-11-06; Trackstand base revision: `b0141b7dc3270980a556bf10df964119d6348726`.
- Upstream: `https://github.com/doorstop-dev/doorstop.git`, release `v3.2`, commit `b4f7f623010f131f5fde37212dde61019ffcf720`.
- Download: shallow tagged clone under repository-root `doorstop/`.
- Runtime: Python 3.10.12, local `doorstop/.venv`; installed from the downloaded source.
- Dependency versions: `doorstop_requirements.txt`; `pip check` passed.
- `./docs/deployment_validation/ds --version`: `Doorstop v3.2`.
- Prepared chain: `REQ001 <- TST001 <- RES001`, three documents, three records, all scientific statuses `untested`.
- Baseline validation and `publish all html` passed. HTML includes custom fields and existing local procedure/report links.
- The wrapper disables implicit initial review, implicit new-link stamping and validation reformatting. Upstream v3.2 defaults do not provide this read-only behaviour.
- Explicit review and clear commands were used only on the new template documents, not on existing project evidence.
- No CAN, ROS, GPU, simulator or bike test was run. The result placeholder is not a measured result.
- Independent inspector used temporary copies to confirm: baseline validation preserves record bytes; changing a reviewed custom field produces an unreviewed-change warning and a suspect child link; reviewing the parent alone retains the suspect link; explicitly clearing the child resolves it; new records are not silently reviewed by the wrapper.
- Inspector confirmed both historical evidence-link targets exist and localhost HTTP responses for the index, test document, diagnostic procedure and historical report were all 200. The ephemeral server was stopped. No browser interaction/screenshot was claimed.
- Detailed offline audit: `/tmp/doorstop-setup-audit-_l57i9up/audit.json` (temporary evidence, not a durable required dependency).
