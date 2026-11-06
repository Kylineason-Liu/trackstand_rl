# Diagnostic roadmap: existing-tool review

Reviewed 2026-11-06 against Trackstand repository revision
`b0141b7dc3270980a556bf10df964119d6348726`.
This is a primary-source documentation/source review. None of these packages was
installed or exercised, and no simulation or hardware test was performed.

## Decision

**Use an existing requirements/evidence tool rather than build a new application.**
Doorstop is the closest match to this repository's YAML/Git workflow and review
tracking. StrictDoc is the stronger alternative when browser-based editing is
the priority. Both are suitable foundations; a completely custom dashboard is
not justified by this search. This recommendation is a fit assessment, not an
installation or compatibility test.

## Compared repositories

| Project | Verified capabilities | Fit and limitations |
| --- | --- | --- |
| [Doorstop](https://github.com/doorstop-dev/doorstop) | YAML or Markdown/front-matter items alongside code; parent links, custom attributes, review fingerprints, suspect-link validation, referenced files, HTML publishing, local GUI/server commands. Python 3.10+; LGPLv3. | Best match for Git-controlled evidence and change review. The document hierarchy requires deliberate modelling. Custom experimental outcomes and their scientific interpretation remain project responsibilities. |
| [StrictDoc](https://github.com/strictdoc-project/strictdoc) | Plain-text requirements, custom fields/grammars, relations, traceability views, source links, static HTML and local browser editing. Python 3.10+; Apache-2.0. | Strong interactive option. Search/filtering, change views and test-report integration are documented as experimental. No automatic transitive invalidation of physical evidence was established in this review. |
| [OpenFastTrace](https://github.com/itsallcode/openfasttrace) | Markdown specification items and source coverage tags, tracing and HTML reports; CLI/build integration. Current 4.x requires Java 17+; GPL-3.0. | Useful for requirement-to-implementation coverage. Less directly suited to a bench-experiment register with hypotheses, physical conditions and measured outcomes; adds a Java tool to this Python-oriented workflow. |

Sources: [Doorstop overview](https://doorstop.readthedocs.io/en/latest/),
[Doorstop quickstart](https://doorstop.readthedocs.io/en/latest/getting-started/quickstart.html),
[StrictDoc user guide](https://strictdoc.readthedocs.io/en/stable/stable/docs/strictdoc_01_user_guide.html),
[OpenFastTrace README](https://github.com/itsallcode/openfasttrace#readme) and
[user guide](https://github.com/itsallcode/openfasttrace/blob/main/doc/user_guide/user_guide.md).

## What change tracking actually means

Doorstop compares an item's current fingerprint with its reviewed fingerprint.
A child link becomes suspect when the linked parent's fingerprint differs from
the saved value. Extra fields can be included in review fingerprints, allowing
configuration identity and test conditions to participate in review tracking.
The `item_sha_required` extension records SHA256 values for referenced files when
an item is reviewed. See [item reference](https://doorstop.readthedocs.io/en/latest/reference/item.html),
[document configuration](https://doorstop.readthedocs.io/en/latest/reference/document.html)
and [validation](https://doorstop.readthedocs.io/en/latest/cli/validation.html).

These features do **not** establish that an old hardware result is valid after a
controller change. The inspected [item implementation](https://github.com/doorstop-dev/doorstop/blob/develop/doorstop/core/item.py)
computes file hashes during `review()` and includes stored references in the
item fingerprint. Automatic rehashing of every referenced source file during
ordinary validation was not verified. Automatic transitive experiment
invalidation was not verified either. Until these behaviours are tested, use an
explicit changed-configuration review and record affected checks as needing
revalidation. Do not clear suspect links merely to remove warnings.

StrictDoc supports Git-based diff/changelog views and structured relations, but
showing a change is not the same as invalidating dependent test results. None of
the reviewed tools can determine whether a physical setup changed unless that
change is recorded.

## Mapping to Trackstand

Keep three component categories: **bike/environment model**, **policy/training**,
and **deployment/control**. Maintain shared observation, action, timing and
lifecycle contracts. Model the evidence chain as linked items:

1. A requirement or contract states what must hold, with units and scope.
2. An investigation records the symptom and competing hypotheses.
3. A test specifies the discriminating experiment and acceptance criterion.
4. A result identifies the revision, model/checkpoint, configuration, physical
   conditions, sample count, report/plot paths, outcome and limitations.

Use custom fields for category, priority, hypothesis, criterion, environment,
configuration identity, conclusion, uncertainty and next action. Scientific
status should remain separate from the tool's document-review status:
`untested`, `supported within tested conditions`, `contradicted`, `inconclusive`,
or `needs revalidation`. A reviewed document is not a passed experiment.

Keep existing reports and raw logs in their present locations; reference them
instead of copying large data into the register. Preserve the progress ledgers
as chronological history, with links to the evolving register. No universal
numerical acceptance thresholds are introduced by this tooling choice.

## Adoption boundary

Before depending on either tool, create a small sandbox example covering one
contract, one test and one result. Verify custom-field publishing, local artifact
links, source/reference changes, suspect links and review clearing with a pinned
package version. Then seed real Trackstand checks without automatically marking
historical evidence as current. This review does not claim that such an adoption
test, a dashboard, or a populated register has already been implemented.
