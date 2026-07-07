# LANE w4_burlmesh_fix_20260707 — micro fix: honest mesh-availability warning (metadata/copy only)

## OBJECTIVE
The `missing_mesh_vertices` warning is misleading telemetry: it fires whenever the EMBEDDED world
player representation lacks vertices, even when mesh vertices exist in the `body_mesh_index/`
sidecar (all 4 eval clips in the freshest wave-3 set carry the warning while having healthy mesh
indexes, e.g. Burlington mesh_frame_count=166). Make the warning honest: distinguish
"embedded vertices absent but mesh index present" from TRUE mesh absence. Telemetry/copy change
ONLY — no gate logic, no scheduling, no render-path changes.

## EVIDENCE TO READ FIRST (the ruled diagnosis — do not re-derive)
`runs/lanes/w4_burlmesh_diag_20260707/REPORT.md` — emitting site `threed/racketsport/virtual_world.py`
`_warnings()`; viewer rendering `web/replay/src/viewerData.ts` `friendlyWorldWarning()`; the
index-presence signal the diagnosis identified (mesh-index manifest/sidecar). Re-grep both sites at
HEAD before editing.

## DESIGN (pinned)
1. `virtual_world.py:_warnings()`: when embedded vertices are absent BUT the run's body mesh index
   sidecar/manifest is present and non-empty → emit `missing_embedded_mesh_vertices` (new, softer
   code) instead of `missing_mesh_vertices`. True absence (no index either) keeps the existing
   strong `missing_mesh_vertices` warning unchanged. CHECK how `_warnings()` can see the
   index-presence signal — pass it in from the caller that already knows artifact paths if needed;
   keep the seam minimal and fail-closed (uncertain presence → keep the strong warning).
2. `web/replay/src/viewerData.ts:friendlyWorldWarning()`: friendly copy for the new code (e.g.
   "mesh rendered from mesh index; embedded world is skeleton-only"); existing copy for true absence.
3. Repo-wide grep for the literal `missing_mesh_vertices` (code, tests, QA/report tooling, docs):
   list every consumer in the report; update tests that assert on it where behavior legitimately
   changed; do NOT weaken any QA/gate consumer — if a gate consumes this warning, STOP and report
   (the diagnosis says none does; verify).

## ACCEPTANCE
- Python test: `_warnings()` emits `missing_embedded_mesh_vertices` when index present + embedded
  absent; emits `missing_mesh_vertices` when both absent; emits neither when embedded present.
- Viewer test: `friendlyWorldWarning()` maps both codes to distinct copy (match the repo's existing
  web test framework — CHECK how web/replay tests run, e.g. vitest via npm script; if the sandbox
  cannot execute them (no network for npm install etc.), write the tests, run whatever is locally
  runnable, and report the exact command + the environmental limitation with proof).
- Full blast radius: the test files covering virtual_world + viewer data (grep and list them), plus
  every consumer test your grep found. All green or proven pre-existing.

## OWNED FILES (anti-collision fence)
`threed/racketsport/virtual_world.py`, `web/replay/src/viewerData.ts`, their test files, your lane
dir. DO NOT TOUCH: `process_video.py`, `orchestrator.py` (fenced), `ball_*`/trainer files (other
live lanes), `remote_body_dispatch.py` (live lane), `ios/**`, `runs/manager/**`, eval labels.

## DISCIPLINE
`.venv/bin/python`; no git branch/commit/push; no network; no new root-level .md; prove
pre-existing failures at HEAD; sandbox-environmental failures classified with proof.

## STRUCTURED REPORT
Acceptance table; CHANGES file:line; full_suite with named failures; honest_issues; next.
