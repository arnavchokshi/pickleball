# LANE ns015_statuspack_20260710 — NS-01.5 honest status + packaging (server-side half)

## HARD RULES
No branches, no commits, no pushes. Read NORTH_STAR_ROADMAP.md first: §2.1 rows P0-E and P0-F,
§3.2 minimum inspectable deep-result bundle, §4 task NS-01.5 (acceptance + stop rule) — that scope
is BINDING. Then AGENTS.md + RUNBOOK.md server/upload sections. .venv/bin/python; MPLBACKEND=Agg.
Honest reporting; wide suite at end (tests/racketsport AND tests/server), failures 0 or proven
pre-existing. Your sandbox has NO network and NO localhost binds — write server tests with in-process
clients (FastAPI TestClient / direct function calls), never socket binds; if an existing test needs a
socket, mark it and move on (the manager re-verifies locally). Artifacts under
runs/lanes/ns015_statuspack_20260710/.

## FILE OWNERSHIP (exclusive)
- server/** (all modules incl. worker/ and routes/)
- tests/server/**
- docs/racketsport/*.schema.json ONLY if you add/extend a job-status or bundle-policy schema
- runs/lanes/ns015_statuspack_20260710/**
FORBIDDEN (other live lanes / integration owner): scripts/racketsport/process_video.py, threed/**,
ios/**, web/**. Where the runner or app side needs changes, ship INLINE DIFF HUNKS in your report
(never a .patch file) for the integration owner to re-derive.

## OBJECTIVE (NS-01.5, verbatim gates)
"Minimum bundle policy; partial propagation through runner/worker/API/app; recursive atomic copy;
stats/coaching before manifest." Acceptance: "Missing BODY/ball/paddle/assets remains `partial`;
complete requires every advertised URL; local and SSH paths agree." Stop rule: "Exit 0 is not
sufficient." P0-E exit: status, missing capabilities, and trust bands survive worker/API/app
unchanged. P0-F exit: recursive atomic packaging; stats/coaching before manifest; every URL checked.

## EVIDENCE TO READ FIRST
- server/worker/**, server/routes/**, server/gpu_runner.py, server/pipeline_invocation.py, server/s3.py
- The landed capture->S3 route (commit 737ced8ff) and its tests under tests/server/
- How PIPELINE_SUMMARY.json + replay_viewer_manifest.json express status today (grep the runner
  outputs; do NOT edit the runner)

## MISSION
1. ONE bundle-policy module server-side: given a run's artifact tree/manifest, compute
   complete|partial|failed + an explicit missing-capabilities list per North Star §3.2. `complete`
   requires every mandatory artifact AND every advertised URL resolvable; an explicit missing reason
   keeps the bundle inspectable but PARTIAL (never complete).
2. Worker packaging: recursive ATOMIC packaging/upload (stage to temp, verify, atomically publish;
   a killed worker mid-package must leave no half-visible bundle), stats/coaching artifacts packaged
   BEFORE the manifest, and a post-publish every-URL check gating the ready flip.
3. Status propagation: worker->DB->API pass the runner's status + missing capabilities + trust bands
   through UNCHANGED (no translation, no "partial"->"complete"/"Replay ready" upgrades anywhere).
   Grep for every place the API/app-facing route derives a ready/complete flag and route it through
   the policy module.
4. Tests (in-process only): fixture bundles for complete / partial-missing-BODY / missing-URL /
   mid-package-kill / stats-after-manifest-attempt cases; each proves the honest outcome. Every new
   CLI (if any) ships its direct-CLI reference test same-lane.
5. Runner-side or app-side gaps you find: inline diff hunks + rationale in the report, NOT applied.

## ACCEPTANCE (all required for PASS)
- A1: policy module + tests prove missing BODY/ball/paddle/assets => partial end-to-end through
  worker->API response (in-process), with missing-capabilities list intact.
- A2: complete is impossible with any advertised URL absent (test flips one URL to 404/missing and
  the status degrades).
- A3: atomicity test green (simulated kill leaves no partial-visible publish).
- A4: stats/coaching-before-manifest ordering enforced by code + test, not convention.
- A5: wide suites (racketsport + server) 0 new failures; socket-bind-only failures listed for manager.
## KILL RULE
Do not weaken ANY existing fail-closed path or status. If the current worker/API architecture cannot
express a gate without runner changes, implement the server half + report the runner hunks — do not
reach into forbidden files.
## BEST-STACK DELTA (mandatory in report)
Expected (c) NO stack delta (product infra; no model/policy change). State explicitly.
## REPORT
Schema-valid report.json (lane_report.schema.json): objective_result vs A1-A5, full_suite counts,
honest_issues, artifacts, inline diff hunks for out-of-fence needs.
