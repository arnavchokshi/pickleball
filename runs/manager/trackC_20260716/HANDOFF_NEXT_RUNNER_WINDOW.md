# Track C handoff — next `process_video.py` runner window (written 2026-07-16, wrap-up)

The runner integration seat is FREE after this window. Take it in this order. `VERIFIED=0`
is binding; nothing below is a promotion. Standing rules: North Star §6; one serialized
owner of `scripts/racketsport/process_video.py` (rule 9); real pytest exit codes only
(no pipes — commit 3b639768c); every lane writes a dated `runs/` report.

## 1. FIRST: Track K stage-175 wiring (QUEUED, not dispatched — owner directive stopped new lanes)
- Request source: Track K `DESIGN.md §7`, "stage-175 slot"; their one-world fusion consumes
  `placement_trajectory_refined.json` as the PREFERRED player-trajectory input.
- Their code landed untracked/uncommitted at write time: `threed/racketsport/one_world_v1.py`,
  `scripts/racketsport/{build_one_world_v1,validate_one_world_v1,report_one_world_metrics}.py`,
  `docs/racketsport/one_world_v1{,_metrics,_validation}_schema.json`,
  `tests/racketsport/test_one_world_{core,clis}.py`. Confirm what actually committed before
  wiring; do not wire against a worktree.
- Wire it the same way `placewire_20260716` wired Track I (that spec is the template:
  `runs/lanes/placewire_20260716/spec.md`): opt-in, default OFF, byte-parity when off,
  typed spine16 failure semantics (expected-optional → typed degrade; programming/schema →
  loud fail), preview band, registered in the ONE canonical stage graph with
  RUN_IDENTITY dependencies/outputs, stage counts + RUNBOOK + truthful `expected_order`
  pin + authoritative-graph test updated coherently (derive counts from code, never trust a
  spec's numbers).
- Known blocker to clear first: Track K's one-world CLIs were unregistered in the scaffold
  index (fix injected as their A13, commit 71477b4ee area) — confirm
  `test_scaffold_tool_index.py` is green before you start, or you inherit a red suite.

## 2. SECOND: blur/diameter into events (NS-01.7 last structural slice)
- Design is BANKED, do not redesign from scratch: `runs/lanes/evidence17_20260716/report.json`
  (honest_issues + next). Summary: generate the immutable ball blur sidecar on the normal
  path after BALL (`threed/racketsport/ball_blur_sidecar.py` exists but is NOT invoked by the
  runner), hash it + `ball_candidates.json` into contact dependencies, join evidence by
  frame/timebase, and apply separately capped likelihood factors ONLY to existing visual
  proposals. Raw observations immutable; no raw-averaging of modalities (NS-01.7 stop rule).
- `event_fusion.BallInflectionCandidate` carries only time/xyz/confidence today — blur and
  apparent diameter never reach the contact fuser. That is the missing wire.

## 3. THIRD: whatever the North Star §5 queue says — re-read it, do not trust this file.
Queue row 1 is now reduced to: ball size/diameter into events (item 2 above) + modality
ablations (blocked on NS-02 independent labels).

## Owner-gated, do not attempt to close in code
- P0-H physical 30s/5min capture proof + NS-01.2b device trace (also carries Swift-side
  emission of the new optional sidecar fields `reference_crop` / `rolling_shutter` —
  Python side landed 1685a8878, Swift side is Track D/owner).
- NS-02 gold capture → independent labels. These gate: NS-01.4 "corrected error beats raw
  path", P0-G "audio/diameter affect independent error", and every modality ablation.
- Ball event-head pickleball fine-tune: BLOCKED. Owner spot-check FAILED 29/50 vs ≥47/50
  (ruling d74897203). The audio×track auto-labeler is REJECTED as a training-label source at
  current thresholds. The 50 owner-reviewed rows are PROTECTED EVAL SEED — never training.
  Public real-GT pretrain leg is unaffected. Label supply = scale the owner clip-review flow
  (50 rich rows in ~20 min, proven).

## Cross-track flags routed and still open at write time
- 24 court-family test failures: another track's IN-PROGRESS court-label worktree landing
  (untracked `owner_IMG_1605` partial frames incl. `frame_000060.jpg`, new
  `pbvision_11min_20260713/labels/court_keypoint_frames/`, five modified
  `court_keypoints*.json`). These are data-count pins (`assert 6 == 5`, `assert 15 == 14`,
  filename mismatches) — data and test pins must land together. Routed to the CAL/MOVE-1
  owner. NOT a calpolicy regression (cleared by error-signature + module-fence + data evidence).
- Track G storage-policy: FIXED same-day (fd5bc1da5); doc tests re-verified 14/14 EXIT 0.

## Environment lessons (cost hours; do not relearn)
- Background tasks are killed at ~1h wall-age → dispatch every codex lane
  `nohup ... & disown` so the launcher exits instantly.
- Mac sleep kills lanes AND watchers; reports usually survive → reconcile from disk on every
  resume before assuming anything.
- Session cwd can drift (a backgrounded `cd` does not persist) → ALWAYS absolute paths.
- Wide suites run ~50 min and cross-contaminate during concurrent landings → attribute every
  failure by solo rerun and/or `git archive HEAD` export before blaming a lane.
- A lane's own cross-lane blame is NOT evidence (tbwire blamed coordwire wrongly; coordwire's
  controlled experiment was right) — the manager re-verifies attribution personally.
- `runs/*` is gitignored → `git add -f` lane evidence; commit by explicit paths, never `-A`.
