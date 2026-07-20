# Lane trkL_selection_impl_20260719 — player-selection layer (P0-I fabrication fix + layers A/B/C)

## HARD RULES (all lanes)
- NO git branches, NO commits, NO pushes. The manager commits after ruling on your report.
- Read first: NORTH_STAR_ROADMAP.md (SS2.1 + SS5 queue), AGENTS.md, the RUNBOOK section for your area.
- The 4 protected eval clips are EVAL-ONLY (Burlington/Wolverine internal scoring allowed; Outdoor/Indoor labels NEVER). The 50-row owner event seed is PROTECTED EVAL, never training.
- VERIFIED=0 is binding. Preview band. Nothing you produce is a promotion. Honest reporting: misses are misses; no threshold shopping — pre-registered values are THE values.
- Sandbox limits: no network/DNS, no MPS (CPU fallback), no localhost binds, no xcodebuild. Anything needing GPU/network = a documented handoff plan in your lane dir, never faked.
- Run your focused tests AND the wide blast-radius suite (MPLBACKEND=Agg, real exit codes); attribute every failure: yours vs pre-existing (known pre-existing: sandbox socket-bind failures).
- Artifacts under YOUR lane dir only. Other lanes run dirs are READ-ONLY evidence.
- Final message = the schema-enforced structured report: objective_result PASS/FAIL/NO-ATTEMPT vs the pre-registered numbers, full_suite counts + attribution, HONEST_ISSUES, artifacts, BEST-STACK DELTA (a promotes / b pending-dormant entry / c none + why).
- CONCURRENT LANES ARE LIVE (file-disjoint — do NOT touch their files): trkL_selection_impl_20260719 (new player_selection files + list_scaffold_tools additive), static_cal_firstlock_20260717 (court_calibration.py, court_line_keypoints.py, process_video.py calibration seams), trk_rfdetr_integrate_20260717 (orchestrator.py, models/MANIFEST.json, configs/racketsport/best_stack.json, RUNBOOK tracking lines), event_head_corpus_20260719 (threed/racketsport/event_head/**, scripts/racketsport/eval_event_head.py, scripts/racketsport/train_event_head.py).

## YOUR FULL DESIGN SPEC
runs/lanes/trkL_selection_20260717/DESIGN_selection_layer.md — implement it EXACTLY (architecture, registered thresholds, implementation shape). Also read runs/lanes/trkL_selection_20260717/GHOST_DIAGNOSIS.md and the JSONs in runs/lanes/trkL_selection_20260717/diagnosis/.
Registered values (frozen-dataclass defaults, never tuned): SIGMA_COURT 0.5m, EMA half-life 2.0s, accept<=0.35 / reject>=0.42, displacement 2.5m, micro-fill cap 12 frames, weights 0.4/0.4/0.2, S>=0.5.

## FILE OWNERSHIP
YOURS (new files): threed/racketsport/player_selection.py; scripts/racketsport/select_players_from_pool.py; docs/racketsport/player_selection_report_schema.json; tests/racketsport/test_player_selection.py; runs/lanes/trkL_selection_impl_20260719/. Plus scripts/racketsport/list_scaffold_tools.py ADDITIVE dict entries only — edit it LAST, fresh-read immediately before editing.
READ-ONLY (diagnosed, never edited): threed/racketsport/player_global_association.py, threed/racketsport/player_id_repair.py, scripts/racketsport/process_video.py, threed/racketsport/orchestrator.py, models/MANIFEST.json, configs/racketsport/best_stack.json.

## MISSION
The association FABRICATES player positions (P0-I): wolverine f45-86 synthetic 42-frame cross-identity bridge, conf pinned 0.35, provenance stripped at export. Build the selection layer that supersedes gap synthesis: geometric interpolation across identity-ambiguous gaps BANNED; same-identity micro-fills <=12f AND <=2.5m AND no net-crossing, exported with per-frame interpolated:true (additive tracks.json field — provenance survives export); Layer A soft court-presence prior; Layer B 4-slot enrollment + open-set rejection + stitch veto (two independent evidence classes required for any destructive action); Layer C identity-conditioned RAW-pool recovery (real detections only). Fusion decides; no single signal decides anything irreversible.

## ACCEPTANCE (local CPU; the GPU card eval is the MANAGER'S follow-up, not yours)
1. Selection OFF => byte-identical tracks.json (no-op invariant test).
2. Wolverine-bridge fixture test from the committed diagnosis JSONs: f44<->f87 stitch REFUSED, GT1 re-bind ACCEPTED, burlington 3-frame fill untouched.
3. Enrollment determinism, open-set band semantics, provenance-survives-export tests green.
4. Focused tests EXIT 0 + wide suite with attribution.
5. Emit VM_EVAL_PLAN.md: exact commands for the manager's A100 micro-session (env-fidelity gate reproducing variant-P rows within 1e-9 -> selection arm both clips -> pull with two-sided sha256), scoring the design's pre-registered acceptance table verbatim.

## BEST-STACK DELTA
(c) none this lane — a flip proposal exists only after the manager's GPU card eval passes the pre-registered table.
