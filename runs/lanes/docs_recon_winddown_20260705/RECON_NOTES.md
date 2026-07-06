# DOCS-RECON-WINDDOWN Recon Notes

Date: 2026-07-05
Lane: DOCS-RECON-WINDDOWN
Session: 019f34a6-03cf-7ed1-9f19-7c5fe23acd06

## Scope

Edited only owned docs plus the lane report directory:

- `MASTER_PLAN.md`
- `CAPABILITIES.md`
- `RUNBOOK.md`
- `TECH_STACK.md`
- `RACKET_6DOF_GOAL.md`
- `docs/racketsport/GPU_COLD_START.md`
- one appended bullet at the end of `BUILD_CHECKLIST.md`
- `runs/lanes/docs_recon_winddown_20260705/`

No source, tests, `PIPELINE_STATUS.md`, `OWNER_CHECKIN_20260705.md`, or protected
Outdoor/Indoor labels were edited.

## Doc-Test Counts

Baseline before edits:

| Test file | Baseline result |
|---|---:|
| `tests/racketsport/test_truthful_capabilities.py` | 10 passed, 5 failed |
| `tests/racketsport/test_scaffold_tool_index.py` | 2 passed, 1 failed |
| `tests/racketsport/test_serving_manifest.py` | 3 passed |
| `tests/racketsport/test_model_manifest.py` | 5 passed |
| `tests/racketsport/test_rtmw_retirement.py` | 2 passed |

After edits:

| Test file | After result |
|---|---:|
| `tests/racketsport/test_truthful_capabilities.py` | 11 passed, 4 failed |
| `tests/racketsport/test_scaffold_tool_index.py` | 2 passed, 1 failed |
| `tests/racketsport/test_serving_manifest.py` | 3 passed |
| `tests/racketsport/test_model_manifest.py` | 5 passed |
| `tests/racketsport/test_rtmw_retirement.py` | 2 passed |

Additional verification:

- `.venv/bin/python scripts/racketsport/process_video.py --help | rg -- '--body-schedule|--fetch-body-monoliths|--wasb-checkpoint|--wasb-repo|--no-ball-arc|--ball-candidates|--no-ball-candidates|--no-sam3d-wrist-bone-lock'` returned all checked flags.
- `git diff --check -- MASTER_PLAN.md CAPABILITIES.md RUNBOOK.md TECH_STACK.md RACKET_6DOF_GOAL.md BUILD_CHECKLIST.md docs/racketsport/GPU_COLD_START.md` passed.

## Claims Fixed

| Claim family | Old/stale wording | New wording | Evidence |
|---|---|---|---|
| VERIFIED count | Root docs were dated 2026-07-03 and did not reflect July 5 work. | `VERIFIED=0` remains explicit; all recent wins are scoped and non-promotional. | `CAPABILITIES.md`; requested owner rule. |
| Speed | Root docs did not include the 2141.1s -> 1144.0s -> 702.425s -> 532.252s Wolverine path. | `MASTER_PLAN.md` now records the speed chain and separates speed from accuracy promotion. | `runs/lanes/pipeline_speed_accuracy_handoff_20260705/HANDOFF.md`; `runs/body_chunkfix_verify_20260705T204618Z/PIPELINE_SUMMARY.json`; `runs/visual1_wolverine_20260705T220517Z/PIPELINE_SUMMARY.json`. |
| BODY payload collapse | Root docs did not capture the 618.536s -> 473.037s `stage_wall_seconds` improvement or the accuracy caveat. | Docs state the speed win is real but the live report records accuracy-invariant divergence from an entangled concurrent `worldhmr.py` change. | `runs/body_chunkfix_verify_20260705T204618Z/source/body_stage_phase_timing.json`; `runs/body_payload_collapse_verify_20260705T221027Z/source/body_stage_phase_timing.json`; `runs/lanes/body_payload_collapse_livecheck_20260705/REPORT.md`. |
| Visual BODY | Root docs did not mention stance-aware smoothing/placement redistribution and reset/root-step improvement. | Docs state reset count 14 -> 2 and worst root-step p95 0.267 -> 0.100 m/frame as scoped visual evidence, not GT. | `runs/lanes/visual_polish_20260705/lane_VPA2_resets_jitter/before_speed1_wolverine_source/visual_quality.json`; `runs/lanes/visual_polish_20260705/after_visual1/visual_quality.json`; `runs/lanes/visual_polish_20260705/STATUS.md`. |
| Mesh index | Root docs described older monolith-heavy BODY outputs. | RUNBOOK/TECH_STACK/CAPABILITIES now state `body_mesh_index/` is default replay output and `smpl_motion.json`/`body_mesh.json` monoliths are opt-in via `--fetch-body-monoliths`. | `.venv/bin/python scripts/racketsport/process_video.py --help`; visual run artifacts. |
| Ball | Root docs still summarized BALL as generic unpromoted tooling. | Docs now state default 3D chain landed in `790930ed` but held-out product F1 0.6969 misses the 0.7248 zero-shot WASB-tennis bar; training concluded negative. | `git show --no-patch --oneline 790930ed`; `runs/lanes/ball_heldout_chain_run_20260704/scoring/outdoor_webcam_iynbd_1500_long_high_baseline_benchmark.json`; `runs/lanes/ball_tracking_long_run_STATUS.md`; `runs/lanes/ball_t4_train_20260704/EVIDENCE_REPORT.md`. |
| Racket | `RACKET_6DOF_GOAL.md` repeated superseded phase-1 numbers and "all acceptance bars met." | Goal doc now carries final_v3 render-only numbers: Wolverine IoU 0.2356, Burlington IoU 0.3424, 29 -> 0 undeclared teleports, five declared switch jumps remain; RKT stays SCAFFOLD. | `runs/lanes/racket_6dof_20260705/i1_fused_estimator/acceptance_record_v2.json`; `runs/lanes/racket_6dof_20260705/STATUS.md`. |
| Court auto-find | Root docs did not distinguish Wave A from mainline capability. | Docs state Wave A is worktree-patch evidence only, Outdoor no-tap median 4.4px, aggregate 213.3px vs 289.5 baseline, hard 200px bar missed. | `runs/lanes/court_autofind_20260705/handoff/cal_geo_r2_report.md`; `runs/lanes/court_autofind_20260705/handoff/court_autofind_wave_a.patch`. |
| GPU state | Root docs could be read as assuming a live remote A100 runtime. | MASTER_PLAN/CAPABILITIES/RUNBOOK/TECH_STACK now say GPU state is reset-pending and must be freshly checked before use. | Owner winddown directive in `runs/lanes/docs_recon_winddown_20260705/spec.md`. |
| RUNBOOK flags | RUNBOOK did not include current BODY scheduling, monolith fetch, WASB, and wrist-lock flags. | RUNBOOK now includes `--body-schedule`, `--fetch-body-monoliths`, `--wasb-checkpoint`, `--wasb-repo`, `--no-ball-arc`, ball candidate flags, and `--no-sam3d-wrist-bone-lock`. | `.venv/bin/python scripts/racketsport/process_video.py --help`. |
| docs/racketsport policy | `docs/racketsport/GPU_COLD_START.md` was a narrative Markdown file in the schema/manifest-only docs folder. | Removed `docs/racketsport/GPU_COLD_START.md`; this fixed one `test_truthful_capabilities.py` failure. | `tests/racketsport/test_truthful_capabilities.py`; canonical doc policy. |

## Could Not Verify / Manager Flags

- Doc-consistency tests are still not green because remaining failures require unowned or explicitly forbidden edits.
- `test_truthful_capabilities.py::test_markdown_doc_inventory_stays_small_and_explicit` still fails due extra Markdown docs in `.claude/worktrees/...` and root legacy docs outside this lane's ownership.
- `test_truthful_capabilities.py::test_storage_policy_keeps_large_tracked_artifacts_explicit` and `test_storage_policy_audit_classifies_large_worktree_artifacts` still fail because of the extra tracked large USDZ artifact under `ios/Replay/.../body_mesh_animated_budget53.usdz`.
- `test_truthful_capabilities.py::test_build_checklist_board_and_count_summary_match` still fails because the pre-existing `BUILD_CHECKLIST.md` body contains a large `Recent Handoffs` narrative section; this lane was append-only for that file.
- `test_scaffold_tool_index.py::test_real_scaffold_tool_index_matches_checked_in_schema` still fails because `scripts/racketsport/monitor_process_resources.py` lacks a direct CLI reference test; source and tests were unowned.
- I did not claim a BODY payload-collapse clean accuracy verdict. The live report records a regression by hard invariant, with speed verified and accuracy entangled with concurrent `worldhmr.py`.
- I did not live-check the A100 VM; docs now say reset-pending and require fresh runtime checks.
- I did not edit `TIER_MAP.md`; it is non-canonical and was outside the explicit ownership list.
