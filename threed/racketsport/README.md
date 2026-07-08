# `threed/racketsport/` — stage library index

This package holds the Python **stage implementations, gates, schemas, and
artifact builders** for the pickleball video-to-3D pipeline. It is a library of
~270 modules; this file is the map so you can find the code for a stage without
grepping 270 filenames.

- **To run the pipeline**, don't start here — use `scripts/racketsport/process_video.py`
  (see [`RUNBOOK.md`](../../RUNBOOK.md) for stage order, flags, and commands).
- **To call the HTTP API or a single stage programmatically**, see [`docs/API.md`](../../docs/API.md).
- **For the full catalog of runnable CLIs** (the `scripts/racketsport/*.py`
  build/apply/run tools), run the generator — do not hand-maintain a list here:
  ```bash
  .venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .
  ```

## How to read this index

The pipeline runs as an ordered stage graph (authoritative order:
`RUNBOOK.md` §Stage Order; the spine is `orchestrator.py`, the production entry
is `scripts/racketsport/process_video.py`). Each pipeline stage has one or a few
**entry modules** here; those entry modules pull in supporting modules from the
same pillar transitively. `process_video.py` imports ~29 of these modules
directly and reaches the rest through them, so "is this module imported by
process_video?" is **not** a reliable live/dead signal — use
`scripts/racketsport/audit_dead_code.py` and `list_scaffold_tools.py` instead.

`VERIFIED=0`: several pillars below are honest **scaffold** (built, not
gate-passed on real labels). The per-stage truth is `CAPABILITIES.md`; this
index only says where the code lives, never that a pillar is proven.

## Pillars (by pipeline stage)

| Stage (RUNBOOK order) | Entry module(s) | Pillar / supporting modules |
|---|---|---|
| ingest | `io_decode`, `sidecar`, `capture_quality`, `owner_capture_intake` | video decode + capture-sidecar contract |
| calibration | `court_calibration`, `court_calibration_metric15`, `court_proposals`, `court_corner_review`, `net_plane`, `court_zones` | `court_*` (35 modules): metric-15pt + manual/auto court geometry. Auto-find scaffold: `court_detector_v2*`, `overlapping_court_calibration`, `court_finding_technology_benchmark`. |
| tracking | `raw_pool_person_authority`, `person_reid_diagnostics` (wired); `person_mot`, `player_global_association` (trackers) | `person_*` / `player_*`: detect + BoT-SORT/ReID + raw-pool global association, `doubles_id`, `tiled_person_detector`, `person_fast`. |
| camera_motion | `camera_motion` | optional flow-based motion compensation before placement. |
| placement | `placement`, `player_grounding`, `player_court_membership` | project tracks into court/world space, `detection_scaling`. |
| rally_gating | `rally_gating` | optional loose rally-span gating (`rally_metrics`). |
| frames | `process_video_body_frames`, `body_frame_materialization` | materialize BODY frames + mesh windows. |
| ball | `ball_bounce_2d`, `ball_manual_court_inout`, `wasb_adapter`, `ball_stage_runner` | `ball_*` (51 modules): WASB/TrackNet detect (`ball_tracknet`, `tracknet_adapter`, `wasb_adapter`), fusion (`ball_model_fusion`), filters (`ball_temporal_filter`, `ball_identity_filter`, `ball_court_filter`), in/out + line calls (`ball_inout_gate`, `ball_line_calls`). |
| ball_arc | `ball_arc_chain`, `ball_inflections`, `ball_physics3d` | 3D chain: `ball_arc_solver` (5.7k lines — public entry `solve_ball_arc_track`), `ball_ransac_arc_gate`, `ball_flight_sanity`, `ball_bounce_candidates`. |
| events | `event_fusion`, `audio_onsets_v2` | fuse ball/audio/wrist cues → `contact_windows`; `wrist_velocity_peaks`, `contact_window_candidates`. |
| ball_fill | `ball_physics_fill` | render-honest fill from accepted arc/contact evidence. |
| body | `body_grounding_refine`, `worldhmr`, `process_video_body_frames` | SAM-3D-Body world path: `worldhmr`, `hmr_deep`, `body_compute`, `body_array_native`, `skeleton3d`, `skeleton_lift_2d`, `pose_temporal`, `sat_hmr_body_fallback`, `sam3d_body_input_prep`. (RTMW retired — SAM-3D-Body only.) |
| placement_refine / grounding_refine | `body_grounding_refine`, `foot_pin` | `foot_*` (`foot_lock_solver`, `footlock`, `foot_contact`), `body_grounding_quality`, `physics_refine`. |
| world | `virtual_world`, `placement` | `virtual_world` (+ `mesh_export`, `replay_*`), `net_plane`. |
| confidence | `confidence_gate`, `virtual_world` | `confidence`, `trust_band`, `drift_guard`. |
| manifest | `replay_viewer_manifest`, `schemas` | `serving_manifest`, `model_manifest`. |
| verify | — (script `verify_process_video_viewer.py`) | headless web-viewer honesty check. |

## Non-linear pillars (not in the main stage path)

| Pillar | Modules | Status (see CAPABILITIES.md) |
|---|---|---|
| Racket / paddle | `racket_candidates`, `racket6dof`, `paddle_pose_fused`, `racket_physics_estimate`, `racket_stage_runner` | SCAFFOLD (render-only 6-DOF paddle). |
| Shot / drill | `shot_classifier`, `shot_rules`, `shot_taxonomy`, `shot_dataset_builder`, `shot_transfer_baseline` | SCAFFOLD. |
| Report / insight | `report_model`, `insight_rules`, `biomech`, `movement_metrics`, `llm_copy`, `habit_model` | SCAFFOLD (metrics/copy). |
| Eval gates & metrics | `eval/` (18 modules: `*_eval.py`, `metrics.py`, `body_gate_report.py`, `summary.py`) | the promotion gates — see `CAPABILITIES.md` + `docs/API.md` (tests). |
| External-GT harness | `external_gt_*` (ASPset-510 body GT compare) | INTERNAL-VAL tooling. |
| CVAT / dataset | `cvat_*`, `*_dataset.py`, `roboflow_corpus`, `online_harvest_ingest` | label ingest + dataset builders. |

## Pipeline infrastructure (not a pillar)

- `orchestrator.py` — the fail-closed pipeline spine (`run_pipeline(stage=...)`). See `docs/API.md`.
- `pipeline_cli.py` — legacy public-contract CLI (schema/fixture plumbing; run as `-m threed.racketsport.pipeline_cli`).
- `pipeline_contracts.py` — the `PIPELINE_STAGE_CONTRACTS` stage graph + artifact/schema map.
- `schemas/` — dataclasses + JSON schema for every stage artifact (mirrors `docs/racketsport/*_schema.json`).
- `eval_guard.py` / `append_lock.py` — enforce the protected-eval-clip and append-only-ledger rules (CLAUDE.md hard rules).
- `model_manifest.py` / `serving_manifest.py` — weight/serving registries.

## Conventions

- `*_gate.py` modules return fail-closed pass/fail decisions keyed to reference
  ranges; they never silently pass. Tests: `tests/racketsport/test_*_gate.py`.
- `*_runner.py` / `*_stage_runner.py` wrap a stage for the orchestrator.
- Underscore-private helpers are internal; a few are shared cross-module by
  design (e.g. `body_array_native` reuses private helpers from `mesh_export`) —
  do not "privatize" one side without checking the other.
- Tests for a module `foo.py` live in `tests/racketsport/test_foo*.py`.
