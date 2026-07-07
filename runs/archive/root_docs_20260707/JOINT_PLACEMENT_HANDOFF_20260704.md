# Joint Quality & Court Placement — Handoff (2026-07-04, ~08:50Z)

**Scope: joints, skeleton/mesh consistency, person tracking as it affects joints, court placement/grounding/stance-aware placement, and replay/world outputs only as evidence of those.**
**Explicitly excluded: all ball work (tracking/arcs/bounce anchors/regen/scoring), paddle/racket work, commits/branches.** Ball lanes under `runs/lanes/ball_*` belong to a separate concurrent session — do not touch.

Nothing in this document is VERIFIED in the repo's capability sense; all numbers are measured evidence at the cited paths.

---

## 1. WHAT WAS DONE

All lane evidence lives in `runs/lanes/<name>/{spec.md, report.json, log.txt, artifacts}`.

**Research wave (read-only diagnosis, all landed):**
- **R1 `r1_refine_audit`** — per-stage audit of the skeleton refine chain on 399 extreme + 300 normal player-frames (near/far split, harness sanity-checked against the known 18.9mm slide number). Findings: raw SAM-3D joints already agree with the mesh at 6.27/26.7mm median/p90; one-euro smoothing adds +84.6/+710mm damage on extreme frames (+1211mm far players); final core jitter clamp +356mm p90; bone-length (+6.8mm, zeroes bone CV), wrist lock (+6.1mm), contact splice (~0) are harmless. Artifact: `runs/lanes/r1_refine_audit/per_stage_audit.md`.
- **R2 `r2_mesh_skeleton`** + A100 validation (`runs/lanes/r2_mesh_skeleton/gpu_validation/`) — the GPU's emitted `pred_keypoints_3d` **is** the MHR70 mesh-regressed output (recomputed vs emitted max 3.59e-7 m over 80 records) on every BODY frame; vertices-only regression is NOT exact (49/70 rows use native joint columns). Consequence: skeleton–mesh consistency requires no GPU change; the damage was all local post-processing.
- **R3 `r3_unified_grounding`** — implementation blueprint `runs/lanes/r3_unified_grounding/design.md` (69 validated file:line citations): native pre-BODY placement→worldhmr anchor path already exists; fix = harden it, delete post-hoc XY translators, stance-aware root smoothing, transl_world consistency.
- **R4 `r4_outdoor_calibration`** — decision memo + overlays. RESOLVED by owner: outdoor corners already tapped (`eval_clips/ball/outdoor_webcam_iynbd_1500_long_high_baseline/labels/court_corners.json` + `court_calibration_metric15pt.json`); court-GEOMETRY files from that labels dir are authorized for calibration/render; gameplay labels remain scoring-only.

**Implementation (all uncommitted in the working tree):**
- **I1 `i1_grounding_unification`** (+ fix round `report_fix1.json`) — stance-aware root smoother in `worldhmr.py` (locks to placement anchors in stance, low-lag prediction in transitions) replacing uniform EMA(0.65)+3m/s clamp; post-hoc translators dead/provenance-gated (`placement_refine` same-pass rewrite disabled, `grounding_refine` XY gated, staged re-anchors never ported); `foot_pin.py` demoted to ≤2cm micro-correction, no between-stance interpolation, never mutates track_world_xy, fails loudly; `virtual_world.py` transl_world source-selection fix (stale up to 2.96m → measured max 0.015m; after fix round, machine-precision on legacy repair path); NaN inputs fail safe; honest clamp metrics (known-answer tested); legacy pre-R3 run dirs route to explicit reuse. Local gates measured: slide p95 18.9mm, kitchen bias 0.0094m, wrists 0.0%, placement harness green.
- **V1 `v1_verify_grounding`** — independent adversarial verifier on I1: 9 attacks, 8 findings (1 blocker + 4 major), all fixed in the I1 fix round or attributed (the 485mm "blocker" was a stored-legacy-data artifact; legacy route measures 19.4mm slide).
- **I2a `i2a_pose_refine_cuts`** — `pose_temporal.py`: hard 30mm displacement cap on all SAM-3D smoothing/clamping (structural, default); one-euro capped with state reset; core guard now a capped true-anomaly clamp with honest engagement counters. Measured on stored Wolverine (R1's harness): extreme-frame skeleton-vs-mesh p90 **1105mm → 25.86mm** (raw ceiling is 26.7mm), median 12.2mm, far-player damage 1211→~0mm, wrists 0.0%, bone CV 0.0%, normal jitter 1.05× (bound 1.15×). All 9 targets passed.
- **G2a `g2a_gpu_runner_opt`** (+ round 2) + benches `runs/lanes/g2b_a100_bench/`, `runs/lanes/g2b_a100_bench_rev2/` — `run_sam3dbody_batch.py`: production-graph warmup with callable-identity assertion, bounded prefetch pipeline, writer-thread binary (pickle) chunk streaming with `--chunk-format`/`--no-monolithic-output` + byte-identical offline converter. A100 rev2 bench: **316.1s → 84.8s total**, TTFR 66.5s (was 316s), GPU sum 16.75s, zero idle gaps, zero bucket inflation, peak 5330MiB; outputs identical to ~1.2e-6 m (fp32 nondeterminism floor; the 1e-6 bar was over-tight — manager-ruled immaterial).
- **FIX2 `fix2_stance_wiring`** — placement natively detects stance (low-speed dwell, 0.45 m/s / 0.15s, keypoint threshold 0.3 m/s) with provenance counts; `remote_body_dispatch.py` `BODY_INPUT_ARTIFACTS` now ships `placement.json` + `foot_contact_phases.json` (when present) + `court_calibration.json`; generated remote runner calls `run_pipeline(..., reuse_existing_stage_artifacts=True)` so remote BODY consumes shipped tracks/calibration instead of re-deriving. Dispatch-layout test asserts `stance_aware_grounding=true` + `grounding_anchor_source="placement_track_world_xy"`.

**E2E attempts (`runs/lanes/e2e_wolverine_fresh*/`, `runs/lanes/e2e_wolverine_attempt3/`):**
- **Attempt 1** (`runs/i1_grounding_unification_a100_wolverine_20260704T0702Z`) — aborted in 4.6s: local `.venv` had been gutted (no torch/ultralytics/playwright). Led to env restore.
- **Attempt 2** (`runs/i1_grounding_unification_a100_wolverine_20260704T0718Z`) — full pipeline ran (1139.6s); viewer verified; far-player divergence 1.86m→0.3635m (5×) even with the new path dormant; exposed: dispatch didn't ship placement.json, placement emitted 0 stance phases, torchreid missing (ReID silently skipped → kitchen bias stuck 0.4537m, slide fail), remote re-derived calibration (1.63m internal offset absorbed by world builder to ≤0.0197m). All wired/fixed by FIX2 + torchreid install.
- **Attempt 3** (`runs/i1_grounding_unification_a100_wolverine_attempt3_20260704T083058Z`) — see CURRENT FACTS. Upstream all worked; BODY starved by GPU lock contention from the concurrent ball-regen lane; gate table therefore does not exist yet.

**Environment work:** local `.venv` (Python 3.14.6) restored: torch 2.12.1 (MPS true), torchvision 0.27.1, ultralytics 8.4.87, playwright 1.61.0 + chromium, torchreid 0.2.5 (+gdown, tensorboard), `lap` auto-installed. VM synced with the 7 fixed .py files (see §5).

---

## 2. CURRENT FACTS (attempt 3, measured)

- **Run:** `runs/i1_grounding_unification_a100_wolverine_attempt3_20260704T083058Z/wolverine_mixed_0200_mid_steep_corner/` — `PIPELINE_SUMMARY.json` status **"partial"**, wall 481.3s.
- **Stages:** ingest/calibration/tracking/placement/rally_gating/frames/events/world/confidence_gate/manifest/verify **ran**; ball + ball_fill **blocked** (expected, no ball flags); placement_refine **skipped** (by R3 design); grounding_refine **skipped** (needs `foot_contact_phases.json`, which needs BODY); **body DEGRADED** — verbatim: *"remote BODY dispatch … did not acquire scripts/gpu-eval-run.sh within 60s (another job likely holds scripts/gpu-train-lock.sh's exclusive lock)"*; *"remote SAM-3D BODY did not complete; no fallback pose skeleton was generated."* Zero remote compute occurred; no VM scratch was created.
- **Native stance phases: EMITTED** — `placement.json` provenance: `native_stance_phase_count: 13`, method `low_speed_dwell`; per-player stance coverage: p1 12.8% (37/290), p2 60.3% (181/300), p3 4.7% (14/300), p4 14.6% (42/287).
- **ReID association APPLIED** — tracking note: "refined tracks.json with raw-pool global association (profile=wolverine_internal_val_trk12_cfg151_minconf03_margin1_appw05_backfill, osnet ReID + motion, device=mps, batch=64)"; `global_association/reid_embeddings.json` present; zero torchreid errors in stdout.
- **Gate artifacts: MISSING** — no `skeleton3d.json`, no `body_grounding_quality.json` (both require BODY). Every body-dependent gate (far-player divergence, transition lag, clamp engagement, slide, kitchen bias, transl_world consistency, remote-vs-local divergence) is **N/A this run**.
- **Trust bands:** court = badge `preview`, gate `court_calibration_pck5px_gate` status `metric15_unverified` (metric-15pt reviewed calibration, grade=warn, metric_confidence=low). track = badge `low_confidence`, gate `trk_idf1_gate` status `do_not_promote`, reason "IDF1=unknown" (no GT scored for this run — possibly structural per-run state, not a regression; undiagnosed).
- **Viewer:** verify ok=true, 0 page errors, 0 assertion errors; screenshot `…attempt3_20260704T083058Z/wolverine_mixed_0200_mid_steep_corner/screenshots/process_video_verify.png`; manifest URL (dev server must be running, see §5):
  `http://localhost:5173/?manifest=/@fs/Users/arnavchokshi/Desktop/pickleball/runs/i1_grounding_unification_a100_wolverine_attempt3_20260704T083058Z/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json`
- Best previous full-pipeline world for visual comparison (pre-fix, accuracy caveats): attempt 2's manifest at `…20260704T0718Z/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json`; best staged world remains `runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z/`.

---

## 3. WHAT IS STILL BROKEN OR NOT PROVEN (blunt)

1. **Attempt 3 is NOT a pass.** The single thing it was meant to prove — BODY emitting `stance_aware_grounding=true` grounded on `placement_track_world_xy`, and the resulting gate table — never executed. BODY got zero GPU seconds. Placement-side wiring is demonstrated; the BODY-side loop is not closed. No engagement claim can be made.
2. **Do not launch Burlington/IMG_1605/Outdoor until Wolverine produces the gate table.** Multi-video runs would burn hours re-discovering whatever the Wolverine rerun exposes, and the failure attribution would be confounded across clips.
3. **The GPU is contended by design right now:** the concurrent ball-regen lane holds `scripts/gpu-train-lock.sh` exclusively and launches back-to-back jobs (observed 08:17Z and again 08:40Z; TrackNetV3 ~20+min/clip, 4-clip × multi-variant queue ⇒ hours). `remote_body_dispatch.py` waits only `lock_wait_timeout_s=60` then degrades. Options (next agent's first decision): wait for the regen queue to drain; or coordinate lock priority with the owner; or pass a much longer lock-wait through `process_video.py`. Do NOT kill the regen — it belongs to the other session.
4. **Missing artifacts are absence-of-BODY, not a separate writer bug** (per attempt-2, where BODY ran and both `skeleton3d.json` + `body_grounding_quality.json` were written) — but confirm on the rerun; if BODY runs and they're still missing, that IS a new bug.
5. **Stance coverage floor:** player 3 has only 4.7% stance coverage (14 frames) — below the 15% plausibility floor FIX2 used on its bench input. If p3 is the far player, sparse stance anchors may weaken the far-player divergence gate. Check per-player stance quality on the rerun before blaming the smoother.
6. **VM sync is fragile:** the VM runs UNCOMMITTED local code copied into `/home/arnavchokshi/pickleball_train_main`'s working tree (7 files, md5s in §5). `git pull`/`checkout`/`reset` on the VM **silently reverts the fixes**. Backups: `/home/arnavchokshi/r3_sync_backup/`; marker: `R3_SYNC_README.txt`. Also note FIX2 later changed LOCAL `remote_body_dispatch.py` + `placement.py` — both run locally, so no re-sync needed for them; if any of the 7 synced files change locally again, re-sync + re-md5.
7. **Environment risks:** dual OpenCV in local venv (`opencv-python` 5.0.0 + `opencv-python-headless` 4.13) — imports fine today, known ABI-conflict pattern; some `net_anchor_court` HoughLinesP test failures appeared post-upgrade. Full pytest ABORTS under sandbox with matplotlib's macosx backend — always run with `MPLBACKEND=Agg`. Local disk was at 98% (10GiB) when the venv got gutted; ~33GiB free as of 01:15 local — watch it. Suite baseline in this dirty tree: ~2518 passed / ~15 failed, failures attributed to sandbox-networking / opencv-upgrade / other-session dirty docs / local `data/testclips` state — none in the joint/placement files.
8. **Dirty-worktree boundaries (respect strictly):** OTHER SESSION owns and/or actively edits: `MASTER_PLAN.md`, `OVERLAPPING_COURT_CALIBRATION_GOAL.md`, `scripts/racketsport/list_scaffold_tools.py`, `threed/racketsport/{court_detector_v2_model,court_finding_technology_benchmark,overlapping_court_calibration,tracknet_adapter}.py` + their tests, `web/replay/src/*` (ACTIVE mid-session edits observed), `third_party/{SAT-HMR,TOTNet,TrackNetV4,WASB-SBDT,blurball}`, all `runs/lanes/ball_*`, `scripts/racketsport/render_overlapping_court_top_residual_refit_review_packet.py`. A codex process belonging to that session may be running at any time. `remote_body_dispatch.py` carries BOTH sessions' edits — surgical additive changes only. NOTHING from tonight is committed (owner deferred commits; a commit+push attempt was permission-blocked). `OWNER_CHECKIN_20260703.md` is the owner-facing log.
9. **Track trust "IDF1=unknown / do_not_promote"** on attempt 3 — likely structural (no GT scoring in-run) but undiagnosed; it does not block the joint gates but confirm it isn't masking a real tracking regression (attempt 3 tracking took 342.8s vs 179s in attempt 2 — also unexplained; possibly ReID batch on MPS).

---

## 4. NEXT AGENT TASK LIST (in order)

1. **Resolve GPU access for BODY** (diagnosis is done: lock contention, §3.3). Check the ball-regen queue state on the VM; pick: wait / coordinate with owner / raise lock wait (e.g. run during a regen gap; there is no `--lock-wait-timeout-s` CLI today — if you add one, that's an authorized joint-wiring change; alternatively rerun when `ps aux | grep tracknet` on the VM is empty).
2. **Rerun Wolverine ONLY** (exact command §5). Preflight first (§5): torchreid import, VM md5 spot-check, GPU lock free.
3. **Immediately verify the loop closed:** `skeleton3d.json` + `body_grounding_quality.json` exist; provenance shows `stance_aware_grounding: true` and `grounding_anchor_source: "placement_track_world_xy"`. If BODY ran but artifacts/flags are absent → new wiring bug: fix ONLY in joint/skeleton/placement/BODY files (worldhmr/orchestrator/pose_temporal/foot_pin/body_grounding_refine/virtual_world/placement/remote_body_dispatch + their tests), nothing else.
4. **Extract the gate table** (bars): far-player body-vs-floor divergence p90 <0.20m (baselines: 1.85951m staged, 0.3635m attempt-2-dormant); transition anchor lag p95 ≤0.10m / median ≤0.05m; root clamp engagement ≤5% overall / ≤10% per-player; foot slide p95 ≤20mm pooled + per-player (harness in §5); pipeline `foot_slide_gate` passed=true; kitchen-line bias ≤0.02m; transl_world consistency ≤0.02m at displayed-skeleton frames; remote-vs-local grounding divergence (should collapse from 1.63m with shipped calibration); per-player stance coverage. Investigate p3's 4.7% stance coverage while you're there.
5. **Only after Wolverine's table is green (or consciously ruled):** run Burlington → IMG_1605 (owner capture) → Outdoor (calibration = the reviewed labels files, geometry only, render-only guard noted in OWNER_CHECKIN), same gate extraction each.
6. **Final visual QA on all four:** headless verifier screenshots + human-style review of extreme poses/far players/stance transitions in the viewer; compare against `runs/manager_stage_sam3d_wolverine_v5_1_20260703T2012Z` (v5.1) for Wolverine.

---

## 5. EXACT COMMANDS / PATHS

- **Repo root:** `/Users/arnavchokshi/Desktop/pickleball` (stay on main, NO commits/branches; preserve all dirty files).
- **Wolverine E2E rerun:**
  ```bash
  cd /Users/arnavchokshi/Desktop/pickleball
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  .venv/bin/python scripts/racketsport/process_video.py \
    --video eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/source.mp4 \
    --clip wolverine_mixed_0200_mid_steep_corner \
    --court-calibration eval_clips/ball/wolverine_mixed_0200_mid_steep_corner/labels/court_calibration_metric15pt.json \
    --out runs/i1_grounding_unification_a100_wolverine_attempt4_$TS \
    --force --rally-gating \
    --global-association-profile wolverine_internal_val_trk12_cfg151_minconf03_margin1_appw05_backfill \
    --verify-viewer --json
  ```
- **Preflight:** `.venv/bin/python -c "import torch,ultralytics,torchreid; print('ok')"`; GPU lock/state: `ssh -i ~/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=configs/ssh/a100_known_hosts arnavchokshi@34.126.67.233 'nvidia-smi; ps aux | grep -E "[t]racknet|[g]pu-train-lock"'`.
- **VM synced-file md5s** (must match local; restore = `git -C /home/arnavchokshi/pickleball_train_main checkout -- <files>` but that REVERTS the fixes — only do it to hand back a clean VM): orchestrator.py `69deb0eb8510465d76821505f7dc2e86`, worldhmr.py `3d74ecefb4d3170634c75639ad9c6a1f`, pose_temporal.py `54ccbb165a08cafcb5f723ac59f9bcac`, foot_pin.py `b73ef7d283a984fa47c96450a8fc4efa`, body_grounding_refine.py `98fe9bf2e45e4b64c2811b8e6b276967`, virtual_world.py `98e3c7c65216911257707819300000cd`, run_sam3dbody_batch.py `c9eee8305319cbe5221fb64c20936a67`. Backups `/home/arnavchokshi/r3_sync_backup/`; marker `R3_SYNC_README.txt`.
- **Gate/measurement harnesses:** foot slide + kitchen bias: `runs/lanes/e2e_wolverine_attempt3/diagnose_placement_acceptance_e2e.py` (adapt run-dir path; original pattern `runs/placement_stage_20260703T1938Z/diagnose_placement_acceptance.py`); divergence/transl computations: `runs/lanes/e2e_wolverine_fresh2/grounding_facts_e2e.json` shows the method; refine-chain audit harness: `runs/lanes/i2a_pose_refine_cuts/audit_refine_chain.py`.
- **Key artifacts to read:** `<clip_dir>/PIPELINE_SUMMARY.json`, `placement.json` (provenance→stance), `skeleton3d.json` (provenance→engagement), `body_grounding_quality.json` (slide gate/metrics), `trust_bands.json`, `confidence_gated_world.json`, `screenshots/`, `replay_viewer_manifest.json`. Stdout logs from attempts: `runs/lanes/e2e_wolverine_fresh2/stdout.log` (attempt 2), attempt-3 facts in `runs/lanes/e2e_wolverine_attempt3/`.
- **Viewer dev server** (currently RUNNING; restart if down): `cd web/replay && npm run dev -- --host 127.0.0.1 --port 5173 --strictPort`, then `http://localhost:5173/?manifest=/@fs<abs clip_dir>/replay_viewer_manifest.json`.
- **Test suite (dirty tree):** `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q --deselect tests/racketsport/test_court_finding_technology_benchmark.py --deselect tests/racketsport/test_overlapping_court_calibration.py --deselect tests/racketsport/test_overlapping_court_calibration_eval.py` (deselects = other session's files; expect ~15 attributed environmental failures, §3.7).
- **Docs/logs:** owner log `OWNER_CHECKIN_20260703.md`; operating model `FABLE_OPERATING_MANUAL.md`; prior full handoff `JOINT_DETECTION_AND_PLACEMENT_HANDOFF.md` (its §6/§9 are now largely superseded by tonight's lanes — trust this document + lane reports first).

## 6. STOP STATE (as of 2026-07-04 ~08:52Z)

- All background agents this session started are **finished** (last one, the attempt-3 E2E lane, delivered its final report and exited). No codex lanes of mine are running; one live `codex exec` process on this machine belongs to the OTHER session — untouched.
- **No wakeups scheduled. No monitors running.** The autonomous loop is stopped.
- The Vite viewer dev server on port 5173 is intentionally left running (passive, serves the manifest links above).
- The ball-regen GPU queue on the VM belongs to the other session — intentionally untouched, still running.
- **No commits made. No branches made.** (One commit+push attempt earlier was permission-blocked and not retried; owner deferred commits.) All tonight's work is uncommitted working-tree state + the VM file sync described above.
- **BALL work intentionally excluded** from this handoff and from all remaining task lists per owner directive.
