# LANE w3_meshfallback_20260707 — non-promotional mesh-frame fallback (wave-3 #7 FIX)

## HARD RULES (violating any = lane rejected)
- You are Codex lane `w3_meshfallback_20260707`. Work ONLY inside /Users/arnavchokshi/Desktop/pickleball.
- No git branches/commit/push. Write runs/lanes/w3_meshfallback_20260707/commit_manifest.md for the manager.
- OWNED files: `threed/racketsport/frame_rating.py`, `threed/racketsport/body_mesh_readiness.py` (only if genuinely needed), their tests (tests/racketsport/test_frame_rating*.py etc.), runs/lanes/w3_meshfallback_20260707/.
- FENCED (do NOT edit; deferred patches under runs/lanes/w3_meshfallback_20260707/deferred_patches/ if required): `scripts/racketsport/process_video.py`, `threed/racketsport/placement.py`, `threed/racketsport/foot_contact.py`, `threed/racketsport/foot_lock_solver.py`, `threed/racketsport/footlock.py`, `threed/racketsport/body_grounding_refine.py`, `threed/racketsport/worldhmr.py`, `threed/racketsport/body_grounding_quality.py`, `threed/racketsport/pose_temporal.py`, `threed/racketsport/camera_motion.py`, `scripts/racketsport/remote_body_dispatch.py`, `scripts/fleet/*`, `threed/racketsport/roboflow_corpus.py`, `threed/racketsport/ball_tracknet_cvat_dataset.py`.
- TRUST HONESTY IS THE LINE: the fallback must NOT promote trust — no changes to trust bands, manual_review_required flags, confidence values, or gate outcomes. It only schedules mesh COMPUTE on frames that remain honestly labeled as they are.
- Protected data: 4 eval clips EVAL-ONLY; held-out harvest videos appear nowhere.
- WIDE suite `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q --ignore=tests/racketsport/court_finding_technology_benchmark.py` + focused suites; classify residuals REAL / PRE-EXISTING / SANDBOX-SUSPECT / CROSS-LANE-SUSPECT (several lanes run concurrently).
- importorskip("torch"); no new root .md; `.venv/bin/python` always.
- Read first: BUILD_CHECKLIST last ~15 bullets; `runs/lanes/w3_img1605_mesh_diag_20260707/REPORT.md` (defines this lane).

## CONTEXT (from the accepted diagnosis)
img1605 produces ZERO mesh frames: `frame_compute_plan.json` shows selected_mesh_frame_count=0 AND eligible_mesh_frame_count=0 — the contact-dense (`ball_aware`) scheduler finds no trusted contacts/proximity/swing on this handheld clip, and the EXISTING uniform fallback is starved because ALL img1605 frames are `manual_review_required`, which the eligibility predicate excludes. Burlington's HUD "missing mesh vertices" notice is a DISTINCT virtual_world issue (burlington has 166 mesh player-frames) — OUT OF SCOPE here.

## THE DESIGN (manager-ruled)
When the scheduler's eligible set is empty (or below a minimal floor) after normal eligibility filtering, engage a NON-PROMOTIONAL fallback: select a bounded uniform stride of frames for mesh compute regardless of manual_review_required status, WITHOUT altering any trust/review/confidence field, and record in frame_compute_plan provenance: `mesh_fallback: {engaged: true, reason: "eligible_zero_all_manual_review", selected: N, policy: "uniform_stride"}`. When normal eligibility yields frames, behavior is BIT-IDENTICAL to today.

## ACCEPTANCE
- Unit test reproducing img1605's actual frame_compute_plan inputs (from runs/lanes/wave2_freshworlds_20260707/): fallback engages, selected_mesh_frame_count ≈ 100 (match the diagnosis's expected count; state the exact policy), provenance record present, zero trust/review fields mutated (assert deep-equality of those fields pre/post).
- Regression tests: burlington/wolverine/outdoor plan inputs → fallback NOT engaged, plans byte-identical to current behavior.
- Wave-end GPU assertion list in your report (the diagnosis's): img1605 produces >0 mesh-index chunks; viewer HUD no longer reports zero mesh; scheduling provenance shows fallback engaged only on img1605.
- WIDE suite green per HARD RULES.

## STRUCTURED REPORT
objective_result vs acceptance; acceptance table; changes file:line; full_suite + classification; HONEST ISSUES; NEXT; commit_manifest path; BUILD_CHECKLIST bullet DRAFT.
