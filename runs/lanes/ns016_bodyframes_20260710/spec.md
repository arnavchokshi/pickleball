# LANE ns016_bodyframes_20260710 — cold-clip BODY frame-materialization regression: root-cause + fix

## HARD RULES
No branches, no commits, no pushes. Read NORTH_STAR_ROADMAP.md (§2.1 P0-H, §4 NS-01.6) + AGENTS.md first.
.venv/bin/python; MPLBACKEND=Agg for any pytest. 4 protected eval clips are EVAL-ONLY (Burlington/
Wolverine internal scoring allowed; Outdoor/Indoor labels NEVER). Honest reporting: a repro you could
not achieve, a root cause you could not pin, or a fix you could not verify is a RESULT — say it plainly.
Wide blast-radius suite at the end (MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q),
failures must be 0 or proven pre-existing. Artifacts under runs/lanes/ns016_bodyframes_20260710/.
Your sandbox has no network and no localhost binds; everything here is local CPU work.

## FILE OWNERSHIP (exclusive — you are the process_video/orchestrator integration owner this wave)
- scripts/racketsport/process_video.py
- threed/racketsport/orchestrator.py
- threed/racketsport/process_video_body_frames.py
- tests/racketsport/test_process_video_body_frames.py + ONE new regression test file if needed
- runs/lanes/ns016_bodyframes_20260710/**
Do NOT touch: server/**, threed/racketsport/court_calibration_metric15.py,
scripts/racketsport/{build_real_court_corpus,import_w6_labelpack_tasks,ingest_owner_ball_labels}.py
(other live lanes own them).

## THE BUG (reproducible, 3 banked signatures)
Cold non-eval clips fail the BODY stage with `missing BODY frame image for frame N` (raise site:
threed/racketsport/orchestrator.py:3287) at BOTH --body-skeleton-stride 2 and stride 1 — the frames
stage never materializes the frame regardless of schedule. Wolverine (eval_clips) is UNAFFECTED.
w7_critique processed the same harvest clip fine at manifest rev-9 code =>
REGRESSION WINDOW: 460992ae9..d47b399a1 (contains the ns06 BODY-efficiency wiring; prime suspects:
bounded frames schedule + array-native BODY path wired default in d47b399a1).

## EVIDENCE TO READ FIRST
- runs/lanes/demo_beststack_20260710/REPORT.md §2 (the 3 signatures + clip identity)
- runs/lanes/demo_beststack_gpu_20260710/report.json (zwcth45s R1/R2 context)
- runs/manager/gpu_fleet.md rows pickleball-h100-demo2 (r3 stride-1 repro)
- git log 460992ae9..d47b399a1 -- threed/racketsport/orchestrator.py threed/racketsport/process_video_body_frames.py scripts/racketsport/process_video.py

## MISSION
1. LOCAL REPRO FIRST (no GPU): reproduce the EXACT signature on a cold clip. Preferred: the same
   harvest excerpt the demo lane used (identity in its REPORT.md; source videos under
   data/online_harvest_20260706/rallies/**). Use a FRESH clip id (cold = no cached stage dirs).
   You do not need BODY inference to complete — the raise happens when BODY input frames are
   assembled; a run that dies with the exact message IS the repro. If a full-pipeline local run is
   impractical, a unit-level repro driving the real frames-materialization + schedule code path for
   the real clip is acceptable ONLY if it raises the identical message for the identical frame set.
2. ROOT CAUSE inside the window: identify the exact commit + code path that stopped materializing
   scheduled frames for cold clips while leaving eval_clips unaffected (cache/warm-dir difference is
   likely load-bearing — explain WHY wolverine survives).
3. FIX with the smallest correct change: the frames stage must materialize exactly the schedule the
   BODY stage will request, or fail loudly AT THE FRAMES STAGE with a typed error naming the missing
   frames. Do NOT revert the ns06 efficiency path wholesale — preserve its gains and fix the defect.
   No silent BODY degradation, no schedule truncation to dodge the raise (an unfailable gate by
   exclusion = fraud class; the missing frames must actually exist after the fix).
4. REGRESSION TEST: fails pre-fix (exact signature), green post-fix. Also prove schedule==materialized
   set equality on BOTH the repro clip and one eval clip (wolverine) via a cheap assertion path.
5. Wide suite + focused tests.

## ACCEPTANCE (all required for PASS)
- A1: pre-fix repro log with the exact `missing BODY frame image for frame N` message banked in lane dir.
- A2: named root-cause commit + mechanism, with the eval-vs-cold asymmetry explained.
- A3: post-fix, the SAME command completes the frames+BODY-input assembly with zero missing frames
  (schedule-vs-materialized equality printed for both clips).
- A4: regression test red-then-green evidence (run it at pre-fix code via git stash or equivalent, log both).
- A5: wide suite 0 new failures.
## KILL RULE
If the root cause is provably OUTSIDE 460992ae9..d47b399a1 or is clip-data corruption, STOP after
step 2 with the diagnosis — no speculative fix.
## BEST-STACK DELTA (mandatory in report)
Expected (c) NO stack delta (default-path bug fix, no manifest revision change). State explicitly;
if your fix changes any default knob semantics, that IS a delta — describe it for manager ruling.
## REPORT
Schema-valid report.json (lane_report.schema.json): objective_result PASS/FAIL/PARTIAL vs A1-A5,
full_suite counts, honest_issues, root-cause commit, artifacts list, and the exact repro command.
