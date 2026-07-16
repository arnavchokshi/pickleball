# MOVE-1 41-rally pb.vision head-to-head RE-RUN — Track A manager report

Date: 2026-07-15 (PDT) / close 2026-07-16T02:51Z. Author: Track A manager session.
Specs: `../spec.md` + `../spec_rerun_20260715.md`. Prior attempt: 2026-07-13 (no result).

## STATUS RULING: **partial**

The mission's core deliverable — a per-rally full-3D head-to-head scorecard of our promoted stack
vs pb.vision on the 697s demo video — was NOT produced (second consecutive attempt). It is now
blocked on a precisely-located code defect, not on budget, auth, or infrastructure. Salvage
deliverables were produced and verified. `VERIFIED=0` stands; nothing here is a promotion; all
pb.vision data remained R&D reference only (denylist scans clean).

## What ran

- VM `pickleball-h100-pbv11r` (H100-80GB a3-highgpu-1g SPOT, us-central1-a on create-ladder
  attempt 3/6 after asia-southeast1-b/-c stockouts). RUNNING 2026-07-15T22:53Z.
- Run identity, ALL two-sided verified: code pin `ac0b14ab0d3a5c00418671f84a725affc54a8213`
  (git bundle sha `6e567499e88d537ef23fc6d1bfb6da7d96dd14323acf9c3fd72748fe74d7b674`), source video
  sha `272a2132ce7c72ea31fe6351c9ea05ac3016bbbfed0a5801d9c3a973ec628383`, OSNet ckpt sha
  `2809d3227f7d078f6045f7feb874a34d0684f0e0057b264b99adccf7d4519154`, best_stack rev 12,
  clip `pbvision_11min_20260713`, fresh content-addressed generation.
- Command: `process_video.py --video source_video.mp4 --clip pbvision_11min_20260713 --sport
  pickleball --max-players 4 --allow-auto-court-corners-preview --body-local --json` (audio ON).

## Per-stage outcome (times from artifact mtimes; SIGINT wrote no PIPELINE_SUMMARY.json)

| Stage | Outcome | Wall |
|---|---|---|
| ingest/calibration/input_quality | done 23:07Z — but calibration seed = auto-corner PREVIEW, grade **poor** (no trusted seed exists for this video); fail-closes `tracking_court_filter`, `ball_world`, `virtual_world_metric` | ~1 min |
| tracking | no tracks.json produced (court-filter fail-close path); effectively skipped/degraded | ~0 |
| ball (WASB candidates → size → track → bounce) | **done** 23:12–23:44Z, full 20,922 frames | ~37 min |
| ball_arc | **STALLED** — segment 7 candidate association, GIL-bound pure-Python RK4 (~1240 substeps/predict over a ~5.16s gap, large unassigned pool); 3 concurring stack captures over 3h06m; see `pyspy_stall_evidence.md` | 3h06m, unfinished |
| events → manifest | never reached | — |

## Headline salvage numbers (2D ONLY — no 3D claims)

`scorecard_2d_salvage.json` (deterministic; definitions in `salvage_2d_coverage.py` docstring):
over the 42 cv-rally windows (10,322 in-rally frames), **our 2D ball presence 78.4%** (78.0–78.6%
across ±3-frame offsets) vs **pb.vision emitted presence 75.6%** — the PB figure reproduces the
frozen 07-13 forensics' 75.6% to the digit, validating the definition. Caveat: ours is raw WASB
detection `visible`; theirs is deliberately trimmed emission (their upstream detection is lower —
forensics context 80.6% vs 58.7% on a different denominator). The mission-brief comparison against
pb's 183/252 per-rally-3D baseline was NOT computable: we produced no 3D this run.

Owner event-review pack: **NO-RESULT** — the events stage never ran; a pack built from raw unfused
bounce candidates would waste the owner's gated 15 minutes and poison the reviewed-union set. It
remains the first deliverable of the next attempt.

## Findings (the real value of this run)

1. **ball_arc segment-association stall is a real, located, full-game-scale defect**
   (`ball_arc_solver.py` `_select_candidates_for_segment`→`predict` RK4). It has now consumed two
   H100 attempts. Fix lane spec'd, NOT dispatched: `runs/lanes/ballarc_scale_guard_20260715/spec.md`.
2. **Dated correction (appended to HANDOFF_20260714.md + fleet ledger):** the 07-13 attempt's death
   was attributed to the Fable spend limit; it most likely hit this same stall first.
3. **Calibration prerequisite:** without a trusted CAL seed for this video, metric world output
   fail-closes and even a completed run cannot yield a court-frame 3D comparison. Next attempt
   needs a reviewed seed (e.g., owner 4-corner taps on one frame) — carried into trust-band notes.
4. **compare_vs_pbvision.py crashes on the full 11-min export** (PB physics pillar, scipy
   least_squares x0-out-of-bounds) — first full-scale invocation ever; frozen original must stay
   byte-identical, v2 fix in the follow-up lane. Its committed tests also carry a `parents[3]`
   root-path bug from the dir move (pre-existing).
5. Snapshot re-bake needs: OSNet ckpt AND torchreid==0.2.5 (both hand-installed this run).
6. Ops: in-VM self-stop MUST be manager-verified at dispatch (it wasn't armed; Mac sleep killed
   Mac-side watchers; manager armed `sudo shutdown -P` mid-run as the hard rail). SIGINT does not
   produce PIPELINE_SUMMARY.json (KeyboardInterrupt escapes `_run_stage_safely`).

## Teardown + cost (unconditional deliverable — DONE)

DELETE at 2026-07-16T02:50:53Z (API-confirmed); instances list shows only non-fleet
body4d-waker-ctrl + historical pickleball-a100-fleet1 TERMINATED; disks list shows 0 lane-created
disks. Wall 22:53:00Z→02:50:53Z = **3.93h**; spot band $2.2–3.7/hr → **est $9–15** (not
invoice-backed). Under both the ~$20 stop-loss and the 5h wall cap (early teardown on stall ruling).

## Artifact inventory (all under `runs/lanes/pbv11_headtohead_20260713/rerun_20260715/`)

- `vm_pull_partial/` — 26 files, two-sided md5 verified (`vm_md5_manifest.txt` vs
  `mac_md5_manifest.txt`): full-game ball_track/candidates/bounce/size artifacts, calibration set,
  run_stdout.log (SIGINT traceback), pipeline_run.json.
- `scorecard_2d_salvage.json` + `salvage_2d_coverage.py` — the 2D salvage slice (deterministic).
- `pyspy_stall_evidence.md` — the stall stack + timing context.
- Provision evidence: `create_attempt_*.log`, `create_loop*`, `vm_create_success.txt`, `vm_ip.txt`,
  `wall_cap_start.txt`, `denylist_scan_1_precopy.txt`, `xfer/`.
- Stale 2026-07-13 STOP report remains at `../report.json` (never a result; superseded by this file).

## Recommended next step

Dispatch `ballarc_scale_guard_20260715` (after coordinator sequences it against Track C
coordwire/tbwire fences) + secure a trusted CAL seed for the demo video; only then re-run MOVE-1.
The stall reproduces CPU-only from the pulled artifacts — no GPU spend is needed for the fix lane.
