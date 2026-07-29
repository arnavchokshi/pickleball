# Full-game (697s pb.vision) co-located attempt — honest partial, failed at player_selection

Lane: `fullgame_demo_20260728`. Date: 2026-07-28. Evidence commit: `2c376d9`.
Status: FAILED before completion; `VERIFIED=0`. Written by the orchestrator
from the lane agent's returned findings (its harness refused report-file
writes); the ledger entry in `runs/manager/inflight_lanes.md` references this
file. Surviving evidence: `vm_pull/captured_pipeline_summary_stages.json`.

## What ran

Full 697.4 s / 20,922-frame pb.vision demo dispatched co-located on
`pickleball-gpu-court23` (`court_skeletons`, `--body-local`, `--force`,
`--max-players 4`), VM synced+stamp-verified at main `c4c892e`, video + the
promoted owner-reviewed calibration seed scp'd (md5-checked).

| Stage | Status | Wall seconds |
|---|---|---:|
| ingest | ran | 46.599 |
| calibration | ran | 36.899 |
| input_quality | ran | 0.283 |
| tracking | ran | **1839.011** |
| player_selection | **failed** | 448.698 |
| remaining 11 stages | never reached | — |

**The one genuinely new capacity number: tracking sustained 11.38 fps across
the full 20,922-frame clip — the first-ever measurement of tracking throughput
at true full-game scale (A100).** Top-level wall at failure 2376.7 s for 5 of
16 stages — NOT comparable to any completed-run baseline, and no ≤2×/NS-06
ratio is computable from it (declined to manufacture one).

## Failures, honestly

1. `player_selection` failed with a typed JSON failure, no traceback captured;
   root cause UNKNOWN (the summary `notes`/`degraded_reasons` were never
   retrieved before the VM died).
2. The process then **hung ~2.5 h with zero progress** after the stage failure
   — a separate, real reliability defect (post-failure hang instead of exit).
3. The VM's pre-armed poweroff rail (20:10:26Z) killed the box mid-hang; the
   dispatch brief's "~11 h budget" was stale — the real rail was ~3.4 h out at
   dispatch. The lane verified this itself; its bounded rail-extension attempt
   was blocked by the permission system and it correctly did not route around
   the denial.
4. `tracks.json` for the full clip (tracking succeeded, so it almost certainly
   exists on the preserved disk) was never pulled before poweroff — named
   process lesson: pull stage artifacts the moment each stage manifest lands.
5. No BODY/placement/world/manifest ran; no replay bundle exists.

## Follow-ups (daytime queue)

- Restart court23, pull the preserved full-game `tracks.json` +
  `PIPELINE_SUMMARY.json` failure fields, root-cause `player_selection` at
  20k-frame scale (first suspects: memory/cardinality assumptions sized for
  10 s clips), fix, and re-run the full-game demo co-located (GPU compute mode
  Default for `--body-local`, per `bodylocal_colocated_fix_20260728`).
- Fix the post-failure hang so a failed stage exits the run instead of
  burning hours.
- Keep the 11.38 fps full-scale tracking number as the planning basis for
  full-game compute budgeting (~31 min tracking per 11.6 min game at current
  settings — a named NS-06 target once correctness closes).
