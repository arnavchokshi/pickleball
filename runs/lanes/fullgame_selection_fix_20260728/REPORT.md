# Full-game player_selection fix — first complete 11-minute game bundle

Lane: `fullgame_selection_fix_20260728`. Date: 2026-07-29.
Commits: `dadfcd4` (fix + regression test + hang mitigation), `7123076`
(recovery evidence + full-scale repro script), `af26cf3` (rerun evidence).
Status: engineering fix + first-ever complete full-game run; `VERIFIED=0`.
Written by the orchestrator from the lane's returned findings.

## Root cause (recovered from the preserved disk, reproduced at full scale)

`ValueError: bound, unbound, and dropped raw UID sets must be disjoint`
(`player_selection.py:3605`), 267 UIDs in both unbound and dropped sets.
Mechanism: Layer-C recovery pulls a raw detection from an unbound fragment into
`used_uids`; if its source track is later lifetime hard-excluded
(off-court beyond `COURT_REGION_HARD_BOUND_M`), the hard-drop pass removes it
from `used_uids` and records it in `dropped_uids` — but the residual-unbound
recomputation excluded only `used_uids`, resurrecting the UID from the stale
pre-recovery snapshot. Rare on 300-frame clips; guaranteed reachable at
20,922-frame scale. Reproduced CPU-locally by feeding the REAL recovered
artifacts into `select_players_payload()` (`repro_full_scale_selection.py`).

## Fix

`_select_slot_players`: compute `dropped_uids` first; exclude
`used_uids | dropped_uids` from the residual recomputation. One function; no
`process_video.py`/best_stack changes. TDD regression test raises the exact
pre-fix error on stashed-implementation code and passes post-fix.
Tests: selection suites 108 passed (1 pre-existing revision-pin failure);
full `tests/racketsport/` 5209 passed / 29 pre-existing unrelated failures.

## Post-failure hang (separate defect)

Evidence shows `main()` returned promptly with the typed error written; the
2.5 h was interpreter shutdown after return. Mitigation: CLI `__main__` guard
flushes and calls `os._exit(code)`. Evidence-based defense-in-depth, not a
live-reproduced hang (declined to re-trigger a multi-hour hang deliberately).

## Full-game reruns (court23, synced `dadfcd4`)

player_selection SUCCEEDED four consecutive times (476.0/485.8/475.8/476.9 s).
Run #4: **all 15/15 stages completed, zero hard failures, real
`replay_viewer_manifest.json` produced — the first complete full-game bundle.**
Overall status `partial` (honest): BODY degraded to skeleton-only past frame
1200. **Total wall 2749.9 s (~45.8 min) for the 697.4 s game ≈ 3.9× video
duration**; tracking 1825.2 s (11.46 fps, matches baseline).

## Named next unlock (out of this lane's fence)

`threed/racketsport/process_video_body_frames.py:76`
`DEFAULT_MAX_SCHEDULED_FRAMES = 1200` — a short-clip-sized hardcoded cap that
silently truncates BODY frame scheduling at full-game scale. Removing/scaling
it (with an explicit budget policy) is what turns the full-game bundle's BODY
coverage from partial to complete.
