# Track H ruling — webux3 viewer usability wave + repair round 1 (2026-07-16)

**RULING: ADOPT (full wave: round-1 items T1–T11 + repair round 1).**
Ruled by the second-shift Track H manager after the ~01:30 host sleep killed the
first-shift manager mid-repair. The repair codex run itself had already completed
(report.json + log_repair1.txt written 02:01); only manager browser verification
was outstanding. No codex resume needed.

## What the overnight lane shipped (round 1, verified by first-shift manager)
- Replay-first layout: video pane top y 786 → 226; upload collapsed below workspace.
- Single shared timeline: native controls removed, scrub strip + 168 glyph markers +
  rally band + inline degraded-reason on empty timelines; explicit play/pause.
- In-pane per-entity trust badges (players/ball/paddles), consolidated
  layers-unavailable chip, expandable warnings with structured degraded reasons.
- Dev-bypass hint; court-play FPS 37.5 → 47.7 (n=3 headed Metal).
- First-shift browser pass: 10/13, results in manager_verify_result.json.

## Repair round 1 items — second-shift browser verification (this ruling)
Environment: headed Chromium, ANGLE Metal, 1600x1000, real bundles only
(fresh_wolv rich bundle, trk_flip degraded bundle, demo_beststack VM-written manifest).
Results: manager_verify2_result.json (15/16), fps_recheck_result.json,
fps_matched.py (segment-matched), shots_repair1/.

### 1. Follow-player playback FPS (bar: >=80% of court-play) — PASS
- Pre-repair: court 47.7 / follow 17.9 (ratio 0.375; n=3, manager_verify_result.json).
- Harness hazard found and fixed this shift: the 10.0s clip could END mid-probe;
  an ended video lightens the render loop and inflates FPS (the round-2 trial-3
  court=116.5 artifact, and likely round-1 follow=12.75 noise). Re-measured with
  loop=true + playing-state asserted before/after every probe + alternating order.
- Guarded n=4/preset: court 56.9 / follow 46.1 avg → ratio 0.809 (fps_recheck_result.json).
- Segment-matched n=4 pairs (both presets probe the SAME video segment):
  early segment follow 71.1/68.6 vs court 69.4/63.1 (parity); heavy tail segment
  follow 31.1/34.9 vs court 48.0/40.0 (both presets dip). Overall court 55.1 /
  follow 51.4 → ratio 0.933. PASS both harnesses.
- Follow-paused probe 118.5 rAF fps (round-1 badge ~97): no paused regression.
- Residual (non-blocking): the clip's last ~5s is heavier for BOTH presets
  (court 40–48, follow 31–35). Preset-specific collapse is gone; tail-segment
  render cost is a shared, pre-existing cost — ranked in remainder.

### 2. VM-written manifest recovery (T10) — PASS
- Repro URL: ?manifest=/@fs/…/demo_beststack_render_20260710/…/replay_viewer_manifest.json
  (valid JSON manifest, all asset URLs dead VM-absolute /@fs//home/arnavchokshi/…).
- Pre-repair: raw "Unexpected token '<'" banner, viewer never opened.
- Now: viewer FULLY OPENS — video readyState>=1, canvas up, rally band + markers,
  9 in-pane trust badges — with loud counted banner "9 assets resolved
  manifest-relative — original absolute paths unreachable (VM-written manifest)".
  No raw JSON token error anywhere. shots_repair1/repair_04_vm_manifest.png.

### 3. Badge/chip overlap (cosmetic) — PASS
- Geometric check on the degraded bundle: zero pairwise bounding-box intersections
  among the 6 honesty-dock elements (5 trust cards + unavailable-layers chip),
  collapsed AND expanded; dock right edge 1573 <= pane right 1586.
  shots_repair1/repair_02_degraded_bottom.png, repair_03_degraded_chip_open.png.

## Gates re-run by this manager (real, unpiped exit codes)
- vitest: 280/280 passed, 22 files — EXIT 0 (round-1 baseline was 273).
- tsc typecheck — EXIT 0. vite build — EXIT 0 (pre-existing chunk-size warning only).
- Regression: round-1 adopted surfaces re-checked in browser (video top 225.75,
  timeline strip + 168 markers, no native controls, trust vocab, degraded
  missing-stays-missing) — all green; zero page errors on all three bundles.

## Trust contract (NORTH_STAR §1.4)
Badges are MORE prominent (in-pane per-entity, "Entity trust · N badges in replay
pane" header chip); missing entities stay loudly missing (Ball: missing / NO TRUST
BAND; mesh: no computed frames; inline timeline absence reasons). No fabrication:
all verification on real run bundles; VM-manifest recovery resolves to the real
neighboring artifacts, never synthesizes.

## Fence audit
Predispatch status shows only web/replay/public/ dirty under web/replay before this
track's lanes: all 14 modified + 4 new web/replay files are this track's work.
web/replay/public/ (Jul 9 symlink into w7_critique world, predates lane) left
untracked. Ledger staged as Track H row hunk only (Track C rows left uncommitted).
Committed by explicit path: web/replay/** + curated lane evidence (raw codex logs
log.txt 11M / log_repair1.txt 1.3M, pid files, vite logs NOT committed, per repo
precedent that runs/* stays ignored except deliberate evidence).

## Ranked remainder (for a future wave; none blocks adoption)
1. Heavy tail-segment render cost (last ~5s: both presets 31–48 fps) — profile the
   shared per-frame world/render path, not the camera path.
2. Bottom-left court/mesh status chips can crowd/collide on the degraded bundle
   (outside honesty dock) — same wrap treatment as the honesty dock.
3. Loop-wrap hitch: FPS dips when the video loops (seek re-buffer) — cosmetic for
   normal (non-looping) viewing.
4. Build chunk-size warning (pre-existing): manualChunks split for three.js.
