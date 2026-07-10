# LANE fixv_viewer_20260710 — viewer truth + quality wave (owner deep-review rulings, Wave V)

Ground truth for every item: runs/research_deepreview_20260710/RULINGS.md +
runs/lanes/dr_viewer_20260710/FINDINGS.md (decision table w/ file:line). Read both FIRST.

## HARD RULES
- No branches, no commits, no git add (manager commits after ruling).
- FILE OWNERSHIP: web/replay/** ONLY, plus runs/lanes/fixv_viewer_20260710/**. Touch nothing else.
  FORBIDDEN: scripts/**, threed/**, server/**, ios/**, configs/**, any other runs/ dir.
- Every behavior change ships a red→green unit test in the same pass (vitest; wire new files into
  the package.json typecheck list). `cd web/replay && npx vitest run` and the typecheck script must
  both pass at the end (they pass today — keep them green; sandbox cannot bind sockets, do NOT
  start vite dev servers or Playwright).
- Honest report; no fps or accuracy claims (presentation-truth work only).
- NO fail-open regressions: never render suppressed/hidden ball samples as measured; never render
  missing entities as present.

## MISSION (each item cites the audit finding; keep changes surgical)
1. PRESENCE: in the Player component, add an explicit low-opacity "no detection" placeholder tier
   when the nearest world frame has all-null pose (joints_world empty + floor_world_xyz +
   track_world_xy null) or no frame exists inside temporal coverage (App.tsx:2139-2200, 3079-3117).
   Style clearly as missing-evidence (dashed floor ring at last-known position + faded label),
   badge it, and add a per-player coverage-gap indicator to the header stats. SolidBodyMeshes is a
   SEPARATE path (viewerData.ts:1214-1252) — give it its own no-coverage handling; do not conflate.
2. SKELETON GATES: (a) parent skeleton visibility must be (plausibleCount>0 || implausible toggle
   on) (viewState.ts:284-323); (b) `App.tsx:2143` — when skeleton_implausible && the implausible
   toggle is off, fall back to the floor-proxy placeholder from item 1 instead of rendering nothing.
3. PADDLE: (a) bound the nearest-frame hold (viewerData.ts:2679-2687) — beyond ~2 source frames
   (~0.083s @30fps, make it a constant) decay opacity to zero by ~0.25s and drop the paddle
   (no unbounded stale holds; Wolverine has 1.467s gaps); (b) hide the 0.52m normal arrow + tip by
   default behind a review toggle (App.tsx:2279-2338, 3308-3349); (c) improve the proxy: rounded
   beveled face from pose dims, amber translucent estimated material, compact preview badge.
4. BALL: (a) preserve the `physics_predicted` band end-to-end — do not collapse to arc_interpolated
   (ballTrail.ts:392-430); distinct styling + HUD label; (b) trail builder must break on segment_id
   change unless an explicit trusted bridge record exists, and insert hidden sentinels for
   suppressed spans (ballTrail.ts:270-314) — unit-test that no line crosses a suppressed interval;
   (c) delete dead `Ball`/`BallGhostMarkerRing` components (App.tsx:2572-2597) or mount-and-justify;
   (d) derive the ball KPI tile/scene count from the same resolved sample source that draws
   (viewState.ts:284-300 vs BallTrailLayer.tsx:41-69).
5. TRUST: mount compact per-entity trust badges (the unmounted TrustBandPanel/PlayerTrustBandPanels
   at App.tsx:1786-1805 or a leaner equivalent); missing mesh-frame trust_badge must fail-safe to
   preview material, never solid (viewerData.ts:1994-2005, App.tsx:2366-2374).
6. LOAD ROBUSTNESS: optional-artifact fetch failures must not abort the core manifest+video+world
   load (App.tsx:575-665) — per-capability error state + missing badge instead.
7. MARKERS: separate persistent landing-spot rendering from the active-event count gate; default
   event layer ON with badges or an explicit empty-state reason (viewState.ts:77-89, 342-372,
   ImpactMarkers.tsx).
8. SEEK/TIME: one canonical time-map helper used by player/mesh/paddle/ball lookups with declared
   per-entity hold tolerances (viewerData.ts:1095-1165, 2097-2124, 2679-2687) — kill the
   divergent-nearest-sample seek-snap family; expose stale-age per entity for debug.
9. PERF (mechanical only, no claims): raise mesh chunk LRU from 2 and prefetch next/previous chunk
   on playback direction (App.tsx:719-767); avoid geometry recreation where trivially memoizable.

## REPORT (runs/lanes/fixv_viewer_20260710/)
report.json via schema + FINDINGS-style summary listing per item: implemented/partial/skipped w/
reason, test names, and file:line of the change. HONEST ISSUES mandatory. BEST-STACK DELTA: (c)
none — presentation truth only, no model/policy flip.
