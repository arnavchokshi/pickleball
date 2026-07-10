# fixv_viewer_20260710 — viewer truth + quality Wave V

Date: 2026-07-10  
Scope: `web/replay/**` plus this lane report only. No branch, commit, or staging operation was performed. `VERIFIED=0` remains binding.

## Decision table

| Item | Ruling | Implementation evidence | Red-to-green tests |
|---|---|---|---|
| 1. Presence | **Implemented** | Canonical player presence and prior floor anchor: `viewerData.ts:1143-1162`. Header gap metric and faded gap labels: `App.tsx:1129-1135`, `1396-1400`. Dashed missing-evidence ring: `App.tsx:2307-2391`. Separate solid-mesh gap path: `App.tsx:2401-2447`. | `viewerData.test.ts:75` all-null pose retains prior anchor; `App.test.tsx:150` truth-surface wiring. |
| 2. Skeleton gates | **Implemented** | Parent count includes explicitly enabled implausible frames: `viewState.ts:309-313`. Suppressed implausible skeleton selects missing-evidence placeholder: `App.tsx:2278-2313`. | `viewState.test.ts:341` all-implausible parent remains visible when enabled; existing implausible skeleton unit coverage remains green. |
| 3. Paddle | **Implemented** | Hold constants and fade/drop: `viewerData.ts:515-521`, `2741-2753`; active opacity: `viewerData.ts:1319-1339`. Default-off review normals: `viewState.ts:24`, `46`, `84`; render gate: `App.tsx:2517-2547`. Rounded beveled dimensions proxy and amber estimated material: `App.tsx:3413-3510`. | `viewerData.test.ts:89` opacity envelope; `App.test.tsx:650-695` estimated rounded/beveled proxy; `App.test.tsx:150` normal control wiring. |
| 4. Ball | **Implemented** | Exact `physics_predicted` band and styling: `ballTrail.ts:3`, `219-255`, `392`. Segment-gap hidden sentinels: `ballTrail.ts:296-369`; dense interpolation refuses unbridged boundaries: `ballArcRender.ts:96-129`. Dead `Ball` and `BallGhostMarkerRing` removed. KPI and layer count receive `resolvedBallSamples`: `App.tsx:894-915`, `927-956`; `viewState.ts:278-301`. | `ballTrail.test.ts:42` exact band/HUD; `ballTrail.test.ts:103` no cross-segment line; `ballArcRender.test.ts:85` no unbridged interpolation; `App.test.tsx:150` dead component check. |
| 5. Trust | **Implemented** | Mounted compact player/ball/paddle badges: `App.tsx:1138-1148`; missing optional capability badges: `App.tsx:1150-1155`. Missing mesh badge fails safe to preview: `App.tsx:350-358`. | `App.test.tsx:103` absent mesh badge is estimated preview; `App.test.tsx:150` mounted trust surface. |
| 6. Load robustness | **Implemented** | Optional artifacts are isolated per capability: `App.tsx:591-682`; helper returns `null` and reports the named capability: `App.tsx:3612-3624`. Manifest and virtual world remain the only fatal core fetches. | `App.test.tsx:140` rejected optional loader resolves null and records capability. |
| 7. Markers | **Implemented** | Event layer defaults on: `viewState.ts:89`. `ImpactMarkers` mount depends on the user toggle, not transient active count: `App.tsx:1368-1386`. Explicit no-source state: `App.tsx:961-966`, `1401`. | `viewState.test.ts:195-207` new defaults; `viewState.test.ts:277-330` on/off layer consequences; `App.test.tsx:150` empty-state wiring. |
| 8. Seek/time | **Implemented** | Canonical nearest-PTS resolver and declared holds: `viewerData.ts:515-548`. Used by player/world-ball: `viewerData.ts:1131-1171`; mesh: `viewerData.ts:2134-2189`; paddle: `viewerData.ts:2741-2753`; dense-ball fallback: `ballArcRender.ts:96-121`. Per-entity stale debug readout: `App.tsx:941-944`, `1430-1449`. | `viewerData.test.ts:68` internal gap refusal and stale age; `ballArcRender.test.ts:85` bounded ball fallback. |
| 9. Mechanical perf | **Implemented** | LRU raised from 2 to 6 and adjacent chunk is prefetched in playback direction: `App.tsx:734-833`. Solid geometry cache and paddle `useMemo` remain active; no FPS claim is made. | `App.test.tsx:165` next/previous window selection; existing solid geometry reuse tests remain green. |

## Verification

- Baseline: 20 Vitest files, 235 tests passed; TypeScript typecheck passed.
- Final: `cd web/replay && npx vitest run` — 20 files, **245 tests passed**, 0 failed, 0 skipped.
- Final: `cd web/replay && npm run typecheck` — passed.
- `git diff --check -- web/replay` — passed.
- Required structure checks: scaffold index completed with 283 tools and zero missing direct-CLI tests; dead-code audit `status: pass`; storage audit `status: fail` from pre-existing missing allowlisted artifacts/generated caches outside this lane.

## Honest issues

- No Vite server, browser, Playwright, screenshot, or device pass was run because socket-bound viewer execution is forbidden in this lane. Visual layout is not browser-proven here.
- The changes make missing, preview, predicted, and suppressed evidence more explicit; they do not improve detection, tracking, paddle pose, ball accuracy, or runtime performance.
- Unrelated dirty files outside the owned scope and the pre-existing untracked `web/replay/public/` directory were preserved.
- `VERIFIED=0` remains binding.

BEST-STACK DELTA: **none** — presentation truth only; no model or policy flip.
