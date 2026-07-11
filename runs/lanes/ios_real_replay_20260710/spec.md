# LANE ios_real_replay_20260710 — kill the bundled-fixture replay: wire real manifest routing (NS-01.5 native half + P0-B)

Ground truth: runs/lanes/plan_nextmoves_20260710/PLAN.md rank 2 (verified code facts:
AppRootView.swift:1631-1703 unconditionally calls WorldBundle.loadBundledSample(); UploadQueue has
jobId but no persisted manifest URL), runs/lanes/ns015_statuspack_20260710/handoff.md (native-app
hunk: Swift decoder rejects honest `partial`; missing trust/missing_capabilities fields),
runs/research_deepreview_20260710/RULINGS.md §S6 (iOS schema lacks ball_arc_render_url),
NORTH_STAR_ROADMAP.md P0-B/P0-E rows. Read all FIRST.

## HARD RULES
- No branches/commits/git add. Manager commits after ruling.
- FILE OWNERSHIP: ios/App/{AppRootView,DinkVisionModels,DinkVisionRuntimeConfiguration,
  DinkVisionUploadCoordinator}.swift; ios/Upload/Sources/PickleballUpload/{RenderGatewayClient,
  UploadQueue}.swift; ios/Replay/Sources/PickleballReplay/World/{WorldBundle,WorldViewerManifest}.swift;
  their exact Swift test targets; runs/lanes/ios_real_replay_20260710/**.
  FORBIDDEN: capture/recording code, signing/provisioning, web/, server/, scripts/, threed/.
  Do NOT touch .xcode-home/ or any owner working state.
- NO fixture fallback on ANY error path: a row without ready output shows an explicit typed
  not-ready/error state; the bundled sample renders ONLY for the explicit sample row.
- Honest report; NS-01.2b (physical proof) stays owner-blocked — say so; no device claims.

## MISSION
1. Persist manifest identity: when a job completes, persist the ready manifest URL (and capture/job
   identity chain) on the upload row; "Open" on that row loads ITS OWN manifest via WorldBundle from
   the persisted URL — never loadBundledSample().
2. Honest status decode: accept server `partial` + `missing_capabilities` + trust/provenance fields
   per ns015 handoff native hunk (re-derive; git apply --check your own version); partial renders
   what exists with explicit missing-capability UI state (data model + view state; polish not required).
3. Schema: add ball_arc_render_url (+ any URL fields the web manifest has that iOS drops — diff the
   web manifest parser fields vs WorldViewerManifest) with optional/backwards-compatible decoding.
4. Tests: URLProtocol-faked gateway + temp-file manifests proving (a) row->own-manifest identity,
   (b) complete and partial decode, (c) relative HTTP + file URL resolution, (d) missing optional
   assets do NOT substitute another capture, (e) not-ready row state, (f) no fixture fallback.
5. Verification: attempt `swift test --package-path ios/Upload` and the Replay package (and any
   package with your tests). If the sandbox blocks SwiftPM builds (cache write), write the tests
   anyway + document the exact failure; the manager runs swift test + simulator build-for-testing
   at ruling. Do not fake pass counts.
## REPORT
report.json via schema; per item status + tests + file:line; HONEST ISSUES (incl. exactly what only
a physical device can still prove); BEST-STACK DELTA (c) none.
