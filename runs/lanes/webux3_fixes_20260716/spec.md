# Lane webux3_fixes_20260716 — replay viewer usability fix wave (Track H)

You are a Codex implementation lane for the DinkVision repo at
/Users/arnavchokshi/Desktop/pickleball. Your manager (Track H) drove the web replay viewer in a
real browser against two REAL run bundles and produced a ranked friction audit. You implement the
fixes. The manager re-verifies every claim in a real browser afterwards — do not claim visual or
FPS outcomes you cannot prove from tests; describe what you changed and why it should move the
number, honestly.

## HARD RULES
- NO git branches, NO commits. Leave the working tree dirty; the manager commits fence-only.
- FILE FENCE (absolute): you may modify files ONLY under `web/replay/` (src, README.md,
  package.json typecheck list, styles) and write artifacts ONLY under
  `runs/lanes/webux3_fixes_20260716/`. Everything else in the repo is READ-ONLY. Never touch
  `scripts/`, `threed/`, `tests/` (Python), `docs/`, `ios/`, any other lane dir, `NORTH_STAR_ROADMAP.md`,
  `runs/manager/**`. Other lanes are live tonight in disjoint files.
- Read first: `NORTH_STAR_ROADMAP.md` §1.3 (deep-result screen requirements) + §1.4 (trust
  contract), and the manager audit
  `runs/lanes/webux3_audit_20260716/FRICTION_AUDIT_20260716.md` (your requirements document, with
  screenshots under `runs/lanes/webux3_audit_20260716/shots/`).
- HONESTY RULES (binding, from the North Star trust contract): missing entities stay visibly
  missing; preview/trust badges become MORE prominent, never less; no new fabricated stats
  surfaces; honest-absence reporting may be consolidated but never hidden; sample/fixture content
  stays watermarked. Every visible entity keeps a compact, always-visible trust badge.
- Real data only. The two audit bundles are real; never invent replacement fixture data. You may
  add SMALL watermarked unit-test fixtures under `web/replay/src/**/__fixtures__/` only if clearly
  named `*_fixture` and derived from the real excerpt fixtures already present.
- Verification for this lane = the FULL web suite, not a subset:
  `cd web/replay && npm test -- --run` (all tests) AND `npm run typecheck` AND `npm run build`.
  All three must be EXIT 0 with zero failures. New/changed behavior gets red→green vitest coverage
  (component-level where practical). If you add source files, add them to the package.json
  typecheck file list.
- You have NO browser in the sandbox. Do not attempt Playwright. Unit/component tests + typecheck
  + build are your proof surface; the manager owns browser proof.
- Anti-passive-wait: ending your turn to wait = lane death. Work until done; end only with the
  final structured report or a hard blocker.

## Context: current numbers and code suspects (manager-measured, headed Chromium + Metal)
- Paused FPS badge ~119; PLAYING at Court preset 37.5 rAF fps; PLAYING at Follow-player preset
  11.25 rAF fps; headless/SwiftShader badge ~6.
- Named suspect (verify before trusting): `web/replay/src/App.tsx` ~lines 958-976 — the rAF loop
  calls `setCurrentTime()` at video-fps rate; `currentTime` is App-level React state, so every
  sampled frame re-renders the whole 3843-line App and recomputes every currentTime-keyed useMemo
  (`activeBodyMeshes`/`solidBodyMeshFramesForTime`, `ballRenderInfo`, `videoBallOverlay`,
  `activeContactPlayerIds`, `activePaddles`, arc samples). Follow preset adds per-render camera
  pose churn (`poseKey` effect) on top.
- Load flow: manifests written on GPU VMs carry dead absolute `/@fs//home/...` URLs while every
  asset sits beside the manifest locally (all 3 richest recent bundles affected).

## Ordered scope (implement in this order; stop only on hard blockers, report honestly what landed)

### T1 (F1) — put the replay workspace first
When a manifest IS loaded: header (title + stat chips + warnings + FPS badge) stays compact at top;
then IMMEDIATELY the tab row (Shots/3D/Court map), video + 3D panes, shared timeline, and the
layers/isolate controls. The upload panel ("Choose File"/"Predict Court"/pipeline progress) and the
"Latest video runs" switcher move BELOW the replay workspace into collapsed `<details>`-style
sections (collapsed by default when a manifest is loaded, expanded when none is). No capability is
removed; the giant "Predict Court" button must no longer be the most prominent element on a loaded
replay. Target: at 1600x1000 the video pane top edge sits within the first viewport (y < 700) with
default layout; put that assertion into a component test (jsdom layout is limited — assert DOM
ORDER instead: video/3D panes precede upload panel in document order, upload collapsed when
manifest present).

### T2 (F2) — playback/Follow FPS
Decouple playback time from whole-App React state: scene components read canonical time from a ref
(subscription/useFrame), React state updates for chips/status throttled to <=5Hz (chips may lag
playback slightly; timeline playhead should stay smooth — drive it via ref/rAF style updates, e.g.
direct style transform on the playhead element, not per-frame React re-render of the App).
Recompute-per-frame helpers get O(log n) frame lookup + result reuse (no per-frame geometry
reallocation; reuse BufferAttributes where meshes update). Follow preset must not do more per-frame
work than Court. Acceptance numbers (manager-verified in browser after you land): PLAYING-Court
>= 37.5 fps (no regression), PLAYING-Follow within 20% of PLAYING-Court, paused no worse than
baseline. Your in-sandbox proof: unit tests for the throttle/subscription logic + frame-lookup
memoization, existing FPS probe machinery untouched or improved, full suite green. Honest note in
report about what you could NOT measure in sandbox.

### T3 (F3) — one shared timeline that explains itself
- The custom timeline strip becomes the PRIMARY scrub for BOTH video and 3D (it already seeks
  proportionally — preserve that + marker exact-PTS + keyboard nav, they are wave-2 landings).
  Remove the native `controls` attribute from the video element and provide minimal explicit
  play/pause + mute + fullscreen buttons wired to the same canonical time (a11y: keyboard
  accessible). One timeline, §1.3.
- Marker legibility: taller strip; marker GLYPH varies by kind (contact/bounce/inflection/rally
  boundary...) and COLOR stays provenance (measured/model_estimated/physics_predicted — reuse the
  existing legend colors); hover tooltip with kind, t (s), provenance, and trust/authority when
  known; a rally/chapter band row when chapters exist (label not clipped).
- Honest absence inline: when a marker family is absent, the strip states it compactly IN PLACE
  (e.g. "no contact/bounce markers: ball_fill blocked") using the manifest notes' degraded_reasons
  when present (parse `degraded_reasons_json=` from manifest notes; already-real data, do not
  invent reasons). Degraded bundles must never show an unexplained empty strip.

### T4 (F4) — camera presets frame the content
Court preset: fit court + active entity bounding volume to the pane (with margin) instead of the
current small centered court; near-court players must not be cut off at default framing. Follow
preset: frame target player + ball/net context; when the follow target has NO detection at current
time, fall back visibly (existing "no detection" chip stays) and auto-suggest/switch-target
affordance is allowed but must be explicit, never silent. Free orbit unchanged. Reset returns to
the preset's fitted framing. Preserve preset persistence + reduced-motion cuts.

### T5 (F5) — trust badges live with the entities
Per-entity compact trust badges move into/onto the replay surface: a persistent in-pane badge strip
(entity name + provenance + authority, compact pill form, small but always visible, one per visible
entity) replacing the detached top band; the top band collapses into an aggregate trust summary
chip that expands to full detail. Prominence must NOT decrease: in-pane badges are closer to the
entities and always on. Colors/wording follow §1.4 exactly (verified/preview/low_confidence/
too_close_to_call; measured/model_estimated/physics_predicted/missing).

### T6 (F6) — consolidate honest-absence toasts
The stacked per-layer "no data / artifact not supplied" toasts consolidate into ONE compact
always-visible absent-layers chip in the pane ("4 layers unavailable ▸") expanding to the per-layer
reasons verbatim. Never hidden, never softened; enabled-but-absent layers keep their inline
messaging semantics (tests exist around these strings — update tests intentionally, preserving the
honesty contract, not by deleting assertions).

### T7 (F7+F8) — status vocabulary + expandable warnings
One ball-status vocabulary across chip/status-line/trust-pill (state WHAT each surface means:
current-frame visibility vs clip-level trust). Every stat chip gets a title/tooltip explaining
what it counts (e.g. Contacts = contact windows in bundle vs current-frame 3D contact). Warnings
chip becomes click-to-expand listing ALL notices + structured degraded_reasons (stage/status/
reasons) from manifest notes.

### T8 (F9) — Shots panel empty state + reflow
When 0 shots: one honest line ("No classified shots in this bundle — <reason if known from
manifest>"), filters hidden; no black void above the video pane when panels toggle (fix the
layout reflow). When shots exist, nothing regresses (see shotTrails tests).

### T9 (F10) — dev load-flow trap
When the app is on a loopback host in dev mode, a `?manifest=` param is present, but the dev bypass
is NOT enabled and no auth token exists: the sign-in screen shows an explicit dev-hint block
("manifest param detected; to open without auth run the dev server with
VITE_REPLAY_VERIFY_DEV_BYPASS=1") — loopback+dev only, never in prod builds. Document the env var
in the web/replay README quickstart. Tests: devAuthBypass/SignInScreen/AppShell suites extended.

### T10 (F11) — loud manifest-relative asset recovery (stretch, only after T1-T9 are green)
On fetch failure of an absolute `/@fs/...` asset URL, retry ONCE with basename-beside-manifest
(and one declared subdir level, e.g. `body_mesh_index/<basename>` for the mesh index) resolved
against the manifest's own directory. On success, a LOUD visible notice chip: "N assets resolved
manifest-relative — original absolute paths unreachable (VM-written manifest)". Never silent, never
substitutes different content (same basenames only), never fabricates. Unit tests with mocked
fetch. If risk feels high near the end of your run, SKIP and say so honestly in the report.

### T11 (F12, cheap) — vocabulary + a11y consistency
"P19"→"Player 19" consistency in the follow dropdown; playback-rate buttons get stable accessible
names; FPS badge keeps working.

## Evidence to read first
- runs/lanes/webux3_audit_20260716/FRICTION_AUDIT_20260716.md (+ shots/, both result JSONs)
- runs/lanes/webux3_audit_20260716/manifest_fresh_wolv_local.json (rich real bundle)
- runs/lanes/trk_flip_20260713/default_production/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json (degraded real bundle; READ-ONLY)
- web/replay/src/App.tsx, replayScene.ts, viewerData.ts, viewState.ts, styles.css and their tests
- git log --oneline -8 -- web/replay (waves fixv/webux2/webux2b — do not regress their landings)

## Mandatory structured report (schema at docs/racketsport/lane_report.schema.json → report.json)
- objective_result: per-task T1..T11 PASS/FAIL/SKIPPED with one-line evidence each
- full_suite: vitest passed/failed counts + typecheck EXIT + build EXIT (real exit codes; failed>0
  while claiming PASS = auto-reject unless proven pre-existing)
- HONEST ISSUES: anything weakened, skipped, uncertain, or needing manager browser attention
- artifacts: changed-file list; test names added; anything under your lane dir
- BEST-STACK DELTA: (c) none expected — viewer-only lane, no model/policy change; state it.
- A dated one-paragraph summary bullet for the ledger.
