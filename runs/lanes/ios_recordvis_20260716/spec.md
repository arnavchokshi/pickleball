# LANE ios_recordvis_20260716 — blocked states impossible to miss on the Record tab (Track D wave 2)

Ground truth READ FIRST: NORTH_STAR_ROADMAP.md §1.2 step 2 + §1.3 (visual direction, motion layer,
accents policy) + §6; ios/README.md (Brand V4 record control, motion layer, reduced-motion rules);
runs/lanes/ios_recordpath_20260715/DEVICE_EVIDENCE.md (REAL-DEVICE TRUTH — read every line);
runs/lanes/ios_recordpath_20260715/GUARD_AUDIT.md + report.json (wave-1 loud-state contract you
must not regress); commit 7d1b19232 (wave-1 fix, already on HEAD).

## CONTEXT (2026-07-16 01:0x)
Wave 1 fixed the silent-death mechanisms; on the owner's real iPhone 14 Pro the fixed build reaches
blocked("Rotate to landscape to record") in 1.4 s with camera/mic authorized — the phone was in
portrait on a desk. The owner still perceived "tap does nothing": the OWNER-VISIBLE failure is
SALIENCE. A blocked Record tab must be impossible to misread, pre-tap and per-tap. Device access is
best-effort overnight (phone may lock) — you have NO device dependency; simulator + tests only.

## MANDATORY SKILL (owner requirement — non-negotiable)
Load and follow the Codex-installed iOS skill **`ios-debugger-agent`** (plugin
`build-ios-apps@openai-curated`,
/Users/arnavchokshi/.codex/plugins/cache/openai-curated/build-ios-apps/2f1a8948/skills/ios-debugger-agent/SKILL.md).
If its XcodeBuildMCP tools are unavailable in-session, read the SKILL.md and follow its workflow
via xcodebuild/simctl; sandbox-blocked steps go to MANAGER_VERIFY2.md exactly like wave 1. Report
must state how the skill was used. `swiftui-ui-patterns` from the same plugin MAY be loaded for the
UI work.

## HARD RULES
- NO branches/commits/git add. Manager commits after ruling. Preserve all unrelated dirty files and
  untracked owner dirs. Base = current HEAD (7d1b19232 or later).
- FILE OWNERSHIP: ios/App/**, ios/Capture/**, ios/AppTests/**, runs/lanes/ios_recordvis_20260716/**.
  FORBIDDEN: everything else (incl. ios/Upload/**, ios/Replay/**, project structure, threed/,
  scripts/, server/, configs/, other lane dirs). Out-of-fence needs = inline hunks in report only.
- Honest reporting; VERIFIED=0; sim green is not device proof; no promotion language.
- Wide blast radius = full SwiftPM suite + full hosted AppTests (wave-1 pattern; /tmp caches,
  --disable-sandbox). The 7 wave-1 C1-C6/watchdog regressions and the guard audit MUST stay green
  and silent-free.

## MISSION — make every blocked/limbo state loud, §1.3-faithful
1. **Persistent pre-tap blocker surface.** When the Record tab's capture state is blocked (any
   reason) — and specifically portrait-vs-landscape — show a persistent, unmissable prompt on the
   Record screen itself (not only the post-tap banner): e.g. a rotate-to-landscape card/overlay in
   brand ink-on-cream with a rotate glyph, visible from cold launch, disappearing the moment the
   blocker clears. Accent rules: this is an empty/guidance state, accents allowed; keep measured
   surfaces clean; at most one accent cluster.
2. **Visible reaction on EVERY tap.** Tapping the record control must ALWAYS produce immediate
   visible feedback within one frame-ish latency: if startable → existing record flow; if
   blocked/preparing → button press-depress + brief wobble/shake (reuse the §1.3 motion layer) AND
   the blocker surface/banner pulses or flashes to draw the eye + warning haptic. Reduced-motion:
   no wobble/pulse — use a static highlight/emphasis change that is still clearly visible.
   NO tap may terminate without visible consequence — extend the wave-1 guard audit with a
   "per-tap visible reaction" column, zero silent rows.
3. **Kill the disabled-dead-zone class permanently.** Replace `.disabled(!canRecordFromTab)` on the
   record button with an always-hittable pattern: taps during bounded preparation give visible
   "Setting up camera…" feedback (state already exists) instead of being swallowed by SwiftUI.
   Preserve: no double-start (wave-1 coalescing), XCUITest hittability semantics
   (DinkVisionRecordButton stays hittable in ALL states — add a regression test asserting the
   accessibility element exists+enabled in idle/requestingAccess/blocked/ready/recording).
4. **Banner prominence + accessibility.** blockedReason banner: high-contrast per §1.3 tokens,
   safe-area aware both orientations, VoiceOver announcement on every blocked-state entry
   (UIAccessibility notification; test the announcement plumbing at model level), Dynamic Type safe.
5. **Failing-first tests** for each of 1-4 (author before implementation; if red runtime is
   sandbox-blocked, document as wave 1 did). Hosted AppTests + SwiftPM.

## ACCEPTANCE (real exit codes, verbatim, no piped masking)
1. Full SwiftPM suite 0 failures EXIT 0, count >= wave-1 baseline (245).
2. Hosted AppTests: only allowed failure = pre-existing ANELatencyBenchmark device-only test; all
   wave-1 + new tests green (report executed/passed/failed + exit code; sandbox-blocked → exact
   commands in MANAGER_VERIFY2.md).
3. New failing-first tests >= 1 per mission item 1-4, all green post-fix.
4. Updated guard audit table incl. per-tap-visible-reaction column, zero silent rows.
5. Sim live-controller before/after proof plan (or artifacts if runnable): portrait cold launch
   shows the persistent rotate prompt; tap produces visible reaction; banner + Retry still work.
6. Report states reduced-motion + VoiceOver behavior explicitly.
KILL: weakening landscape/capture correctness; breaking wave-1 regressions; accents on measured
surfaces; any tap path without visible consequence; fabricated device claims.

## REPORT
Schema report.json via --output-schema: objective_result vs the 6 items, full_suite real exit
codes, changes file:line, HONEST ISSUES (device visual confirmation still pending), skill-usage
statement, session_id, BEST-STACK DELTA: expected (c) none — state explicitly.

## ANTI-PASSIVE-WAIT
Ending your turn to wait = lane death. Bounded foreground polls only. End with report.json written
or a hard blocker stated in it.
