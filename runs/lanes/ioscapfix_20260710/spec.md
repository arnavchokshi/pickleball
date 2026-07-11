# LANE ioscapfix_20260710 — fix pre-existing xcodebuild break: CaptureSidecarWriter vs CaptureSidecar drift

## HARD RULES
- No branches/commits/git add. FILE OWNERSHIP: ios/Capture/**, ios/Core/** (minimal), their tests +
  golden sidecar fixtures ONLY if the contract truly requires it, runs/lanes/ioscapfix_20260710/**.
  FORBIDDEN: ios/App, ios/Upload, ios/Replay (just landed by another lane), web/, scripts/, threed/.
- The v1 capture sidecar schema contract is authoritative (NS-01.1 landed golden fixtures) — do NOT
  change schema/fixtures to make code compile unless git history proves the fixtures already encode
  the newer fields; prefer aligning the CALLER.

## SYMPTOM (manager-verified at HEAD, clean tree)
`xcodebuild build-for-testing -scheme Pickleball -destination 'generic/platform=iOS Simulator'` fails:
ios/Capture/Sources/PickleballCapture/CaptureSidecarWriter.swift:92:30: error: extra arguments at
positions #25, #26, #27, #28 in call (to PickleballCore.CaptureSidecar.init). ios/Capture + ios/Core
are unmodified in the tree => committed-file drift. Yesterday's session had build-for-testing green;
suspects: commits 0e81c9300 / 737ced8ff touching Capture/Core. SwiftPM `swift test` (245 tests) is
GREEN — the break only manifests in the xcodeproj test build (target membership differences).

## MISSION
1. git-archaeology: identify exactly which commit introduced the drift and which side (writer args vs
   Core init) reflects the intended NS-01.1 contract (read the sidecar schema + golden fixtures).
2. Align minimally; keep schema fixtures green; swift test full suite must stay green (run it).
3. Attempt the xcodebuild command; if the sandbox blocks it (sandbox-exec), say so — the manager
   verifies locally at ruling.
## REPORT
report.json via schema; root-cause commit + exact fix rationale; HONEST ISSUES; BEST-STACK DELTA (c).
