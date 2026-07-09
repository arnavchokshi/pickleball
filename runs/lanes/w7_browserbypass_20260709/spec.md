# LANE w7_browserbypass_20260709 — browser-verify dev-bypass (stop carrying INFRA-3)

## HARD RULES
No branches, no commits. Protected clips rules stand. .venv/bin/python; MPLBACKEND=Agg. Do NOT edit: scripts/racketsport/process_video.py, configs/racketsport/best_stack.json + their tests (paddle lane), threed/racketsport/mhr_decode*.py (P2-2 lane), web/replay mesh/render/styling modules (ghost-viewer lane owns those). Artifacts under runs/lanes/w7_browserbypass_20260709/ only.

## OBJECTIVE
scripts/racketsport/verify_process_video_viewer.py (the headless-browser closeout instrument; input = replay_viewer_manifest.json, NEVER PIPELINE_SUMMARY.json) has been blocked as a closeout tool by the product sign-in wall (INFRA-3, booked in wave-6 as "browser-verify dev-bypass"). Build a DEV-ONLY bypass so local verification works again, without weakening real auth:
1. LOCATE the sign-in wall (server/ product-infra auth + whatever the replay web app enforces). State what you found as file:line in the report — this spec deliberately does not assert where it is.
2. Implement bypass gated on ALL of: explicit env var (e.g. REPLAY_VERIFY_DEV_BYPASS=1), localhost-only request origin, and non-production mode. Default = OFF, fail-closed. Production configs must be provably unaffected.
3. Wire verify_process_video_viewer.py to use it (flag or env), documented in its --help.
4. Tests: default-off test, localhost-only test, production-mode-refusal test, plus the verifier's own tests still green.

## SANDBOX HONESTY
Your sandbox cannot bind localhost or launch browsers — unit-test the gating logic; the live headless verify is the MANAGER's post-land check. Say exactly this split in the report; do not fake a browser run.

## SELF-VERIFICATION
Full blast-radius: your new tests + tests for the touched server/auth modules + the verifier's tests, wide MPLBACKEND=Agg over the touched dirs. Fix what you break; prove pre-existing failures at HEAD; localhost-bind PermissionError class is known sandbox-preexisting — classify, don't chase.

## REPORT
Self-write runs/lanes/w7_browserbypass_20260709/report.json (lane_report.schema.json structure) with acceptance rows for 1-4, BEST-STACK DELTA (expected: none — verification infra, not a pipeline gain; if you add any default-resolved knob it MUST go through best_stack.json), honest_issues, next.
