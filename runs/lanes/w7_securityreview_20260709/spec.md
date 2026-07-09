# LANE w7_securityreview_20260709 — P7-4c security/PII/secrets PRE-LAUNCH gate review (READ-ONLY)

## HARD RULES
STRICTLY READ-ONLY on the repo: you edit NOTHING outside runs/lanes/w7_securityreview_20260709/. No commits. This is a defensive review of our own product surface before any flags flip live. Concurrent lanes have dirty files in the tree (process_video.py, best_stack.json, mhr_decode.py) — review committed HEAD state where it matters and note tree-vs-HEAD divergence only if security-relevant.

## OBJECTIVE (NORTH_STAR R6.4 first-class gate; report-only — fixes become later lanes)
Produce the P7-4c pre-launch security review with findings ranked CRITICAL/HIGH/MEDIUM/LOW, each with file:line evidence:
1. SECRETS: scan for committed credentials/tokens/keys (git-tracked files only; check data/credentials/ handling stays gitignored+chmod600 as documented; .claude/settings.json permission grants; any hardcoded API keys/URLs w/ embedded tokens; runs/ force-added artifacts leaking tokens).
2. PII/BIOMETRICS: map where non-owner player video/biometric data flows and persists (server upload path, S3/Mongo INFRA plan in the product-infra docs, ReID/profile machinery). Tie findings to the PART 0 biometric-consent item (session-only tracking of non-owner people is the standing default — flag anything violating it).
3. AUTHZ: product server auth surface (server/routes/auth.py, security.py, render_app.py wall) — token handling, presigned-URL scoping, job/clip authorization per-account isolation. Include the NEW dev-auth bypass (web/replay/src/devAuthBypass.ts + verify_process_video_viewer wiring, landed this wave): adversarially re-check its fail-closed claims (exact-flag/loopback/non-prod/manifest-only) — can any prod build/config combination expose it?
4. INJECTION/EXEC: user-controllable inputs reaching subprocess/shell/ffmpeg/path joins in the upload->process->serve chain.
5. DEPENDENCY QUICK-PASS: obviously risky pinned deps (no network — from lockfiles/requirements only; mark version-check items needs-network).
Deliverable: runs/lanes/w7_securityreview_20260709/SECURITY_REVIEW.md (the ranked findings + a go/no-go pre-launch checklist) + report.json.

## REPORT
Self-write runs/lanes/w7_securityreview_20260709/report.json (lane_report.schema.json structure): acceptance = one row per area 1-5 (finding counts by severity), full_suite = N/A-read-only (state it), BEST-STACK DELTA none, honest_issues (what you could NOT assess offline), next.
