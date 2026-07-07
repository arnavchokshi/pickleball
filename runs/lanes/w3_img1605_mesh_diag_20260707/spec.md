# LANE w3_img1605_mesh_diag_20260707 — img1605 ZERO mesh frames: root cause (wave-3 #7; diagnosis ONLY)

## HARD RULES
- You are Codex lane `w3_img1605_mesh_diag_20260707`. Work ONLY inside /Users/arnavchokshi/Desktop/pickleball.
- DIAGNOSIS LANE: you own NO repo files; edit nothing outside runs/lanes/w3_img1605_mesh_diag_20260707/. No git commit/branch/push.
- Protected data: 4 eval clips EVAL-ONLY (artifacts fine; Outdoor/Indoor labels never). Held-out pwxNwFfYQlQ / vQhtz8l6VqU nowhere.
- Honest reporting; every number carries artifact path + repro snippet. Always `.venv/bin/python`.
- Read first: BUILD_CHECKLIST.md last ~15 bullets.

## CONTEXT
- img1605 (the handheld eval clip) produces ZERO mesh frames in fresh worlds — pre-existing across waves (flagged in the wave-2 closing run; NOT a wave-2 regression). Other clips produce mesh frames; burlington's browser verify separately shows a "missing mesh vertices" HUD notice (471/600 measured) — possibly related, possibly not.
- Mesh scheduling is contact-dense (`ball_aware` mode): mesh frames are selected around ball-contact/rally activity. img1605's BALL chain performance on handheld footage has historically been the weakest.
- A viewer-consumable ~30MB mesh index + membership/rollup/mesh-index tools exist (built 2026-07-04).

## OBJECTIVE
1. Trace the mesh path for img1605 end-to-end with values at each hop: scheduler inputs (contacts/rally segments found?) → frame selection (how many requested?) → GPU mesh production (produced?) → mesh index/rollup (indexed?) → viewer manifest (referenced?). Name the FIRST hop where it drops to zero, with the exact code path (file:line) and input values that caused it.
2. Manager's candidate theory to confirm/refute: `ball_aware` scheduling finds no trusted contacts on img1605 → selects 0 frames → nothing downstream. If confirmed, check: is there a designed fallback (e.g., uniform-stride mesh sampling when contact-dense finds nothing) that failed to engage, or does no fallback exist?
3. Is burlington's "missing mesh vertices" notice the same mechanism (partial variant) or distinct? One paragraph + numbers.
4. FIX PROPOSAL (≤1 page, no code changes): minimal source-level fix (e.g., fallback scheduling policy), expected mesh-frame counts per clip after fix, the test that pins it, and whether proof needs a GPU run (mesh production is GPU-side — if yes, the wave-end fresh-worlds run is the proof vehicle; specify what to assert).

## ACCEPTANCE
First-zero hop named with file:line + input values; theory explicitly confirmed/refuted; fallback existence answered; fix proposal with per-clip expected counts + assertion list. No code changes anywhere.

## EVIDENCE
- runs/lanes/wave2_freshworlds_20260707/ (img1605 world: PIPELINE_SUMMARY, scheduling artifacts, mesh index; burlington equivalents for contrast)
- runs/manager/wave2_browser_verify/ (the HUD notices)
- Code READ-ONLY: mesh scheduling (`ball_aware`), mesh index/rollup tools, viewer manifest builder.

## STRUCTURED REPORT
objective_result PASS/PARTIAL/BLOCKED; the 4 answers; artifact index; HONEST ISSUES; NEXT; state "no code changed" explicitly.
