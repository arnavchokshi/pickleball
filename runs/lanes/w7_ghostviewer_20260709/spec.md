# LANE w7_ghostviewer_20260709 — viewer half of the GHOST-MESH owner ruling (2026-07-09)

## HARD RULES
No branches, no commits. .venv/bin/python for any py tests; web tests per web/replay conventions. Do NOT edit: pipeline files (scripts/racketsport/**, threed/racketsport/** — the tier-provenance PIPELINE half is a separate later lane; paddle lane owns process_video.py), server/ auth (browser-bypass lane). Owned: web/replay viewer source (mesh rendering/styling + its tests) ONLY. Artifacts under runs/lanes/w7_ghostviewer_20260709/ only.

## OBJECTIVE (owner ruling, binding)
human_review-tier frames get meshes with GHOST/ESTIMATED styling — never hidden, never solid. Implement the VIEWER half:
1. Read the current body mesh index + replay_viewer_manifest schema (threed/racketsport/replay_viewer_manifest.py + threed/racketsport/trust_band.py are the source of truth for tier vocabulary — read, don't edit) and identify how per-frame tier/band provenance reaches (or will reach) the viewer.
2. CONTRACT (manager-pinned): the viewer keys ghost styling off a per-mesh-frame tier field carrying the trust-band vocabulary; when the field is ABSENT the viewer behaves exactly as today (fail-safe — the pipeline half lands later). If the existing manifest already carries a usable band/tier field, USE IT and say so; if not, implement against the documented field name you derive from trust_band.py vocabulary and record the exact expected key in your report so the pipeline lane implements the same key.
3. Ghost styling: translucent + visually distinct "estimated" treatment consistent with existing viewer trust-band affordances (read how existing bands are styled first; blend in, don't invent a new design language). Never hidden, never indistinguishable-from-solid.
4. Tests: unit tests for the tier->material mapping incl. absent-field fail-safe; keep the existing viewer test suite green.

## SANDBOX HONESTY
No localhost binds/browser in sandbox — the zoomed live browser check is the MANAGER's post-land step (headless far-camera checks miss body-scale issues — on record). State the split honestly.

## SELF-VERIFICATION
Run the web/replay test suite (whatever the package defines: npm test / vitest) + any py tests touching replay manifest reading you may add. Fix what you break; pre-existing failures proven at HEAD.

## REPORT
Self-write runs/lanes/w7_ghostviewer_20260709/report.json (lane_report.schema.json structure): acceptance rows 1-4 (incl. the exact field key contract for the pipeline half), changes file:line, suite numbers, BEST-STACK DELTA (expected none — display styling; any new default knob must route via best_stack), honest_issues, next.
