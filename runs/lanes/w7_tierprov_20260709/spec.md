# LANE w7_tierprov_20260709 — PIPELINE half of the GHOST-MESH owner ruling (tier provenance emission)
# (PRE-STAGED at ghost-viewer acceptance; DISPATCH ONLY after paddlewire_p31 frees process_video.py)

## HARD RULES
No branches, no commits. .venv/bin/python; MPLBACKEND=Agg. Protected clips rules stand. Do NOT edit: web/replay/** (viewer half LANDED — read its report for the contract), server/ auth (browser lane), threed/racketsport/mhr_decode*.py (P2-2 lane). Artifacts under runs/lanes/w7_tierprov_20260709/ only.

## OBJECTIVE (owner ruling 2026-07-09, binding)
human_review-tier frames get meshes with ghost/estimated styling — never hidden, never solid. The VIEWER half is landed (runs/lanes/w7_ghostviewer_20260709/report.json — read it first). Implement the PIPELINE half against its PINNED CONTRACT:
1. Body mesh index frames carry optional per-frame `trust_badge` with trust_band.py vocabulary (verified | preview | low_confidence). Emit `trust_badge: "preview"` for human_review-tier mesh frames; emit the appropriate badge for other tiers per the existing trust-band machinery (threed/racketsport/trust_band.py is the vocabulary source — reuse, don't reinvent).
2. MESH ELIGIBILITY: per the ruling, human_review-tier frames are no longer EXCLUDED from mesh emission — they are emitted WITH the ghost badge, within the byte-budget policy (current best_stack default byte-budget-300 stands; the owner's 300-vs-400 ruling is pending and does NOT block this lane). The byte-budget frame-selection must count ghost frames like any other (no silent tier re-weighting without reporting it).
3. Schema: update the mesh-index/replay manifest schema handling so trust_badge round-trips (schemas/__init__.py if needed — remember remote BODY dies with extra_forbidden on schema drift; keep the field optional).
4. Tests: emission tests (human_review->preview badge present; verified frames unchanged), schema round-trip, byte-budget interaction, absent-field back-compat (old artifacts still validate).

## BEST-STACK DELTA (mandatory)
Ghost-mesh emission becomes the DEFAULT path (it implements an owner ruling): flip/extend the relevant best_stack.json entry in THIS lane (the mesh tier-eligibility entry is PENDING on the owner display ruling — implement ghost emission as WIRED_DEFAULT per the ghost ruling, keep the 300-vs-400 cap + any further tier-eligibility raise PENDING on the owner ask). State the delta precisely in the report.

## SELF-VERIFICATION
Full blast-radius: mesh-index/export tests, process_video tests, schema tests, byte-budget tests (MPLBACKEND=Agg wide over tests/racketsport). Fix what you introduce; prove pre-existing at HEAD. Acceptance runs THROUGH scripts/racketsport/process_video.py (no lane-local replica).

## REPORT
Self-write runs/lanes/w7_tierprov_20260709/report.json (lane_report.schema.json structure): acceptance rows 1-4 + BEST-STACK DELTA, changes file:line, full_suite honest, honest_issues, next.
