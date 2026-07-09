# LANE w7_p62stats_20260709 — P6-2 minimal stats v0, BODY+COURT-ONLY (post-hoc consumer)

## HARD RULES
No branches, no commits. .venv/bin/python; MPLBACKEND=Agg. Do NOT edit: scripts/racketsport/process_video.py, configs/racketsport/best_stack.json, threed/racketsport/schemas/__init__.py (tierprov lane owns them RIGHT NOW), gate_check_body_decode.py. Protected clips: Burlington/Wolverine internal use OK; Outdoor/Indoor artifacts read-only, labels NEVER. Artifacts under runs/lanes/w7_p62stats_20260709/ only.

## OBJECTIVE (wave-7 queue #7; refresh ruling: BODY+COURT-only stats move FIRST, ball/paddle stats wait for trusted P1/P3)
Build P6-2 minimal match stats v0 as a POST-HOC CONSUMER of banked pipeline artifacts — no pipeline-stage insertion this round:
1. NEW module threed/racketsport/match_stats.py + CLI scripts/racketsport/compute_match_stats.py: consumes an existing run dir (placement tracks / skeleton3d / court map artifacts — read the artifact schemas from the repo, do not guess) and emits match_stats.json: per-player distance covered, movement speed distribution (p50/p95), court-coverage heatmap grid (metric court coordinates), time-in-zone splits (kitchen/baseline/transition), left-right court balance. EVERY stat carries the trust band inherited from its source artifacts (banded, never silently fake) and a coverage fraction (frames contributing / total).
2. EXPLICITLY EXCLUDED (report must state): anything ball- or paddle-derived (shot counts, rally stats, contact stats) — those wait for trusted P1/P3 per the refresh. Do not compute them even as "preview".
3. Prove it on 2+ banked internal-val run dirs (e.g. under runs/lanes/w4_freshproof_20260707/ — wolverine + burlington; read-only). Sanity-assert: distances are plausible for the clip duration (no teleporting; flag frames with world jumps), zones sum to ~coverage.
4. First check runs/ + docs for the stream-4 P6-1 shot-rules DESIGN doc (grep "shot" under runs/lanes/*stream4* / boards) and align field naming/zone definitions with it so P6-1 can build on this — cite what you found.
5. Direct-CLI reference test + scaffold registration + unit tests (synthetic fixtures with known distances/zones). Any future process_video integration = PROPOSED inline diff in the report only (integration lane follows once file fences free).

## SELF-VERIFICATION
MPLBACKEND=Agg: your new tests + scaffold index + storage audit; fix what you introduce; prove pre-existing at HEAD.

## REPORT
Self-write runs/lanes/w7_p62stats_20260709/report.json (lane_report.schema.json structure): acceptance rows 1-5 (with the 2-clip measured stats quoted), changes file:line, full_suite honest, BEST-STACK DELTA (consumer CLI, no default knob expected — if you add ANY configurable default, it must route via best_stack and you must say so), honest_issues, next.
