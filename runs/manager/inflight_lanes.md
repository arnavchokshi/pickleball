# In-flight lanes (write at session end, read at session start — FABLE_OPERATING_MANUAL §14 step 9)

One row per still-running lane so the next session neither double-dispatches nor loses a resume.

| lane | kind (codex/sonnet/workflow) | session/task id | resume command | owned files | vm (if any) | expected done | dispatched |
|---|---|---|---|---|---|---|---|
| p63_reference_ranges_20260707 | codex | bg task (Fable succession session) | codex exec resume <session_id from report.json> | ONLY NEW FILES: docs/racketsport/reference_ranges_{schema,v0}.json, scripts/racketsport/validate_reference_ranges.py, tests/racketsport/test_reference_ranges.py (+scaffold-index line) | none (no GPU) | ~1-2h from 2026-07-07 dispatch | 2026-07-07 by Fable final session |
| live_offline_docs_20260707 | codex | bg task (live-vs-offline session 2026-07-07) | codex exec resume <session_id from report.json> | CAPABILITIES.md, TIER_MAP.md, NORTH_STAR_ROADMAP.md, EDGE_PLAYBOOK.md, MASTER_PLAN.md, ios/README.md, tests/racketsport/test_truthful_capabilities.py, BUILD_CHECKLIST.md (append) | none | ~1h from 2026-07-07 dispatch | 2026-07-07 live-tier manager session |
| live_tier_blueprint_20260707 | codex | detached nohup (live-tier session 2026-07-07) | codex exec resume <session_id from report.json> | TECH_BLUEPRINTS.md (additive pillar) | none | ~1h | 2026-07-07 live-tier manager session |
| runbook_doctor_20260707 | codex | detached nohup (live-tier session 2026-07-07) | codex exec resume <session_id from report.json> | RUNBOOK.md, NEW scripts/racketsport/doctor.py, NEW tests/racketsport/test_doctor.py | none | ~1-2h | 2026-07-07 live-tier manager session |
