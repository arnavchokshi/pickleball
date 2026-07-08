# LANE w6_meshcap_20260708 — mesh-cap byte-budget policy + budget CLI passthrough (OWNER CRITIQUE #1 fix, per W6 PLAYBACK RULING)

## HARD RULES (binding)
- NO git branches, NO commits, NO pushes. Working-tree changes only in your OWNED FILES. Manager commits at checkpoints.
- Do NOT edit BUILD_CHECKLIST.md or runs/manager/ boards — proposed bullet text goes in your report.
- Protected eval clips EVAL-ONLY; no Outdoor/Indoor labels ever.
- Honest reporting; .venv/bin/python; MPLBACKEND=Agg; new CLI flags ship scaffold-index/direct-CLI reference tests same-lane.
- Artifacts under runs/lanes/w6_meshcap_20260708/ ONLY. Other lanes' run dirs READ-ONLY.
- NO GPU dispatch; the fresh-world proof rides a later manager GPU errand.

## FILE OWNERSHIP (exclusive)
- OWNED: scripts/racketsport/process_video.py, scripts/racketsport/remote_body_dispatch.py (both freed by w6_gate1b_knob DONE @ dd5e5980d), the mesh frame-selection module (threed/racketsport/frame_rating.py or wherever the plan builder lives — locate it), + their test files.
- DO NOT TOUCH: threed/racketsport/ball_arc_solver.py + its tests (w6_magnus STILL RUNNING), threed/racketsport/body_postchain.py semantics (you may read; do not change bypass behavior), court/calibration files incl. threed/racketsport/court_calibration.py, net_plane.py, profile_registry.py, scripts/racketsport/calibrate.py (ANOTHER SESSION's CALV1 lanes own them right now), CAPABILITIES.md, cvat_upload/**, web/replay/**, ios/**.

## CONTEXT (measured, from runs/lanes/w6_playbackdiag_20260708/playback_decision_table.md — read it first)
All 4 close-proof worlds selected exactly 100 mesh source frames (DEFAULT_TARGET_MESH_FRAME_BUDGET=100, process_video.py:180): effective mesh fps 9.99/10.00/5.21/10.10, worst gap 110 frames (1.83s). Measured index sizes at 100 frames: 13.57/21.31/13.68/12.26 MiB actual. Linear per-frame extrapolation: Wolverine ALL-scheduled(266f)=77MiB, IMG1605 ALL(243f)=27MiB, Burlington ALL(675f)=298MiB, Outdoor ALL(1251f)=1871MiB. So short clips can ship ALL scheduled frames cheaply; long clips must fill a byte budget. --mesh-coverage-mode is CLI-exposed; target_mesh_frame_budget is options-only (NO CLI flag) and NOT threaded through remote_body_dispatch (frame_compute_plan.json ships in the artifact lists — verify whether the VM-side selection actually honors the shipped plan or recomputes).

## THE DESIGN (pinned WHAT)
1. CLI passthrough: add --target-mesh-frame-budget to process_video.py; add whatever remote_body_dispatch threading is needed so a remote BODY dispatch honors it end-to-end (follow the exact pattern w6_gate1b_knob used for body_postchain knobs — grep its landed code at dd5e5980d).
2. BYTE-BUDGET policy (opt-in, new flag, e.g. --mesh-byte-budget-mib N): mesh frame selection fills a per-clip byte budget instead of a fixed frame count — estimate bytes/frame from the same quantities the index builder uses (persons per frame x measured per-player-frame bytes; a conservative estimator is fine if it is measured-calibrated and reported per-run in the plan summary); if ALL scheduled frames fit the budget, select all. Selection ORDER stays the existing coverage logic (ball_aware priority) — the budget changes HOW MANY, never WHICH-first.
3. Defaults UNCHANGED: with no new flags, the deterministic CPU fixture must produce byte-identical artifacts to HEAD (dd5e5980d) — same proof style as gate1b acceptance #2.
4. Plan summary must record: policy used, budget, estimated vs selected bytes, selected frame count — so any world can be audited for which policy produced it.

## ACCEPTANCE
1. Flags exist end-to-end (grep proof CLI -> options -> plan builder -> remote dispatch serialization); scaffold/direct-CLI reference tests green.
2. No-flag deterministic fixture: byte-identical to HEAD dd5e5980d.
3. Unit tests: (a) byte-budget selects ALL frames when everything fits; (b) truncates to budget with priority order preserved when it does not; (c) plan summary audit fields present.
4. Focused blast-radius suites green (the test files of your owned modules + scaffold/doc guardrails). FULL wide suite NOT required this lane (tree is concurrently dirty; wave-close adjudication is the final word) — run it only if the tree quiets; say which you did.
5. Report the exact GPU proof command for a later errand: wolverine + outdoor with a sensible default byte budget (pick from the measured table; state the predicted index sizes + expected effective mesh fps).

## KILL / STOP
- If VM-side selection RECOMPUTES the plan (shipped frame_compute_plan.json not honored), implement the minimal honor-the-shipped-plan fix ONLY if it is contained in your owned files; otherwise report the dependency as HONEST ISSUES + the proposed design — do not touch unowned files.

## REPORT (schema-enforced)
objective_result; full_suite line (scoped, per acceptance 4); CHANGES file:line; predicted-size table; GPU proof command; HONEST ISSUES; proposed BUILD_CHECKLIST bullet; NEXT.
