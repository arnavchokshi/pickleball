# LANE w6r_northstar_refresh_20260709 — NORTH STAR REFRESH + docs truth-up (owner directive 2026-07-08; manager rulings binding)

## HARD RULES
- NO branches/commits/pushes; no BUILD_CHECKLIST/board edits (bullet text in report). Protected clips EVAL-ONLY.
- .venv/bin/python; run the doc-consistency/truthful-capabilities/allowlist guardrail test files at the end (they are your blast radius); MPLBACKEND=Agg if any wide run needed.
- Artifacts under runs/lanes/w6r_northstar_refresh_20260709/ only. NO new root .md files (edit existing docs only).
- THE ANCHOR (owner's words, keep it verbatim in NORTH_STAR Part I): the most accurate and fast processor from pickleball video -> 3D world, to give players feedback on how to improve.
- STYLE CONTRACT: NORTH_STAR stays VERY detailed — you REPRIORITIZE and ADD/REMOVE per the rulings; you never summarize away task-level detail. Additions carry evidence citations (paths or arXiv ids + dates). LATER CORRECTS EARLIER: use dated notes where the historical record matters; rewrite in place where the audit ruled text stale-wrong.
- CONCURRENT SESSION: a court-focused session (CALV1) edits court sections + boards. git fetch first; make your edits merge-friendly (surgical, anchored); if a court section changed under you mid-lane, re-read before editing that section.

## FILE OWNERSHIP (exclusive)
- OWNED: NORTH_STAR_ROADMAP.md, CAPABILITIES.md, TECH_BLUEPRINTS.md (dated notes only), runs/manager/wave7_boot_prompt.md (sync pass at the end).
- READ-ONLY EVIDENCE: runs/research_w6refresh_20260709/{RULINGS.md, internal_audit_synthesis.md, internal_audit_full.json, sota_ball_synthesis.md, sota_body_synthesis.md, sota_ball_full.json, sota_body_full.json}; BUILD_CHECKLIST.md (last ~30 bullets); runs/lanes/w6_*/report.json as cited.
- DO NOT TOUCH: process_video.py or ANY code file, web/replay/**, ios/**, cvat_upload/**, configs/racketsport/best_stack.json.

## OBJECTIVE
Apply the manager's RULINGS (runs/research_w6refresh_20260709/RULINGS.md — R1..R7, BINDING; read them + the three synthesis docs FIRST) to the docs of record so that: everything DONE/current-state reads clear, concise, and true; everything FUTURE is reprioritized per the rulings with useful additions and honest removals; the plan re-aligns to the anchor.

## THE WORK (all items; R-numbers refer to RULINGS.md)
1. **NORTH_STAR Part III/VI reprioritization (R1,R2,R5,R6):**
   - P2-2 task block: replace the open-ended decode root-cause framing with the R1 CHECKLIST lane definition (the 5 enumerated audit steps + synthetic round-trip gate + the ceiling rule + the locked-identity/latent-smoothing workaround path w/ arXiv:2512.21573 citation + gate recalibration note).
   - Labeling tasks (P0-4/P1 series): encode R2 — checkpoint evals at 1k/3k/6k/10k w/ decision gates, uniform-random audit stratum, seen-vs-unseen ledger split, occlusion-augmentation recipe item, the 486-row anomaly re-run prereq.
   - Wave-7 section (VI.5): resequence per R6 — P3-1 paddle wiring IMMEDIATE; browser-verify dev-bypass small lane; keep P6 items; add the P5-1 clean-room gate-scoring prereq before speed claims (R5a) + the masklet-conditioning eval candidate (R5b).
   - NEW first-class tasks (R6.4): security/PII/secrets review task (pre-launch gate, owner-visible); training-data licensing check task (GPL PnLCalib, Roboflow ToS vs monetization); input-quality guardrail (elevate P5-5b w/ the product rationale).
   - NEW standing structures (R6.5/6.6): a consolidated OWNER-TIME QUEUE section (single ranked list, refreshed each wave boot) and a fleet-spend-vs-ask table requirement in the wave-boot ritual (PART VI VI.0 step add).
   - Court (R6.3): do NOT resequence their sections yourself beyond adding the manager's dated note that P4-0 is ruled ahead of a 3rd auto-find retrain pending owner re-confirmation w/ the 244.3/212.6px evidence — their session owns the section details.
   - REMOVALS: anything the audit ruled stale-wrong (R7 list) is corrected in place; the old flat 10-20k label bar text is superseded by the gated version; Magnus/spin future work notes point at TT3D bounce-kink + hardware evidence (R3) instead of solver tuning.
2. **Docs truth-up (R7, all 8 items):** CAPABILITIES ball + body rows; NORTH_STAR P0-4 corpus count; P4-1 landed-status fix; P4-2/P4-3 honest numbers; PHASE-5 speed header softened to gate-unscored; PHASE-7 Swift count 140; TECH_BLUEPRINTS paddle note. Each edit cites its evidence path.
3. **Research banking note:** add ONE dated pointer in NORTH_STAR (research index section or PART A evidence list) to runs/research_w6refresh_20260709/ as the 2026-07-09 refresh evidence base.
4. **wave7_boot_prompt.md sync:** update its queue to match the refreshed NORTH_STAR (P3-1 inserted; R1 checklist framing for the decode lane; R2 gates in the labeling line; the new first-class tasks listed; note that the refresh LANDED so the boot prompt's "subject to refresh" header is resolved).

## ACCEPTANCE
1. Every R1-R7 ruling is traceably applied (report a table: ruling -> doc -> section -> edit summary).
2. All 8 R7 stale items fixed with citations.
3. NORTH_STAR still reads as THE master plan: PART 0 untouched, I.7 critical path intact (updated where rulings moved it), Part III task-level detail preserved or increased, PART VI wave plan coherent with wave-7 boot prompt.
4. Doc-consistency/truthful-capabilities/allowlist guardrail tests green; no new root .md.
5. Anchor sentence present verbatim; VERIFIED=0 discipline language unchanged everywhere.

## REPORT (schema-enforced): objective_result; full_suite (guardrail runs); CHANGES (doc:section, one line each); ruling->edit table; HONEST ISSUES (esp. anything a ruling asked for that conflicts with existing doc structure — propose, don't force); proposed BUILD_CHECKLIST bullet; NEXT.
