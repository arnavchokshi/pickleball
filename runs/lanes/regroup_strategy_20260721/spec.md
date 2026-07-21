# regroup_strategy_20260721 — THE regroup: exact next steps from the evidence pack (ULTRA)

You are the strategy synthesis the owner demanded after 2 days of intense work produced only one user-visible gain. Their mandate, verbatim spirit: "we have so much data I don't get why we aren't using it... stop and regroup... figure out what our EXACT next steps should be. In the past you've given next steps with high confidence just for nothing to come out of it. We need to change something — more agents, more out-of-the-box thinking, more organization about what we are trying to accomplish."

## READ (in order)
1. runs/regroup_20260721/REGROUP_INPUTS.md — the compiled evidence pack (data inventory with numbers, model-consumption map, gap matrix, owner labels, 3-day effort audit, idea pool).
2. runs/regroup_20260721/gap_matrix.json — the never-queued cells (80,967 Roboflow pickleball images used by NOTHING; 106 real court frames unused while court trained synthetic; pb.vision 12-venue court supervision untouched; etc.).
3. NORTH_STAR_ROADMAP.md §1 (the product), §2.2 (capability truth), §2.3 (standing kills — respect them or justify scoped reopens with the new evidence).
4. runs/lanes/ball_event_abc_20260720/seed_20260720/ state: Arm A (owner-only) val F1 = 0.0; arms B (pb.vision teacher) and C (placebo) training now — their verdict may exist by the time you finish (check; if present, incorporate).
5. The 3-day truth: every honest evaluation failed the same way (venue/domain generalization) — selection layer Indoor all-axes miss, court line detection cross-venue, tennis→pickleball transfer. Competitors won via data ops + capture discipline (runs/research_sota_20260719/).

## PRODUCE: THE EXACT PLAN — and make it different in KIND from past plans:
1. **Data-utilization first**: for each of the 5 trainable components, the exact sequence of training/eval changes using data WE ALREADY OWN (the never-queued cells are the shortlist). Per step: exact asset + exact command-level change + expected measurable effect + the CHECK that would prove/disprove it + effort (hours) + GPU cost. NO step without a check. Respect teacher-quality/holdout discipline (name the quarantines per step); where a standing rule blocks a use (e.g. Roboflow eval-disjointness for person fine-tune; RD_ONLY posture note on pb.vision ball-detector training — note the owner has since granted signed full usage, so propose the ruling update), say exactly what ruling/mitigation unblocks it.
2. **Sequencing with dependencies**: a 5-day plan (day granularity), max parallelization, but each day ends with a MEASURED number or a named negative — no day of pure machinery.
3. **What we STOP doing**: name the activity patterns from the effort audit to cut (e.g. building new selection/algorithm layers before data diversity exists; per your read).
4. **Organization**: propose the standing structure that prevents never-queued cells recurring (e.g. a DATA_LEDGER.md every lane must consult + a utilization check in the lane template).
5. **Honest confidence calibration**: for each proposed step, state the failure probability and what we learn if it fails. The owner is done with overconfident promises — under-claim.
6. The single highest-yield FIRST action for tomorrow morning.

OUTPUT: write runs/regroup_20260721/EXACT_PLAN.md + your final message = the plan's executive summary. VERIFIED=0 discipline; no promotions claimed. Read-only otherwise (you may write ONLY into runs/regroup_20260721/ and runs/lanes/regroup_strategy_20260721/).
