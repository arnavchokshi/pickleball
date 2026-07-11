# LANE plan_nextmoves_20260710 — Codex ultra-thinking planning consult (READ-ONLY)

You are GPT-5.6 at maximum reasoning, acting as co-planner with the Fable manager for a 10+ hour
autonomous window (owner away; owner directive: Fable + Codex jointly decide next moves).

## HARD RULES
Modify ZERO repo files; write ONLY under runs/lanes/plan_nextmoves_20260710/. No commits.
Protected Outdoor/Indoor labels off-limits.

## CONTEXT (read in this order)
1. runs/research_deepreview_20260710/RULINGS.md (today's dual-track audit + V/P/T/B/M plan; V+P LANDED)
2. NORTH_STAR_ROADMAP.md §2, §4, §5 (queue + gates)
3. runs/manager/inflight_lanes.md (live fences: court wave owns court files; labeling session owns
   import/ingest; fixv+fixp landed c6ffa05d5/0bdabbe67)
4. runs/lanes/ns015_statuspack_20260710/handoff.md (runner/native deferred hunks)
5. runs/lanes/dr_pipeline_20260710/FINDINGS.md ranked defects

## CONSTRAINTS
- GPUs: H100 spot default, <=$5/hr, manager caps own usage at 3 concurrent (court session may hold 1).
- Owner-blocked (CANNOT do): physical device capture/upload proof (NS-01.2b), gold capture (NS-02.1/2),
  iPhone FPS trace, pb.vision clip identity, labeling continuation.
- Already dispatched this tranche (do not re-plan, but may critique): reid asset restore (Sonnet),
  ball candidates code lane (UKF + wire RANSAC/blur flags + offline scoring), spine lane
  (ns015 runner hunks + NS-01.7 slice: post-BODY refined events + audio path + dependency hashes),
  Mac real-GPU browser FPS measure.
- Codex implementation lanes run gpt-5.6-sol xhigh this window (owner directive).

## QUESTIONS (answer ALL, ranked, with reasoning grounded in the read docs)
1. Tranche-2 GPU allocation across: (a) TRK scoring sweep (association profiles w/ margins vs
   loose-pool, frozen benchmark_person_trackers scorer, labeled clips), (b) ball candidate E2E
   attestation + scoring, (c) anything you rank higher. Sequencing + kill criteria per run.
2. iOS replay fixture-hardcode fix (wire real manifest loading; dr audit found it never loads real
   output): do it NOW in this autonomous window (owner away from Xcode) or defer? Weigh: owner's
   active surface, swift test-ability without device, NS-01.2b dependency.
3. NS-01.4 typed-coordinates adoption remainder (paddle estimator + other ad-hoc transforms):
   this window or after TRK/ball results?
4. TT3D joint-anchor-search build (the ball centerpiece): start now in parallel with UKF scoring,
   or gate on UKF/wiring-flip results? It has a pre-registered kill (fallback <5/11 on wolverine).
5. What is the single biggest risk or blind spot in this plan for an unattended 10h window?
6. Anything in NORTH_STAR §5 queue or the RULINGS defect ledger that is executable-now and
   HIGHER value than the above (name it + why).

## DELIVERABLE
runs/lanes/plan_nextmoves_20260710/PLAN.md: ranked tranche-2 plan (lane name, objective,
acceptance/kill, files fence, GPU y/n, est duration) + explicit AGREE/DISAGREE list vs the
dispatched tranche-1. report.json via the output schema; BEST-STACK DELTA (c) none.
