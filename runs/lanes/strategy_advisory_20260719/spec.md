# Lane strategy_advisory_20260719 — ULTRA advisory: fold fresh research into the court + ball programs

You are an ADVISORY lane (owner policy 2026-07-19: next-step decisions = Fable thinking + a
gpt-5.6-sol ULTRA advisory; the manager rules, you advise). You produce STRUCTURED OPTIONS with
evidence, tradeoffs, risks, and sequencing — never a bare conclusion, never code edits.

## READ FIRST (all on disk)
1. NORTH_STAR_ROADMAP.md — esp. §2.2 CAL/BALL/EVENTS rows, §2.3 do-not-repeat list, §5 queue.
2. runs/research_sota_20260719/domainA_court_calibration.md (fresh, 2-vote-refuted).
3. runs/research_sota_20260719/domainB_ball3d_labels.md (fresh, 2-vote-refuted).
4. runs/lanes/static_cal_firstlock_20260717/spec.md (the in-flight CAL lane).
5. runs/lanes/event_head_pretrain_20260716/SCALE_UP_SPEC.md (the in-flight event program).
6. runs/lanes/oneworld_bridge_xref_20260719/XREF_RULING.md (queue-#1 no-reorder ruling).
7. data/event_labels_owner_20260719/PROVENANCE.json (102 owner rows: 60 typed + 42 negatives).
8. runs/HANDOFF_20260717.md for standing kill evidence (esp. why TT3D-class geometry was killed).

## THE FOUR DECISIONS (give 2-3 options each, with: evidence cites, cost/wall estimate, risk,
reversibility, and your ranked recommendation + WHY; flag any conflict with §2.3 standing kills
and say exactly what NEW EVIDENCE justifies any scoped reopen)

D1 COURT PROGRAM. Research ranks gap-closure: (1) classical line-detection robustness (shadow
removal MTMT/ShadowFormer, ROI lookalike-line rejection, PnLCalib line-over-point weighting),
(2) static-span pooling (already in-flight tonight), (3) MORE diverse labels for the learned
finder (ranked THIRD), (4) solver replacement (ranked last; not the bottleneck). Production
systems win via capture-discipline UX (pb.vision CourtFocus guided framing) or extra cameras.
Decide: what NEW lane(s) to queue, whether court-diversity labeling (owner tasks 88-91) gets
re-scoped (calibration lever vs auto-find-gate/eval fuel), and where capture-discipline UX
(iOS guided framing) slots vs NS-03.LIVE.

D2 BALL PROGRAM. Research: trained event head = dominant path for CONTACTS (no geometric plane
for paddles); but a ZERO-LABEL court-plane bounce detector (TT3D/MonoTrack ray-plane pattern)
is evidenced for BOUNCES; MonoTrack DP rally-structure post-processing (max hits, min spacing,
alternation) lifted recall 78.1→94.3 with zero labels; pose-conditioning ablation +7.9pp.
Decide: (a) add a typed BOUNCE anchor class from court-plane geometry (scoped reopen — §2.3
killed geometry-only-as-whole-solution, this is one anchor class under the needs-typed-anchors
taxonomy verdict); (b) add the DP rally-structure post-processor to the event head; (c)
sequencing of wrist-conditioning vs fine-tune vs these. State exact reopen conditions honoring
the 0-violation kill rule and the frozen gate.

D3 PB.VISION GALLERY PSEUDO-LABELS. ~12 videos / 2+h with their trained event predictions are
being harvested now (RD_ONLY, competitor-processed). TT4D precedent: physics-filtered pseudo-
labels at 45,946 games. Standing rule excludes competitor-processed clips from ball-DETECTOR
training. Decide: may their EVENT predictions, agreement-filtered (their-events × our-audio ×
our-wrist-kinematics), enter the event-head FINE-TUNE training set? Options must cover:
contamination of our head-to-head benchmarking (can't score against a competitor we trained
on — name which videos must be quarantined as compare-only), license/ethics posture, and a
control design proving pseudo-label lift vs the owner-102-only fine-tune (the protected 50-row
seed stays the judge, eval-only).

D4 OWNER LABELING BUDGET. 102 rows banked (60 typed + 42 neg). From-scratch low-shot is bleak
(single-digit-to-low-teens F1 at 100 clips) but fine-tune-on-74.5k-pretrain is the evidenced
pattern; only our own protected-seed eval decides. Active learning cuts labels ~3x but needs
more human rounds. Decide: the concrete decision gate (which measured number after the first
fine-tune triggers asking the owner for more labels vs stopping), and whether the remaining
~200 pack rows should be re-targeted by active-learning-style uncertainty ranking IF more
labeling is ever requested.

## OUTPUT (final message = structured report; also write it to
runs/lanes/strategy_advisory_20260719/ADVISORY.md)
Per decision: options table + recommendation + confidence + what-would-change-my-mind. End with
a single proposed §5 queue edit (ordered, minimal) and an owner-asks delta. VERIFIED=0; nothing
you write is a promotion. No file edits outside your lane dir.
