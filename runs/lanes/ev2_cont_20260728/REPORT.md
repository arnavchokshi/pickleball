# E-v2 continuation — best event-head result to date; fine-tune rejected

Lane: `ev2_cont_20260728` (orchestrator-executed continuation of
`ev2_train_20260728`). Date: 2026-07-28 (day session).
Status: candidate signal; `VERIFIED=0`; NO anchor ingestion authorized.

## What ran

1. **Pretrain continuation to the full 12k-step design**: resumed (model-only)
   from last night's step-2977 checkpoint (`30b25a8e…`) and trained the
   remaining **9,100 steps** on the same gate-PASSed staged corpus (identical
   ledger pin `f09e62e1…`; quarantines intact), A100, 270-min wall not hit,
   exit 0. Final val (public corpus): best **micro-F1@±2 0.0942**,
   `max_positive_class_probability` **0.888** (prior lineage best: 0.0 / 0.435).
   Checkpoint: `best_event_head_cont9100.pt` sha256 `e11529bc16…`, archived at
   `s3://sway-videos/pickleball-models/20260728/event_head_ev2_cont9100_best.pt`.
2. **Owner fine-tune at spec** (400/400 steps this time, fresh Step-0 gate
   proof, 61-train/41-val eval-only): completed cleanly — and made things
   WORSE (below). The consumed one-touch protected-50 eval was NOT re-run.
3. **Eval battery on BOTH checkpoints**: matched-window 50-clip public sweep,
   owner-41 val, firing-rate grid on the 697 s pb.vision demo.

## The result (public 50-clip sweep, tolerance ±2 frames)

| Checkpoint | HIT @0.5 | HIT @0.1 | BOUNCE @0.1 |
|---|---|---|---|
| **pretrain-cont 9100** | tp 12 / fp 1 — **precision 0.923**, recall 0.141, F1 0.245, 27.8 ms | tp 59 / fp 74 — precision 0.444, recall 0.694, **F1 0.541**, 30.0 ms | tp 11 / fp 114 — F1 0.126 (**first nonzero BOUNCE ever**) |
| after 400-step fine-tune | collapsed (≈0 fires) | collapsed | F1 0.075 |

Program context: the previous all-time best was val F1@±2 **0.3631** (July-16
scaffold, which then diverged). **0.541 on the held 50-clip sweep with 92%
precision at high threshold and ~30 ms mean timing error is the strongest
event-head evidence this program has produced.** The trained-event wall
(§2.2 BALL) is broken on the PUBLIC (tennis/TT) domain.

## Fine-tune verdict: REJECT

The full-length 61-label fine-tune (lr 5e-4, 400 steps) destroyed the public
signal (max positive prob 0.888 → 0.219; HIT F1 0.541 → ≈0) without gaining
owner-val performance (macro-F1 0.0 before and after). Last night's
wall-truncated 100-step stub was accidental early stopping and scored better
(F1 0.323 @0.05 public). 61 labels cannot carry a 400-step fine-tune at this
lr. Keep the PRETRAIN checkpoint as the candidate; fine-tuning waits for
in-domain volume (pb.vision pixels) or a far gentler recipe.

## Firing rate on the pickleball demo (697 s)

Pretrain-cont: **0.000 events/s at every threshold** (0.5→0.05). Fine-tuned:
in-band 0.41/0.81 events/s at 0.06/0.05 — but with ~2% public precision, an
in-band rate is noise, not a pass. Interpretation: the public-domain head does
not transfer zero-shot to pickleball pixels (consistent with §2.3 history —
the 7.16/s zero-shot disaster was the mirror failure). **The binding gap is
now DOMAIN, not architecture or training machinery**: the named lever is the
usage-cleared pb.vision in-domain videos as training pixels
(NORTH_STAR §2.2 DATA/EVENTS; plan `runs/lanes/next_steps_events_ball3d_20260728/PLAN.md`).

## Discipline notes

- No anchors from either checkpoint may be ingested (typed anchors +
  agreement filtering remain prerequisites).
- Owner-41 stays eval-only; protected-50 one-touch not re-consumed.
- Public corpus license posture: RD_ONLY (recorded in the train manifest);
  this checkpoint is a research diagnostic, not a product-stack candidate
  until license review per standing rule 11.
- Cost: ~2.6 h A100 ≈ $5.0 (training 2.3 h + eval batteries).

## Next (in order)

1. Stage pb.vision in-domain pixels (usage-cleared 2026-07-20; hold 2-3 videos
   out) and continue pretraining this checkpoint on them.
2. Re-run the firing-rate grid + owner-41 low-threshold sweep on the
   domain-adapted checkpoint; the 0.3-1.0 events/s band with real precision is
   the gate that matters.
3. Only then: typed anchors → agreement-filtered ingestion → sub-frame bounce
   timing consumers (queue rows 4-6).
