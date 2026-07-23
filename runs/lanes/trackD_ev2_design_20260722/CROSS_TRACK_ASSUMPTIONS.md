# Cross-track assumptions binding on E-v2 (Track D side of the record)

1. st0epgnab7dr COURT-TRAINING EXCLUSION (Track A, 2026-07-22): Track A excluded video
   st0epgnab7dr from its court-training promotion set specifically because it is E-v2's Stage-P
   source-disjoint teacher holdout — checkpoint AND decode-threshold selection run on its
   validation batches. Under cascade fusion (everything-helps-everything), court-training on that
   video would make court-derived inputs unrepresentatively strong exactly where E-v2 selects,
   biasing selection. Track A's side: runs/lanes/court_owner_pack_20260722/results/PROMOTION_RECORD.md.
   OBLIGATION: when Track D retires this holdout (E-v2 concludes, or a re-registration drops it),
   notify main so st0epgnab7dr rejoins court training automatically (owner already approved it).
2. This assumption must appear in REGISTRATION.md's assumptions when the repair round lands
   (REGISTRATION.md is under active repair-round edit; adding it there is a required
   re-review checklist item, recorded in STATUS.md).
