# Holdout evaluation ruling (Fable, 2026-07-21): PREREGISTERED MISS — the honest state of people-cleaning

One-shot preregistered evaluation (ledger row 2026-07-21), single scored run per clip, no tuning:
- Indoor (FRESH judge): MISS all 6 axes (IDF1 0.559 / 4 sw / 395 spectFP / 750 farFP / cov4 0.457 /
  near-miss 0.125). The selection layer does NOT generalize to an unseen crowded venue.
- Outdoor (disclosed historical): MISS 5/6 (spectFP 0 = only pass; IDF1 0.756 / cov4 0.604).
- RF-DETR reproduction: MISS the 0.0001 bar — discrete axes exact, continuous axes drifted UPWARD
  on fresh inference (wolverine cov4 +0.093). Finding: the frozen card is specific to reused
  raw-detection dumps; a reproduction bar over fresh inference needs a float-variance-aware spec.
  Flip remains unauthorized.

WHAT THIS MEANS: fabrication-prevention (structural trust) is landed and real; selection QUALITY is
venue-overfit — the registered thresholds/evidence fusion were shaped on 2 venues and fail on a 3rd.
do_not_promote holds everywhere. The honest path forward is the venue-generalization problem
(enrollment/court-presence assumptions break on crowded indoor + high-angle outdoor), fed by more
diverse REAL data — not another threshold round on the same clips. One-shot spent; these numbers
stand as the preregistered record.
