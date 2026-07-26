# Contamination memo — `ball_reviewed_corpus_chain_1121_3026` (ruling requested)

Asset: 3,026 human-reviewed ball centers, 40 rally folders, 6 YouTube source videos, at
`runs/lanes/w7_ballingest4_20260709/reviewed_corpus/` (40 folders verified on disk). Ledger row:
`runs/manager/data_ledger.json` → state **QUARANTINED**, BALL ruling CONDITIONAL, finding UNRULED.
`VERIFIED=0` binding; this memo decides nothing — it assembles the evidence for the owner's ruling.

## What was found (the 74.8% finding)
`runs/lanes/w7_ballretrain2_20260709/control_contamination_finding.json`: the "official tennis
control" model's zero-shot predictions match the CVAT ground-truth box centers to **~0.003–0.006 px**
on **26/40 clips (1,787/2,388 scored rows = 74.8%)** — not achievable by independent inference.
Pooled F1 splits 0.710 (26 flagged clips) vs 0.236 (14 clean clips) vs 0.596 blended. Suspected
mechanism (unconfirmed): CVAT semi-automatic prefill accepted with minimal correction; CVAT
provenance was outside that lane's scope.

## What "contamination" concretely means here — two distinct things
1. **Evaluation contamination (proven)**: labels on 26 clips mirror a model's outputs, so any metric
   scored against them is inflated (0.710 vs 0.236 above). Those rows can never be evaluation GT.
2. **Split contamination (structural)**: the corpus spans all six families including
   **HyUqT7zFiwk and Ezz6HDNHlnk — the parents of the frozen 167-row judge**
   (`w7_audit_stratum_scratch_350` partitions: val = HyU:100 + Ezz:67; Track B baseline pooled
   F1@20 0.5670 is scored on that judge, `runs/tracks/trackB_ball_20260722/STATUS.md`). Training on
   HyU/Ezz rows would leak the judge's venues/cameras into training.
It does **not** mean the media is unusable: rights are FYI-only per the owner directive 2026-07-22
(ledger `policy_directives`), and the same six rally videos are separately authorized as Stage-F
E-v2 EVENT training media (`online_harvest_20260706` ledger row).

## Overlap analysis (per-source rows from `runs/lanes/ball_b0_split_20260721/report.json`)
| Family | Rows in 3,026 | Judge parent? | Flagged clips | Protected-clip overlap |
|---|---|---|---|---|
| 73VurrTKCZ8 (outdoor day) | 397 | no | 1/8 | none — protected 4 are different identities |
| Ezz6HDNHlnk (outdoor night) | 400 | **YES (67 judge rows)** | 2/8 | none |
| HyUqT7zFiwk (indoor) | 560 | **YES (100 judge rows)** | 1/1 | none |
| _L0HVmAlCQI (night) | 554 | no | 19/19 | none |
| wBu8bC4OfUY (night) | 555 | no | 2/3 | none |
| zwCtH_i1_S4 (day) | 560 | no | 1/1 | none |

- **Judge**: family overlap = HyU + Ezz (960 rows). Frame-level overlap = **0 shared row keys**
  between the 350 scratch frames (judge source) and the 3,026 (B0 report, verified).
- **Train pools**: pb.vision B1/B2 train families are disjoint (different source IDs). The two
  protected harvest reservations (`pwxNwFfYQlQ`, `vQhtz8l6VqU`, heldout_eval_ledger HARVEST-1/2)
  are not among the six.
- **Protected eval clips** (`eval_clips_ball_protected_4`): different source identities; B0 scanned
  the 350 scratch frames vs all 2,953 protected frames (pixel-SHA + pHash): **0 collisions**.

## What the B0 split already built (2026-07-21, `runs/lanes/ball_b0_split_20260721/`)
Verdict `BALL_CLEAN_JUDGE` (fix2, byte-bound): **train = 2,249** (2,066 historical rows from the four
non-judge families + 183 scratch), **validation = 167 scratch-only** (HyU 100 + Ezz 67, fresh
no-prelabel review, 94 positives + 73 owner-attested negatives); the 960 historical HyU/Ezz rows are
excluded from everything; the 1,998 `confirmed_prelabel` rows are lineage-classified, and the 1,546
included in train carry weight 0.25, `ground_truth=false`, `evaluation_eligible=false`.

## Decision table
| Option | Meaning | Evidence for | Evidence against |
|---|---|---|---|
| 1. Clear fully | all 3,026 usable, any role | none | re-admits 960 judge-parent rows; re-blends machine-anchored rows into eval — exactly what produced the false 0.596 |
| 2. Clear with excluded families | ratify the B0 construction above | 0 shared row keys; 0 protected collisions (scratch); source intersection 0; prelabel rows demoted to 0.25-weight train-only; two review rounds passed | residual gaps listed below |
| 3. Keep held-out | corpus stays quarantined | zero risk | Track B loses its only in-domain human training rows (2,249); 167 judge unaffected either way |

**Recommendation: Option 2** — it is already engineered, cryptographically bound, and it is the
minimal construction that respects both contamination senses. Option 1 is indefensible on the
evidence; Option 3 is the automatic fallback if the residuals below bother you.

## Missing / not verifiable (state it, don't guess)
1. Root cause of the sub-0.01px matches is **unconfirmed** (CVAT job provenance for job_21 +
   ballingest2 batch never inspected).
2. No exhaustive protected-collision scan of the **2,066 historical train rows** (B0 scanned only
   the 350 scratch frames; the 3,026 row's ledger `overlap_check_coverage` is FAIL).
3. Ledger rows are stale vs B0: `w7_audit_stratum_scratch_350` still says label_count 0 /
   collision-check NOT_RUN despite B0's 350/350 ingest + 0-collision scan. A ruling should direct
   ledger reconciliation.
4. Channel-level identity: the ledger records video IDs, not channel IDs, so "same channel,
   different video" overlap cannot be checked from the ledger alone.
