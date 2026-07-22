# Owner check-in

Updated: 2026-07-22 03:00 PT. `VERIFIED=0` — nothing promoted; numbers below are frozen-protocol
measurements on independent held-out data unless marked otherwise.

## Your to-dos (only things blocked on you)

1. **Court labeling (CVAT tasks 88-91, ~45-60 min)** — the ONLY blocker you hold. The ingest
   adapter is built, reviewed, and waiting; the moment the export lands, the court diversity
   experiment runs. (CVAT Docker is currently off at your request — whenever you're ready.)
2. Nothing else. Task-87 ball labels + attestation: received and already powering results below.

## Best results per capability (time = wall-clock to produce; all costs actual)

| Capability | Best honest number (2026-07-22) | Protocol | Time/cost |
|---|---|---|---|
| BALL detector baseline | pooled F1@20 **0.5670** (indoor 0.7395 / outdoor-night 0.2933) | official WASB zero-shot on your attested 167-frame source-held judge | 41 min VM, ~$1 |
| BALL judge (your task-87 labels) | 167 eval rows (94 pos / 73 attested neg), zero contamination across 2,953 protected frames | 3 adversarial review rounds | ~1 day incl. reviews |
| EVENT head causal test | teacher timing signal is REAL: +0.13 F1 vs 0.0 placebo, timing error 64→2 frames — but detector fails safety guards (over-fires on hard negatives, under-fires 3-10x on full video) → experiment closed honestly, **NO_LIFT** | frozen A/B/C, your 41 owner-val labels, protected-50 never touched | ~$20 total |
| PERSON public-data retrain | **dead end confirmed cheaply**: public Roboflow boxes collapse to 7 original-footage families (< the 8 required) once real lineage is enforced; $0 GPU wasted | pixel-level leak scanning, 66M pairs | CPU only |
| PERSON mixed-pool experiment (your call) | running now on GPU: your 40+-source self-training design, control vs mixed | held-out human families only | ~$5-8 budget |
| BALL retrain (A/B) | running now on GPU: human-only vs human+pb.vision-teacher, judged on your 167 frames | frozen matched arms | ~$3-6 budget |
| COURT adapter | built + 4-round reviewed; waiting only on your tasks 88-91 | 66 usable train rows/18 families staged | CPU only |

Fresh numbers land here as each run finishes.
