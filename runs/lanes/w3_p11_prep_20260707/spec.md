# LANE w3_p11_prep_20260707 — P1-1 pretrain warm-start HARNESS on the Roboflow corpus (wave-3 #4; prep only, GPU run is a later lane)

## HARD RULES (violating any = lane rejected)
- You are Codex lane `w3_p11_prep_20260707`. Work ONLY inside /Users/arnavchokshi/Desktop/pickleball.
- No git branches/commit/push. Write runs/lanes/w3_p11_prep_20260707/commit_manifest.md for the manager.
- OWNED files (edit only these + NEW files as listed + runs/lanes/w3_p11_prep_20260707/):
  `threed/racketsport/roboflow_corpus.py` (extensions), `threed/racketsport/ball_tracknet_cvat_dataset.py`, `tests/racketsport/test_roboflow_corpus.py`, `tests/racketsport/test_ball_tracknet_cvat_dataset.py`; NEW: `scripts/racketsport/train_ball_pretrain.py` (+ config file(s) under configs/ or alongside existing config conventions — follow repo precedent), new test files.
- FENCED (do not edit; propose diffs if needed): `scripts/racketsport/process_video.py`, `threed/racketsport/camera_motion.py`, `scripts/racketsport/estimate_camera_motion.py`, `scripts/racketsport/remote_body_dispatch.py`, `scripts/fleet/*`, placement/grounding code.
- **DATA DISCIPLINE (the one that matters most)**: INTERNAL-VAL ONLY. Training/eval touch ONLY `data/roboflow_universe_20260706/` (via the aggregated index) and its internal splits. ZERO held-out shots: nothing from runs/manager/heldout_eval_ledger.md rows, nothing from pwxNwFfYQlQ / vQhtz8l6VqU, no Outdoor/Indoor labels. Your loader must ASSERT at construction that no sample hash collides with the 35 protected eval hashes (the aggregation lane's check — reuse its mechanism from the corpus card / roboflow_corpus.py) and fail loud on any collision.
- **DISK**: the Mac was recently at 96% (now ~58G free). The corpus is INDEX-BASED — never copy/duplicate images; any caching must be opt-in, bounded, and documented. Report du delta of everything you create (target ≤2GB incl. smoke checkpoints).
- Honest reporting; WIDE suite `MPLBACKEND=Agg .venv/bin/python -m pytest tests/racketsport -q --ignore=tests/racketsport/court_finding_technology_benchmark.py` + focused suites; classify residual failures REAL / PRE-EXISTING (prove at HEAD) / SANDBOX-SUSPECT / CROSS-LANE-SUSPECT.
- importorskip("torch") (train code must degrade to skip cleanly where torch absent); new CLI ⇒ direct-CLI reference test same-lane; no new root .md. `.venv/bin/python` always.
- Read first: NORTH_STAR PART 0 + IV + Part III P1 section; BUILD_CHECKLIST last ~15 (esp. [P1-0 DOWNLOAD RULED], [P1-0 ROBOFLOW AGGREGATE], [P1-0 COMPLETE], [VISIBILITY-SCHEMA RULED]).

## CONTEXT (established)
- Corpus: `data/roboflow_universe_20260706/aggregated/` — 61,260 kept samples (110,003 considered, 44.3% dHash-deduped), buckets core_pickleball=59 sources / adjacent_sport_aux=3; temporal split 84,459 sequence / 25,903 still (pre-dedup counts; use the index's actual fields); 0 collisions vs 35 protected eval hashes; fork/mirror maps + ambiguous-tennis spot-check in corpus_card.json. Loader-smoke passed in the aggregation lane.
- 4-level visibility schema (clear/partial/full/out_of_frame + WBCE weights 1/2/3/3) landed end-to-end in wave 2 — but PUBLIC corpus labels have visibility HONESTLY ABSENT (never invent it; the aggregation preserved that honesty).
- Standing lesson (BALL history): an 8,631-frame public pickleball fine-tune made BOTH architectures WORSE on held-out — public data ≠ our domain. P1-1 is therefore a WARM-START pretrain evaluated on INTERNAL corpus splits only; owner in-domain data remains the finisher. Do not claim domain transfer; do not touch held-out to "check".
- Existing training infra: the repo has prior ball fine-tune/training paths (the U-Rochester-era experiments + TrackNet/WASB adapters + ball_tracknet_cvat_dataset.py). DISCOVER and REUSE/EXTEND them — a from-scratch trainer is wrong unless you show the existing one is unusable (say why in the report).

## OBJECTIVE — everything up to (excluding) the GPU run
1. **Index-based dataset**: a training Dataset over the aggregated index resolving images in-place from the 65 dataset dirs; point-label conventions normalized to ours; absent-visibility policy explicit (document the WBCE weight used when visibility is unknown — pick a principled default and state it); temporal-sequence sampling for sequence-bucket sources vs stills-as-aux; corrupt/missing-file tolerance fail-loud with a named skip-list artifact (not silent).
2. **Aux-domain mixing**: config knob for core_pickleball : adjacent_sport_aux mixing ratio (default you propose, justified in one paragraph).
3. **Train harness**: `scripts/racketsport/train_ball_pretrain.py` warm-starting our primary architecture (WASB-style is the anchor per the detector zoo's measured value; support the TrackNet-family config too if the existing infra makes it near-free — report what exists). Checkpointing: atomic, resumable, durable-path-friendly (the GPU lane will checkpoint every stage boundary per fleet preemption rules).
4. **Internal-val eval harness**: metrics on the corpus internal_val split (F1@px, recall, precision at the conventions the repo already uses for ball eval) + a zero-shot baseline number produced by the SAME harness (evaluate the current pretrained checkpoint un-finetuned on the same split) so the GPU lane's improvement claim has a baseline from identical code.
5. **CPU smoke proof**: tiny-subset run (bounded steps) showing loss decreases and checkpoint save/load round-trips; internal-val harness runs end-to-end on the tiny subset. Wall-clock + du reported.
6. **GPU runbook**: `runs/lanes/w3_p11_prep_20260707/gpu_runbook.md` — exact commands for the Sonnet GPU lane (env setup, data transfer strategy for an index-based corpus [state exactly what must be shipped to the VM: the index + the 65 image dirs = ~7GB, or the on-VM re-download alternative — recommend one with a paragraph], train command, expected artifacts, checkpoint cadence, internal-val command, expected runtime order-of-magnitude on A100-40GB vs H100-80GB if statable).

## ACCEPTANCE
- Loader unit tests: index resolution, eval-hash-collision assertion (fails loud on a synthetic collision), absent-visibility policy, sequence-vs-still sampling — all green.
- CPU smoke: loss strictly decreases over the bounded run; checkpoint round-trip byte-consistent state; internal-val harness produces the metric table on the tiny subset.
- Zero-shot baseline number produced by the harness on the REAL internal_val split (CPU — WASB CPU inference is slow; a documented bounded subsample of internal_val is acceptable IF the subsample is seeded/reproducible and its size justified; report wall-clock).
- Runbook complete enough that a GPU lane needs zero design decisions.
- WIDE suite green per HARD RULES; du delta ≤2GB.

## SELF-ITERATION + BOUNDED FIX AUTHORITY
Iterate to green within owned files. If existing adapters need a fenced-file change, deferred patch + report.

## EVIDENCE
- data/roboflow_universe_20260706/aggregated/ (corpus_card.json/md + indices) — ground truth for fields/splits.
- threed/racketsport/roboflow_corpus.py (the 1,029-line aggregation module — your loader foundation).
- Existing ball training/eval code: ball_tracknet_cvat_dataset.py, wasb/tracknet/totnet adapters, ball_cvat_benchmark.py, prior fine-tune experiment dirs under runs/ (find them; BALL memory says they exist from the U-Rochester attempt era).

## STRUCTURED REPORT
objective_result vs acceptance; acceptance table; what-existed-vs-what-you-built (reuse honesty); absent-visibility + mixing-ratio decisions w/ one-paragraph justifications; zero-shot baseline table; changes file:line; full_suite + classification; du delta; HONEST ISSUES; NEXT; commit_manifest path; BUILD_CHECKLIST bullet DRAFT in report.
