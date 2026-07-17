# Fine-tune arm data-feasibility ruling — Track F manager, 2026-07-16

Gate question (coordinator): does the training data the ruled recipe needs (on_court_player boxes
+ spectator hard negatives, OWNED/reviewed, source-disjoint from the frozen eval clips) exist on
disk today in usable form?

## RULING: OUTCOME (c) — the data does not exist. Fine-tune NOT dispatched. $0 GPU this ruling.

Inventory evidence (full sweep, agent-verified against the files):

1. The ONLY human-reviewed person boxes on disk are the four protected clips in
   `runs/cvat_imports/2026_06_30/` (11,459 boxes total): burlington + wolverine are the frozen
   eval clips themselves (`internal_val_only` in `threed/racketsport/eval_guard.py`), indoor +
   outdoor are `strict_holdout`. All four are players-only — NO spectator/negative class exists
   anywhere on disk. Usable for training: zero.
2. `runs/cvat_imports/harvest_review_20260707/` (6 source-disjoint YouTube clips): ball-only
   reviews; person_ground_truth files are truly empty (0 boxes) — not a schema artifact.
3. `data/roboflow_universe_20260706/` (~62.5k third-party images w/ player classes, 15x CC BY
   4.0): REJECTED for this arm — not owned, not owner-reviewed, single class (no negatives), and
   decisively: per-frame source-disjointness from the eval clips is UNVERIFIABLE (YouTube-derived;
   could contain the same games) → training on it risks silently poisoning the frozen card.
4. Owner footage: 41 raw clips, zero labels. `owner_IMG_1605` has machine-only person prelabels
   explicitly `train_eligible: false` (standing rule: predictions are not GT).
5. Both eval clips are themselves YouTube-sourced — owner-shot footage is disjoint from them by
   provenance, which makes it the ideal clean training source.

No synthetic shortcut is taken (per directive; also per the RKT research finding that
synthetic-only person/domain claims would be unsupported).

## Exact data requirement (what the fine-tune lane needs before it can dispatch)

- POSITIVES: `on_court_player` boxes — every visible/truncated/occluded on-court player —
  ~1,500-2,000 boxes ≈ 375-500 frames (≈4 players/frame).
- NEGATIVES (the class that exists nowhere today): explicit `off_court_person` boxes on
  ~300-500 spectator/passer/far-court instances + 100-200 confirmed empty-court/sideline-only
  frames (zero-box confirmations — fast).
- STRATA: near/far side, net occlusion, player overlap, frame-edge truncation, empty frames;
  ≥5 distinct owner sessions; split by session (never adjacent frames across splits).
- REVIEW PROTOCOL: prelabel-assisted (YOLO26m proposals) with a correction-required flow AND a
  15-20% scratch-labeled stratum — the BALL program measured confirm-heavy review inflating
  scores (~74.8% byte-identical prelabels); we pre-commit the mitigation. Every label carries
  source/frame/reviewer per NS-02.2; machine boxes never pass through unreviewed.
- RIGHTS: owner-shot → commercial-clean lane. OPTIONAL R&D-FLAGGED SUPPLEMENT if owner footage
  proves spectator-sparse: the 6 harvest clips are verifiably source-disjoint (distinct YouTube
  IDs) and may supply spectator-dense frames, at the price of R&D-only posture on the resulting
  weights (NS-07.3 flag).

## Collection plan (concrete, probed 2026-07-16)

SUPPLY EXISTS: `runs/owner_footage_intake_20260702/raw/` holds 39 landscape owner clips,
~3,193s, incl. seven long 1920x1080 game clips (IMG_7768 587s, IMG_5014 563s, IMG_4983 556s,
IMG_1014 335s, IMG_9545 213s, IMG_4982 153s, IMG_9543 134s) across multiple sessions
(IMG-number clusters ≈ ≥6 distinct sessions). Full probe log:
runs/lanes/trk_detbench_20260716/owner_footage_probe_20260716.txt (39 landscape / 2 portrait,
per-clip resolution + duration).

1. TOOLING LANE (ready to dispatch, CPU-only, no GPU): adapt the proven owner clip-review
   channel (owner_event_labels_20260715 pattern: sampler + HTML review page + ingest CLI, 15
   tests) to person boxes: stratified frame sampler over the 7 long clips + short-clip pool
   (~450 frames: ~300 gameplay-stratified, ~150 spectator/empty-focused), box-drag correction UI
   over YOLO26m proposals, `off_court_person` class, one-key empty-frame confirm, scratch
   stratum enforced, ingest with provenance + session-disjoint split manifest. Estimated lane:
   half-day Codex, fence = new lane dir + a new data/owner_person_labels_<date>/ dir.
2. OWNER SESSION (the only human dependency): ~450 frames at 10-15 s/frame ≈ **75-115 minutes,
   one sitting**, same channel UX the owner already used for events.
3. INGEST + FINE-TUNE DISPATCH: after ingest passes its audit (label counts, strata coverage,
   split disjointness, scratch-vs-prelabel correction stats), the fine-tune lane dispatches per
   benchmark_spec_trk.md arm 4 + the trk_detbench cautions: ≤8 GPU-h, frozen-threshold, eval
   through the PRODUCTION pool path only (FEEDER_DRIFT ruling), full frozen card + pre-registered
   stop rules. A fine-tuned candidate earns nothing until that card.

## Status

Fine-tune arm: BLOCKED-ON-DATA (typed, per directive outcome c). GPU slot not consumed.
Pool-construction diagnostic: booked spec-only at runs/lanes/trk_pooldiag_20260716/spec.md.
VERIFIED=0 unchanged.

## ADDENDUM — 2026-07-16 night (owner ruling; supersedes the collection plan's dispatch path)

1. FINE-TUNE ARM PARKED by owner ruling ("this really isn't that deep, I'm pretty sure the
   RF-DETR will run good enough"). This document stays on file as the data requirement IF the
   arm is ever revived. The zero-shot RF-DETR-L production-path result is the active track.
2. CVAT mystery CLOSED: the person labels the owner remembered are the four protected demo
   clips (11,459 boxes) already inventoried above — no missing labels exist.
3. CONTENT-FAILURE LESSON (binding): the "39 owned landscape clips" supply claimed above FAILED
   content verification — runs/owner_footage_intake_20260702/raw/ is dance-rehearsal/personal
   footage (Sway product), NOT pickleball (contact-sheet proof:
   runs/lanes/owner_person_labels_20260716/content_audit/). The staged pack was revoked and
   deleted before any owner labels were collected. Any future owner-facing pack requires:
   (a) decoded-thumbnail content verification of every source clip, (b) owner confirmation of
   the source-clip list BEFORE rendering, in addition to functional page verification. The
   collection plan's owned-supply section is therefore VOID until a real owned-pickleball
   source exists (e.g. fresh owner captures under NS-02).
