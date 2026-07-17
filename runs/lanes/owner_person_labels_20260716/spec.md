# Lane spec — owner_person_labels_20260716 (Codex gpt-5.6-sol high, CPU-only, Track F)

Coordinator GO 2026-07-16: build the owner person-box labeling channel that unblocks the
RF-DETR-L fine-tune arm (data ruling: runs/lanes/trk_detbench_20260716/FINETUNE_DATA_RULING.md —
read it first; it defines the data requirement this channel must satisfy). Precedent to imitate:
the owner event-labeling channel (runs/lanes/owner_event_labels_20260715/ + its INGEST_README.md
+ committed CLIs in d0ce58bdd) — but STILLS + BOXES, not video clips.

## Objective (acceptance = all of these true)

Build three tools + stage one owner session:

1. **Sampler** (`tools/sample_frames.py` in this lane dir): deterministic (seed 20260716)
   stratified frame sampler over the 39 landscape owner clips at
   `runs/owner_footage_intake_20260702/raw/` (probe list:
   runs/lanes/trk_detbench_20260716/owner_footage_probe_20260716.txt; EXCLUDE the 9
   "Screen Recording*" files and the 2 portrait files → gameplay .mov clips only).
   - Extract candidate frames w/ ffmpeg (uniform random timestamps, per-session quotas;
     session = IMG-number cluster, mapping written to the split manifest).
   - Run YOLO26m (models/checkpoints/yolo26m.pt, repo venv, CPU or MPS — NO fleet GPU) on
     candidates at conf 0.25 to produce PROPOSALS and to stratify:
     >4 persons → spectator-rich stratum; ≤2 persons → empty/sparse stratum; else gameplay.
   - Final pack ~450 frames: ~300 gameplay, ~100 spectator-rich, ~50 empty/sparse (fill from
     gameplay if a stratum is short — record actuals). 15-20% of frames (across ALL strata,
     seeded) = SCRATCH stratum: their proposals are withheld from the page (measures
     proposal-anchoring bias). Frames written as JPEG q=2 1920x1080 into the staging folder.
   - Emits `pack_manifest.json` (frame → clip/session/timestamp/stratum/scratch flag/proposals)
     — manifest stays in the LANE DIR, NOT the staging folder (page must be stratum-blind).
2. **Review page** (static, self-contained, generated into
   `/Users/arnavchokshi/Desktop/person_labels_20260716/START_HERE.html` + `frames/` + one JS/CSS
   inline or adjacent):
   - Shows one frame at a time w/ proposal boxes (none for scratch frames); owner can DRAG to
     draw, drag corners/edges to adjust, delete, and toggle class per box:
     `player` (green) / `off_court_person` (orange).
   - Keyboard: A/D or arrows = prev/next frame; W = new box mode; X = delete selected;
     C = toggle class of selected; E = confirm-empty (marks frame "no people outside court /
     all boxes correct-empty"); S = save. Big on-screen buttons for the same. Progress
     "N of 450" + per-frame done-state tick.
   - AUTOSAVE to localStorage on every mutation (stable key `person_labels_20260716`) + a big
     mid-session **Save** button + end-of-session **Export** button downloading
     `person_labels_export.json` (schema: frame id, boxes [x1,y1,x2,y2, class,
     source=proposal_confirmed|proposal_adjusted|proposal_deleted|drawn], empty_confirmed,
     ms_spent). NO native <video> elements anywhere (stills only). NO stratum/scratch/clip-name
     leakage in the UI or DOM (frame ids are opaque hashes; mapping lives in the lane-dir
     manifest).
   - Works offline from file:// (no external assets, no fetch).
3. **Ingest CLI** (`tools/ingest_labels.py`): export JSON + pack_manifest →
   `data/owner_person_labels_20260716/` with: per-label provenance (clip, PTS, reviewer=owner,
   source flag), session-disjoint train/val split manifest (hold out 1-2 whole sessions),
   AUDIT report: label counts per class/stratum, empty-confirm counts, proposal
   confirmed/adjusted/deleted/drawn stats, scratch-vs-proposal comparison (anchoring metric:
   per-frame box count + IoU-matched agreement between scratch-frame owner boxes and the
   WITHHELD proposals), eval/protected-disjointness assertion (trivially true — owner footage;
   assert clip ids ∉ eval_guard registry anyway). Idempotent; dry-run mode; refuses partial
   exports without --allow-partial.
4. **Tests** (lane dir, pytest): sampler determinism (same seed → byte-identical manifest),
   schema round-trip export→ingest, scratch-withholding (no proposals leak into scratch frames
   on the page), stratum-blindness (grep the generated HTML/JS for clip names/stratum strings =
   zero hits), ingest audit math on a synthetic fixture, empty-confirm path.

## Hard rules

- CPU/MPS local only; NO GPU fleet, NO network needed (STOP typed if a dependency is missing
  from the repo venv rather than pip-installing from the network).
- Fences: write ONLY runs/lanes/owner_person_labels_20260716/**,
  /Users/arnavchokshi/Desktop/person_labels_20260716/** (via --add-dir), and
  data/owner_person_labels_20260716/** (ingest output; create). DO NOT touch
  ~/Desktop/event_labels_20260715/ (frozen owner pack), any pipeline code, configs, or other
  lanes' files. No git commits.
- The staged page is DECLARED STAGED ONLY by the Track F manager after independent browser
  verification (the event page shipped broken twice — your own checks are necessary, not
  sufficient). Deliver it working; the manager verifies.
- Label-integrity lessons (binding): dt/interaction integrity — every mutation autosaved;
  page reload mid-session must restore state exactly; Export must equal localStorage state;
  no interaction may silently change an earlier frame's answers.
- Owner session budget: page must support ~450 frames in 75-115 min → target ≤2s frame-to-frame
  navigation (preload next image), zero page reloads between frames.

## Report (lane dir report.json + final message)

Sampler actual counts per stratum/session; scratch fraction; page byte size; test results w/
real exit codes; ingest dry-run proof on a synthetic export; the exact one-line owner
instruction; expected owner session time computed from frame count; any deviations. VERIFIED=0;
this lane produces tooling + a staged pack, no labels and no model claims.
