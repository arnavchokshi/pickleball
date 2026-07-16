# LANE SPEC — owner_event_labels_20260715 (Codex, gpt-5.6-sol high)

Dispatcher: Track E manager (Fable), 2026-07-15. Pin: repo HEAD d74897203bd7ff92ff03bff437eebc0041f0f5fe.

## WHY (context you must ground on)

Today the owner spot-checked 50 Tier-A audio×track bootstrap labels and the gate FAILED
(29/50 true contacts vs the ≥47/50 bar; 12/29 true contacts had |timing error| ≥0.2s). The
auto-labeler is REJECTED as a training-label source. What was PROVEN: the owner labeled 50
rows in ~20 minutes with a local clip-review page (`~/Desktop/spotcheck_tier_a_clips/START_HERE.html`)
— half-speed ±0.6s clips centered on candidate PTS; per clip a 5-way decision
(paddle/ground/other/none/unclear) + click on the ball at contact + single-frame ◀/▶ nudges
recording a timing offset. That channel is the new label supply. Your job: turn the ad-hoc
pack into a proper, tested, reusable generator + an unbiased stratified sampler + an ingest
script, and STAGE tonight's ~300-clip owner session. The 50 already-reviewed rows are being
committed by another track as a PROTECTED EVAL SEED — they must be excluded from all sampling.

## HARD RULES

1. NO git branches, NO git commits. The manager commits after ruling.
2. Read first: `NORTH_STAR_ROADMAP.md` (§2.2 DATA row, §5 queue, §6 standing rules),
   `AGENTS.md`, `runs/manager/inflight_lanes.md` (collision fences).
3. PROTECTED DATA: `eval_clips/ball/**` (Outdoor/Indoor + all protected eval clips) is
   EVAL-ONLY and must never be read, sampled, or referenced by the sampler. The 50 reviewed
   spot-check rows (`runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json`) are a
   PROTECTED EVAL SEED: exclusion-input only, never re-sampled.
   `data/pbvision_11min_20260713/**` is competitor R&D reference: NEVER sampled, never labels,
   never training. VERIFIED=0 stays binding; no promotion language anywhere.
4. Do NOT touch: `NORTH_STAR_ROADMAP.md`, `scripts/racketsport/process_video.py`, `ios/**`,
   `runs/lanes/pbv11_headtohead_20260713/**`, `runs/lanes/event_bootstrap_20260713/**` (read-only
   evidence), `brand-exploration/**`, `cvat_upload/**`, `runs/manager/**`, `runs/HANDOFF_20260714.md`,
   `configs/**`, `data/**` (read-only; your outputs go under YOUR lane dir only).
   Preserve all unrelated dirty worktree changes — other tracks are live in this worktree.
5. Honest reporting. Run the WIDE blast-radius test suite (`MPLBACKEND=Agg
   .venv/bin/python -m pytest tests/racketsport -q`), not a hand-picked subset. Failures>0
   while claiming PASS = rejected unless each failure is proven pre-existing (reproduce on
   stash or cite the inflight ledger's known sandbox-socket / pre-existing set).
6. Every new CLI ships its direct-CLI reference test (subprocess invocation of the real
   entrypoint) in the same lane. New scripts must survive
   `.venv/bin/python scripts/racketsport/list_scaffold_tools.py --root .`,
   `.venv/bin/python scripts/racketsport/audit_dead_code.py --root .`, and
   `python3 scripts/racketsport/audit_storage_policy.py --root . --json` (0 unknown; if your
   staged pack under the lane dir trips the storage audit, register it the same way existing
   staged labeling packages are allowlisted — inspect how `data/event_bootstrap_20260713` /
   `cvat_upload/*` entries are handled and follow that mechanism; do not weaken the audit).
7. No new root .md files. All artifacts under `runs/lanes/owner_event_labels_20260715/`.
8. No network. Everything is local. No GPU.

## FILE OWNERSHIP (exact; you own nothing else)

- NEW `scripts/racketsport/build_event_review_session.py`
- NEW `scripts/racketsport/ingest_event_review_results.py`
- NEW `tests/racketsport/test_event_review_session.py` (may add
  `tests/racketsport/test_event_review_ingest.py` if you prefer two files)
- `runs/lanes/owner_event_labels_20260715/**` (manifest, staged pack under `pack/`, report, log)
- IF (and only if) the storage audit requires it: the single allowlist entry per rule 6.

## OBJECTIVE + ACCEPTANCE NUMBERS

Build three things, then execute a real staging run.

### A. Stratified sampler (subcommand `sample` of build_event_review_session.py)

Universe: exactly the 40 rally clips `data/online_harvest_20260706/rallies/<source_id>/*.mp4`
across all 6 sources (`_L0HVmAlCQI`, `73VurrTKCZ8`, `Ezz6HDNHlnk`, `HyUqT7zFiwk`,
`wBu8bC4OfUY`, `zwCtH_i1_S4`). `data/online_harvest_20260712` contains no rallies (frames-only
harvest) — assert this and record it in the manifest. Anchors must lie in
[0.7s, clip_duration−0.7s] (so the ±0.6s render window fits).

Target N=300 rows total, three strata (record `stratum` per row):

- `audio_onset` (target 120): candidates from
  `data/event_bootstrap_20260713/audio_onsets_v0/<clip_id>.json` `onsets[]`, anchor =
  onset `corrected_time_s` snapped to the nearest frame PTS (same first-video-PTS
  normalization the bootstrap used — sanity-check your snapping reproduces
  `spot_check_tier_a_50.json` anchors from their `evidence.audio.corrected_time_s` within
  one frame on ≥45/50 rows; if systematic disagreement >0.25s, STOP and report — do not ship
  wrong anchors). ANTI-SELECTION-BIAS RULE: do not take top-scored onsets; split each
  source's onset pool into score terciles (by `score`) and draw evenly from all three,
  recording `score` and `score_band` (`high`/`mid`/`low`) per row.
- `track_discontinuity` (target 75): recompute candidates DIRECTLY from
  `data/online_harvest_20260706/prelabels/<clip_id>/ball_track.json` (do NOT reuse bootstrap
  tier logic — it failed review; a fresh simple detector keeps provenance clean). Detector
  (document exact constants in the manifest): over frames with `visible` and conf ≥ your
  documented threshold, flag frame f when (direction change ≥ 60° with adequate pre/post
  speed) OR (speed ratio ≥ 1.8 or ≤ 1/1.8) OR (a ≥3-frame visibility gap boundary). Rank by a
  documented strength score; draw evenly across strength terciles (same anti-bias rule);
  record features per row. Anchor = the frame's `t` (PTS seconds).
- `uniform_random` (target 105): uniform over [0.7, dur−0.7] per clip, clip share
  proportional to clip duration within each source's allocation; enforce ≥1.3s separation
  between uniform picks in the same clip; NO exclusion near signal candidates (this stratum
  is the unbiased audit + true-negative supply demanded by North Star §2.2 DATA row).

Per-source allocation per stratum: proportional to total rally duration per source
(pre-measured: _L0HVmAlCQI 430.1s, 73VurrTKCZ8 287.3s, Ezz6HDNHlnk 808.9s, HyUqT7zFiwk 971.0s,
wBu8bC4OfUY 538.5s, zwCtH_i1_S4 606.0s; re-derive from ffprobe yourself, don't trust these
numbers), floor 8 rows/source/stratum, cap 30% of the stratum per source, largest-remainder
rounding, deficit taken from the largest source. If a source's candidate pool can't fill its
audio/track quota after dedup, backfill first from other terciles of the same source, then
from other clips of the same source, then redistribute to other sources — record every
shortfall/backfill in the manifest.

Cross-stratum dedup within a clip: if a `track_discontinuity` anchor lands within 0.3s of an
already-drawn `audio_onset` anchor, drop and redraw (count collisions in the manifest).
Same-stratum min separation 0.8s within a clip for audio/track strata.

HARD EXCLUSIONS (each one assert-tested):
- E1: no sampled anchor within ±0.75s of any of the 50 reviewed rows'
  (`clip_id`, `anchor.pts_s`) from `spot_check_tier_a_50.json` on the same clip.
- E2: nothing sampled from `data/pbvision_11min_20260713` (skip its onset file too).
- E3: every sampled video path is under `data/online_harvest_20260706/rallies/`; assert no
  path under `eval_clips/` or `data/testclips/` can enter the universe.
- E4: anchor bounds as above.

DETERMINISM: master seed 20260715; derive per-(stratum, source, clip) RNG substreams via
hashing (adding a clip must not reshuffle other clips' draws). Same seed ⇒ byte-identical
manifest JSON (test this). Presentation order: one deterministic shuffle of all rows so
strata are interleaved; the PAGE MUST NOT reveal stratum to the owner (blind labeling —
no stratum strings in the HTML or filenames; test this).

Manifest (`session_manifest.json`, write to the lane dir): per row
`{label_id (opaque, e.g. els20260715_NNN), row (presentation order), stratum, score_band,
signal_features, clip_id, source_group, video_path, video_sha256, source_fps, anchor_pts_s,
anchor_frame, suggested_split}` + header `{session_id, seed, generator_version,
generator_sha256 (sha256 of build_event_review_session.py), git_head, created_at, universe
description, allocation table, exclusion counts (E1 hits avoided, collisions, shortfalls),
sampler constants, expected_owner_minutes}`. `suggested_split` copies the source-disjoint
split from `data/event_bootstrap_20260713/manifest_v0.json` `source_split` (train:
73VurrTKCZ8, Ezz6HDNHlnk, _L0HVmAlCQI, wBu8bC4OfUY; validation: HyUqT7zFiwk, zwCtH_i1_S4)
so future train/val/test stays source-disjoint. `expected_owner_minutes`: compute honestly
from the measured 50-rows-in-~20-min baseline (~24s/row) plus a faster floor for
decision-only rows; report a range, not one flattering number.

### B. Pack renderer (subcommand `render`)

Given a session manifest, cut clips + emit the page into an output dir:

- ffmpeg per row: accurate-seek cut of [anchor−0.6, anchor+0.6] from the SOURCE rally mp4;
  half-speed video (`setpts=2.0*PTS`), pitch-preserving half-tempo audio (`atempo=0.5`);
  re-encode h264+aac. Output `NNN_pts<anchor>_<clip_id>.mp4` (NNN = presentation row,
  zero-padded). Acceptance: 300/300 rendered, 0 failures; every output ffprobe-decodes with
  duration in [2.25, 2.55]s AND has an audio stream. If a segment can't render, resample
  deterministically within the same (stratum, source) and record the substitution; >5%
  substitutions = STOP and report.
- `START_HERE.html`: self-contained single file, SAME UX as
  `~/Desktop/spotcheck_tier_a_clips/START_HERE.html` (read it first; reproduce: video paused
  at clip center on load, 5-way decision buttons — paddle/ground/other(net/body)/none/unclear;
  only paddle/ground/other proceed to the click phase; click records x,y normalized to the
  displayed video (origin top-left, 4 decimals); ◀/▶ nudge one SOURCE frame per press with
  dt recorded in SOURCE seconds relative to anchor (clip is half-speed: dt =
  (currentTime − duration/2)/2; nudge step in clip time = 2/source_fps — per-row
  `source_fps` from the manifest, do NOT hardcode 1/15: one source is 23.98 fps);
  go-back-one-clip undo; progress "N of M"; localStorage resume under key
  `event_labels_20260715_answers_v2`; Export button downloads JSON). IMPROVEMENTS required:
  keyboard shortcuts (1..5 = decisions, ←/→ = nudge, documented on-page) and export rows
  carry `label_id` for a robust ingest join. ITEMS rows carry
  `{row, file, label_id, anchor_pts_s, source_fps}` — NOT stratum.
- Export format (v2 results schema): `{results_schema_version: 2, session_id,
  page_generator_version, coords: "normalized to displayed video, origin top-left",
  dt: "seconds in SOURCE time relative to the labeled anchor PTS",
  answers: {"<row>": {label_id, decision, x?, y?, dt?}}}`; none/unclear rows carry decision
  only.

### C. Ingest script (`ingest_event_review_results.py`)

`--results <owner export json> --manifest <session_manifest.json> --out-dir <dir>` ⇒ a
versioned reviewed-labels dataset: `reviewed_labels_v2.jsonl` (one row per answered item:
label_id, clip_id, source_group, video_path, video_sha256, anchor_pts_s, stratum, score_band,
decision, contact {x_norm, y_norm, x_px, y_px at source resolution}, dt_s,
corrected_contact_pts_s = anchor_pts_s + dt_s, suggested_split, review {session_id,
reviewed_by: "owner", ingested_at}, provenance {seed, generator_version, generator_sha256,
manifest_sha256, results_sha256}) + `dataset_manifest.json` (counts by stratum×decision,
schema version, honest_limits note that these are owner-reviewed bootstrap-era labels, not a
VERIFIED promotion). Validation (each rejection tested): decision in the 5-way enum;
paddle/ground/other require x,y∈[0,1] and |dt|≤0.65; none/unclear must NOT carry x/y/dt;
label_id must join the manifest; duplicate rows rejected; unanswered rows listed explicitly.
The owner's real results don't exist yet — test with synthetic exports built from a small
manifest fixture.

### D. Real staging run (execute, don't just build)

1. `sample` with seed 20260715 ⇒ `runs/lanes/owner_event_labels_20260715/session_manifest.json`.
2. `render` ⇒ `runs/lanes/owner_event_labels_20260715/pack/` (START_HERE.html + all clips).
   The manager will move the pack to `~/Desktop/event_labels_20260715/` after ruling — you
   stay inside the lane dir (sandbox). Do NOT git-add the pack or any mp4.
3. Write `runs/lanes/owner_event_labels_20260715/INGEST_README.md` with the exact one-line
   ingest command for when the owner's export comes back, plus the counts table.
4. Report per-stratum × per-source counts, collisions, shortfalls, substitutions, pack size,
   and the anchor sanity-check result (A. above).

Acceptance summary (PASS requires all): manifest exactly N=300 (or documented shortfall with
arithmetic), byte-identical on same-seed re-run; all four exclusion classes assert-tested
green; 100% clips rendered + ffprobe-validated (duration + audio); page blind to stratum;
new tests + direct-CLI tests EXIT 0 in `.venv`; wide suite failures all proven pre-existing;
scaffold/dead-code/storage audits clean; staging artifacts on disk.

## EVIDENCE TO READ FIRST

- `~/Desktop/spotcheck_tier_a_clips/START_HERE.html` — the proven UX to reproduce (read the JS).
- `runs/lanes/event_bootstrap_20260713/spot_check_tier_a_50.json` — exclusion seed + anchor
  conventions; `.md` twin has the owner's merged decisions.
- `data/event_bootstrap_20260713/manifest_v0.json` — source_split, honest_limits, provenance style.
- `data/event_bootstrap_20260713/audio_onsets_v0/*.json`, `data/online_harvest_20260706/prelabels/*/ball_track.json` — signal artifacts.
- `data/online_harvest_20260706/rallies/*/*.provenance.json` — fps/resolution/provenance fields.
- `runs/HANDOFF_20260714.md` — event-head program context.

## MANDATORY STRUCTURED REPORT (schema-validated report.json; the manager rules on this)

- `objective_result`: PASS/FAIL against every acceptance number above, with the real numbers.
- `full_suite`: wide-suite passed/failed counts + proof each failure is pre-existing.
- HONEST ISSUES: anything weak, hacky, or unverified — say it plainly.
- Artifacts: absolute paths of everything produced.
- Dated ledger bullet: one line for `runs/manager/inflight_lanes.md` (the manager writes it).
- BEST-STACK DELTA (mandatory): expected (c) NO stack delta — this is a data-channel tool, no
  model/policy/weights change; state it explicitly. If you believe otherwise, justify.

## ANTI-PASSIVE-WAIT

All work is local CPU (ffmpeg for ~300 short clips ≈ minutes). Do not end your turn to wait
on anything; run everything foreground to completion. End only with the final report.
