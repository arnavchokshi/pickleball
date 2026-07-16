# MANAGER RULING — owner_event_labels_20260715 (Track E, 2026-07-15)

RULING: ADOPT (scoped pass). Labeling-channel tooling only — no capability promotion,
no best-stack delta, VERIFIED=0 binding. Ruled on the manager's independent verification
battery (below); the lane's own report.json was still being written during a contended
wide-suite run and is corroboration, not the load-bearing evidence (time-box directive,
coordinator 2026-07-15). If the lane's final report lands with a failure implicating the
adopted files, this ruling is reopened.

## Context

Owner spot-check of 50 Tier-A audio×track bootstrap labels FAILED the gate (29/50 true
contacts vs >=47/50; 12/29 timing off >=0.2s) -> auto-labeler REJECTED as label source;
event-head fine-tune blocked on label supply. Proven alternative: owner clip-review page
(50 rich rows in ~20 min). This lane productized that channel and staged tonight's session.

## What was adopted (files)

- `scripts/racketsport/build_event_review_session.py` — `sample` (stratified sampler) +
  `render` (ffmpeg half-speed +/-0.6s clips, atempo audio, self-contained review page).
- `scripts/racketsport/ingest_event_review_results.py` — fail-closed owner-export ->
  versioned reviewed-labels dataset with full provenance.
- `tests/racketsport/test_event_review_session.py`, `tests/racketsport/test_event_review_ingest.py`
  (15 tests incl. direct-CLI subprocess coverage with real ffmpeg fixtures).
- `scripts/racketsport/list_scaffold_tools.py` — one-line manager edit registering
  `event_review` in the label category (cleared the cross-track scaffold-inventory failure).
- Lane artifacts: `spec.md`, `session_manifest.json`, `INGEST_README.md`, this ruling.
  NOT committed: `pack/` (171MB clips, staged to Desktop), `log.txt`/`log2.txt`.

## Staged session (the deliverable)

`~/Desktop/event_labels_20260715/START_HERE.html` + 300 half-speed clips (302 files, 171MB).
Seed 20260715, generator event_review_session_v1_20260715
(sha256 76ad1bc016babd25ae3a5b0ece0a1c0937db9e555563486c1c8b143f8ec900fb), git_head at
staging 0fc14310f. Counts (source x stratum; every one of the 6 harvest sources drawn):

| stratum | 73VurrTKCZ8 | Ezz6HDNHlnk | HyUqT7zFiwk | _L0HVmAlCQI | wBu8bC4OfUY | zwCtH_i1_S4 | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| audio_onset | 14 | 24 | 27 | 16 | 19 | 20 | 120 |
| track_discontinuity | 10 | 14 | 15 | 11 | 12 | 13 | 75 |
| uniform_random | 13 | 21 | 23 | 15 | 16 | 17 | 105 |
| total | 37 | 59 | 65 | 42 | 47 | 50 | 300 |

Anti-bias design: score-tercile-balanced draws within source (not top-score), fresh simple
track-discontinuity detector recomputed from raw ball_track.json (bootstrap tier logic NOT
reused), uniform_random stratum signal-agnostic (the North Star §2.2 uniform-random audit
supply), presentation order shuffled and the page BLIND to stratum. Expected owner session:
75-120 min (measured 24 s/row rich baseline; 15 s/row decision-only floor).

Exclusions enforced and independently audited: 50-row protected eval seed (+/-0.75 s,
same clip; 190 audio + 350 track candidate hits avoided), pbvision_11min (never sampled),
universe restricted to data/online_harvest_20260706/rallies/ (renderer additionally
refuses non-universe paths, tested), anchors within [0.7, dur-0.7]. Source-disjoint split
bookkeeping copied from bootstrap manifest_v0 (train: 4 sources / validation:
HyUqT7zFiwk, zwCtH_i1_S4) recorded per row as suggested_split.

## Manager verification (all real exit codes, run by Track E manager)

1. Exclusion audit E1/E2/E3/E4, independent code: 0/0/0/0 violations, EXIT 0.
2. Same-seed determinism: two fresh `sample` runs byte-identical; vs staged manifest the
   ONLY diff is provenance `git_head` (HEAD moved 0fc14310f -> 4f77918 between runs);
   `generator_sha256` unchanged (script on disk == script that staged).
3. Pack: 300/300 mp4s ffprobed — 0 duration violations (all within [2.25,2.55] s),
   0 missing audio streams; render_report 300 requested/rendered/validated.
4. Page: ITEMS(300) join manifest with 0 mismatches; all referenced files exist; no
   'stratum' string anywhere; keyboard shortcuts present; inline JS `node --check` EXIT 0;
   export filename matches INGEST_README exactly.
5. Anchor convention regression: manifest's built-in check reproduces the 50 reviewed
   spot-check anchors from audio corrected_time_s 50/50 within one frame (max 0.042 s).
6. New tests: 15 passed, EXIT 0 (unpiped).
7. Scaffold index: test_scaffold_tool_index.py 3 passed, REAL_EXIT=0; scaffold + dead-code
   audits EXIT 0.
8. Ingest end-to-end dry-run against the REAL session manifest with a synthetic export:
   EXIT 0; provenance chain (seed/generator/manifest/results sha256), corrected_contact_pts_s
   math, px conversion at true source resolution, stratum re-join all correct.
9. Wide suite (by composition, not a fresh contended run): Track C wave-close pristine-tree
   evidence `runs/manager/trackC_20260715/waveclose_wide_suite.log` = 3684 passed / 24
   skipped / 1 failed, the single failure being test_scaffold_tool_index (caused by this
   lane's then-unregistered CLI) — now fixed and 3/3 green; import-isolation grep proves no
   file outside the lane references the two new modules. The lane's own in-flight wide-suite
   corroboration lands in report.json when its contended run finishes.

## Honest issues / open items

- Storage audit exits 1 REPO-WIDE for a PRE-EXISTING reason: 5 stale allowlist entries for
  `cvat_upload/w5_labelpack_20260708/packages/*.zip` deleted from disk 2026-07-09 (dir mtime
  evidence); zero unallowed files observed — this lane adds none. Needs a one-line allowlist
  reconciliation by whoever owns owner-package bookkeeping.
- localStorage key is `event_labels_20260715_results` (spec suggested `_answers_v2`);
  session-unique, cosmetic deviation accepted.
- Lane report.json pending at ruling time (contended wide suite; codex resume session also
  emitted tool-call errors). Superseded-by-manager-verification; reconcile on landing.
- 1h background task cap killed the original codex process at ~21:01; detached nohup resume
  succeeded (note: `codex exec` flags must precede the `resume` subcommand).

## BEST-STACK DELTA

(c) none — data-channel tooling; no model/weights/policy change. No best_stack.json entry.

## Next (for whoever ingests)

Owner exports `event_labels_20260715_results.json` -> run the exact command in
`INGEST_README.md` -> versioned reviewed-labels dataset with provenance under
`runs/lanes/owner_event_labels_20260715/reviewed_v2/`. These labels are owner-reviewed
bootstrap-era evidence under VERIFIED=0; ingest is not a promotion. Training-lane design
(event head) consumes them AFTER source-disjoint split policy is confirmed; uniform_random
rows double as the unbiased audit stratum and may be reserved for eval by a future ruling.

## Close-out addendum (2026-07-15, wind-down)

Lane report.json never landed: the resumed codex process was terminated cleanly by the
manager at coordinator wind-down after the deliverable was shipped, ruled, and committed
(d0ce58bdd). The ruling rests entirely on the manager verification battery above. Codex
session id 019f68df-5f28-7703-ad6e-bea1cf89e4a0 (logs: log.txt, log2.txt in this lane dir)
is recorded for forensic resume if ever needed.
