# Owner event review ingest

After the owner exports the completed page, run exactly:

```bash
.venv/bin/python scripts/racketsport/ingest_event_review_results.py --results ~/Downloads/event_labels_20260715_results.json --manifest runs/lanes/owner_event_labels_20260715/session_manifest.json --out-dir runs/lanes/owner_event_labels_20260715/reviewed_v2 --root .
```

## Staged session counts

| stratum | 73VurrTKCZ8 | Ezz6HDNHlnk | HyUqT7zFiwk | _L0HVmAlCQI | wBu8bC4OfUY | zwCtH_i1_S4 | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| audio_onset | 14 | 24 | 27 | 16 | 19 | 20 | 120 |
| track_discontinuity | 10 | 14 | 15 | 11 | 12 | 13 | 75 |
| uniform_random | 13 | 21 | 23 | 15 | 16 | 17 | 105 |
| total | 37 | 59 | 65 | 42 | 47 | 50 | 300 |

The ingest is fail-closed: contact decisions require normalized x/y plus source-time dt,
none/unclear must carry no contact fields, label IDs and presentation rows must join this
manifest exactly, and unanswered rows are listed in `dataset_manifest.json`. These labels
remain owner-reviewed bootstrap-era evidence under `VERIFIED=0`; ingest is not a promotion.

## Ingest operator notes (2026-07-16)

- Partial exports are expected (mid-session "Save progress" button on the owner-live page):
  ingest EXACTLY ONE results file — the newest/most-complete export (supersede by answer
  count, tie-break mtime). Partials ingest cleanly; unanswered rows are listed in
  dataset_manifest.json (verified: 4/300 partial, EXIT 0, unanswered=296).
- Spot-check dt outliers on EARLY presentation rows (pre-hotfix dt-suspect: click-toggled
  playback could corrupt dt before the page fix landed).
- Multi-event windows: owner labels the event nearest clip center (paddle priority
  near-center) — valid typed contacts; per-stratum precision stats carry the
  validated-by-nearby-different-event caveat.
