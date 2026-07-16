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
