# DATA-1 Test Clip Coverage Report

`scripts/racketsport/report_testclip_coverage.py` is a read-only triage CLI for the
registered DATA-1 test clips when they are present under `data/testclips`. This
checkout may legitimately have no `data/` root; in that state the report exits
zero with `root_exists=false`, `total_clips=0`, and `ready=false`.

It reuses `threed.racketsport.testclips.build_testclip_manifest` for clip readiness,
label counts, metadata validation, and coverage-matrix gaps. Optional frame-pack
counts are read from `runs/label_frames/<clip>/label_frame_manifest.json`.

## JSON Summary

```bash
python scripts/racketsport/report_testclip_coverage.py \
  --root data/testclips \
  --frames-root runs/label_frames
```

The CLI prints JSON by default and exits zero because it is a report, not a gate.
The top-level `ready` field remains false until both label readiness and the
coverage matrix are satisfied.

## Markdown Triage

```bash
python scripts/racketsport/report_testclip_coverage.py \
  --root data/testclips \
  --frames-root runs/label_frames \
  --markdown-out docs/racketsport/testclip_coverage_report.generated.md
```

The Markdown report lists:

- current clip, metadata, label, and frame-pack counts
- missing matrix coverage
- per-label readiness counts
- per-clip missing-label and extracted-frame counts

The report never writes label files and never marks clips ready. Dataset labels must
still be created intentionally under `data/testclips/<clip>/labels`.
