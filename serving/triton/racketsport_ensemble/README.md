# Racketsport Triton Ensemble Scaffold

This directory is intentionally documentation-only until EVAL-0 locks model
variants. The CPU-only serving readiness report is generated with:

```bash
.venv/bin/python scripts/racketsport/build_serving_manifest.py \
  --manifest models/MANIFEST.json \
  --out runs/serving/serving_manifest.json
```

The report maps `models/MANIFEST.json` entries into two serving tiers:

- `offline_deep`: server GPU deep-tier components for accurate body mesh,
  ball/racket, and physics runtime readiness.
- `live_light`: server fallback/light-tier components for preview-path serving
  readiness. On-device Apple Vision/Core ML assets remain outside
  `models/MANIFEST.json` unless a later client task records them there.

The manifest builder is CPU-only. It does not start Triton, download models,
probe checkpoint existence on the local host, use a GPU, mutate
`models/MANIFEST.json`, or mark ENV/EVAL gates complete. EVAL-0 benchmark and
approval artifacts still decide which candidate variants become final.
