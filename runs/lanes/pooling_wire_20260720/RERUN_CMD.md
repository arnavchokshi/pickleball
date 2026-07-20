# pooling_wire_20260720 GPU replay command

Run exactly once on the reviewed GPU checkout after syncing the owned,
uncommitted files and confirming their two-sided SHA-256 hashes:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/racketsport/process_video.py \
  --video data/pbv_replay_20260720/xkadsq9bli3h/max.mp4 \
  --clip xkadsq9bli3h \
  --out runs/lanes/pooling_wire_20260720/gpu_replay \
  --max-players 4 \
  --body-local \
  --allow-auto-court-corners-preview \
  --court-line-evidence-pooling \
  --no-ball-arc \
  --verify-viewer \
  --json
```

`--no-ball-arc` is the existing typed skip for the unrelated stage that
exceeded the prior Drill replay's 20-minute cap. The replay still exercises
CAL pooling, unchanged-bar readiness, TRK, BODY, placement/world, packaging,
and viewer verification. `VERIFIED=0` remains binding regardless of this
single clip-specific replay.
