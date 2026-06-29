# PB-MAT Runtime Seam

Date: 2026-06-28

PB-MAT is an experimental ball-model runtime seam, not the approved production
default. The current MVP/default registry path remains TrackNetV3 plus available
review candidates until PB-MAT has a trained checkpoint, registry entry, and BALL
gate evidence. This document covers the runtime artifact seam only. It does not
claim a trained PB-MAT checkpoint exists, does not fine-tune TrackNet, and does
not promote BALL to verified.

## Prediction Artifact

The PB-MAT runtime or export job should write:

```json
{
  "schema_version": 1,
  "artifact_type": "racketsport_pbmat_predictions",
  "source_mode": "pbmat_json",
  "fps": 60.0,
  "image_size": [1920, 1080],
  "output_stride": 4,
  "model": {
    "id": "pbmat_hybrid",
    "checkpoint_sha256": "sha256-value"
  },
  "frames": [
    {
      "frame_index": 12,
      "t": 0.2,
      "visibility_score": 0.91,
      "blur_score": 0.3,
      "occlusion_score": 0.1,
      "selected_candidate": 0,
      "candidates": [
        {
          "xy": [101.0, 202.0],
          "confidence": 0.72,
          "source": "coarse_heatmap",
          "refined_xy": [103.25, 204.5],
          "refined_confidence": 0.86
        }
      ]
    }
  ]
}
```

Rules:

- `xy` and `refined_xy` are source-video pixel coordinates.
- `visibility_score` gates whether a frame becomes visible in `ball_track.json`.
- `refined_xy` wins over coarse `xy` when present.
- `ball_points.json` and click-review outputs remain held-out labels only and
  must not be read by PB-MAT runtime paths.
- Metadata written by the adapter is `not_ground_truth=true` and
  `verified=false` until a real BALL gate passes.

## Conversion Command

```bash
python scripts/racketsport/run_pbmat_ball.py \
  --predictions-json runs/pbmat/<clip>/pbmat_predictions.json \
  --out runs/pbmat/<clip>/ball_track.json \
  --metadata-out runs/pbmat/<clip>/pbmat_run.json \
  --visibility-threshold 0.5
```

The output `ball_track.json` uses `source="pbmat"` and can be passed to existing
overlay, benchmark, virtual-world, and stage-runner tools.

## Heatmap/Crop Helper API

`threed.racketsport.pbmat_adapter.decode_pbmat_heatmap_candidates(...)` decodes
top-K stride-space heatmap peaks with optional subpixel offsets.

`threed.racketsport.pbmat_adapter.remap_crop_refined_xy(...)` maps crop-local
refined points back to source-video coordinates while clipping to the crop and
frame bounds.

These helpers are CPU-only on purpose. A PyTorch/TensorRT PB-MAT model can use
them as reference behavior while the GPU runtime remains isolated.

## Benchmarking

After conversion, compare PB-MAT like any other no-click candidate:

```bash
RUN_ROOT=runs/eval0/prototype_gate_h100_v2

python scripts/racketsport/benchmark_ball_trackers.py \
  --run-root "$RUN_ROOT" \
  --review-root "$RUN_ROOT/ball_click_review_30" \
  --clip burlington_gold_0300_low_steep_corner \
  --clip wolverine_mixed_0200_mid_steep_corner \
  --clip outdoor_webcam_iynbd_1500_long_high_baseline \
  --clip indoor_doubles_fwuks_0500_long_mid_baseline \
  --candidate "pbmat_hybrid=tracknet_smoke_0000_0010/pbmat_ball_track.json" \
  --out-json "$RUN_ROOT/ball_tracker_benchmark/benchmark_summary_pbmat.json" \
  --out-md "$RUN_ROOT/ball_tracker_benchmark/benchmark_summary_pbmat.md"
```

This is still `scored_not_gate_verified`. BALL verification requires the
representative labeled gate described in `IMPLEMENTATION_PHASES.md`.
