# TrackNetV4 Ball Runner Seam

This is an integration seam for converting TrackNetV4 prediction CSVs into the
repo's `BallTrack` contract. It does not train TrackNetV4, vendor the external
repo, or prove BALL gates.

Primary upstream reference: https://github.com/TrackNetV4/TrackNetV4

## CSV Conversion

The adapter accepts the official TrackNetV4 prediction CSV format:

```csv
Frame, Visibility, X, Y
0,1,321,240
1,0,-1,-1
```

It also accepts a simple lowercase form:

```csv
frame,x,y
0,321,240
1,
2,-1,-1
```

Accepted column aliases are:

- frame: `Frame`, `frame`, `frame_id`, `frame_index`
- visibility: `Visibility`, `visibility`, `visible`, `is_visible`
- x: `X`, `x`, `x_px`, `ball_x`
- y: `Y`, `y`, `y_px`, `ball_y`

If visibility is absent, blank coordinates or `-1,-1` are treated as invisible.
Invisible rows with blank coordinates are written as `xy=[0.0, 0.0]` because the
current `BallTrack` schema requires an image coordinate on every frame.

Example:

```bash
python scripts/racketsport/run_tracknetv4_ball.py \
  --predictions-csv runs/tracknetv4/clip_predictions.csv \
  --fps 60 \
  --out runs/tracknetv4/ball_track.json \
  --metadata-out runs/tracknetv4/ball_track_run.json
```

CSV conversion metadata is always:

- `not_ground_truth=true`
- `verified=false`
- `source_mode=tracknetv4_csv`

## External TrackNetV4 Run

By default, the runner mirrors the upstream command shape:

```bash
python src/predict.py \
  --video_path <VIDEO> \
  --model_weights <MODEL_PATH> \
  --output_dir <OUTPUT_DIR> \
  --queue_length <N>
```

Local command:

```bash
python scripts/racketsport/run_tracknetv4_ball.py \
  --video data/testclips/example.mp4 \
  --checkpoint /path/to/model_final.keras \
  --tracknetv4-repo /path/to/TrackNetV4 \
  --prediction-dir runs/tracknetv4/example \
  --fps 60 \
  --out runs/tracknetv4/example/ball_track.json \
  --metadata-out runs/tracknetv4/example/ball_track_run.json
```

The runner validates these paths before invoking subprocess:

- `--tracknetv4-repo` must exist.
- Without `--command`, `--tracknetv4-repo/src/predict.py` must exist.
- `--video` must be a file.
- `--checkpoint` must be a file.

If the upstream repo writes an unusual CSV name, pass either a path relative to
`--prediction-dir` or an absolute path:

```bash
--expected-csv predictions.csv
```

For wrappers or nonstandard repo layouts, pass a shlex command template:

```bash
--command "{python} {repo}/tools/predict.py --video {video} --weights {checkpoint} --out {output_dir}"
```

Available placeholders are `{python}`, `{repo}`, `{predict_py}`, `{video}`,
`{checkpoint}`, `{output_dir}`, and `{queue_length}`.

## Upstream Snapshot Caveats

The H100 currently has the public TrackNetV4 repo cloned at
`/workspace/TrackNetV4`. In that snapshot:

- `docs/RESULT.md` lists model download entries, but the links are placeholders
  (`#`) rather than usable checkpoint URLs.
- `src/predict.py` references custom-layer names such as
  `MotionIncorporationLayerV1`, `MotionIncorporationLayerV2`,
  `CombineOutputs`, and `MotionFramesInput` that are not imported or defined in
  the repo snapshot. A real run will need either an upstream fix, a local patch
  to the custom object map, or a custom wrapper passed with `--command`.

## Verification Semantics

The sidecar metadata always records `not_ground_truth=true`. By default it also
records `verified=false`, even after a subprocess succeeds, because a fake repo
or local wrapper can write a schema-valid CSV without proving that real
TrackNetV4 inference ran.

Only use `--mark-real-run-succeeded` after a real external TrackNetV4 command
has completed successfully and produced the CSV being converted. Even then, the
metadata means only that the external run completed; it is not a BALL gate, not
human-reviewed ground truth, and not an accuracy/F1/contact-timing verification.
