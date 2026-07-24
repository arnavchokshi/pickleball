# Multimodal event dataset coverage (WS3.2)

Measurement-only artifact; VERIFIED=0 stands. Deterministic rebuild:
`scripts/racketsport/build_multimodal_event_dataset.py build` with the inputs
pinned in `MANIFEST.sha256.json`.

Window: 64 frames, label at bin 32. Source: owner_102_manifest.json config.window_frames == 64 (E-v2 64-frame context); both manifests carry num_frames == 64 on every row.

## Per-modality coverage

| label set | rows | records | modality | artifact-bound | signal in window |
|---|---|---|---|---|---|
| owner | 102 | 102 | audio_onsets_v2 | 102/102 (100.0%) | 99/102 (97.1%) |
| owner | 102 | 102 | ball_inflections | 102/102 (100.0%) | 56/102 (54.9%) |
| owner | 102 | 102 | wrist_velocity_peaks | 0/102 (0.0%) | 0/102 (0.0%) |
| teacher | 1189 | 49 | audio_onsets_v2 | 49/49 (100.0%) | 48/49 (98.0%) |
| teacher | 1189 | 49 | ball_inflections | 0/49 (0.0%) | 0/49 (0.0%) |
| teacher | 1189 | 49 | wrist_velocity_peaks | 0/49 (0.0%) | 0/49 (0.0%) |

Rates are over EMITTED records; rows never emitted are in the unbuildable ledger below.

## Cue proximity to the label (ceiling predictor)

Fraction of event records with at least one cue of the modality within ±2
frames of the label frame (the frozen judge's macro-F1@±2 tolerance).

| label set | modality | records with cue within ±2 | rate |
|---|---|---|---|
| owner | audio_onsets_v2 | 29/59 | 49.2% |
| owner | ball_inflections | 19/59 | 32.2% |
| owner | wrist_velocity_peaks | 0/59 | 0.0% |
| teacher | audio_onsets_v2 | 14/49 | 28.6% |
| teacher | ball_inflections | 0/49 | 0.0% |
| teacher | wrist_velocity_peaks | 0/49 | 0.0% |

## Split table

| label set | train | val | quarantined |
|---|---|---|---|
| owner | 61 | 41 | 0 |
| teacher | 49 | 0 | 0 |

## Assertions (measured while building)

- audio-only teacher events: 0 (build refuses non-zero)
- protected seed ids checked: 50; identity matches in records: 0
- protected train-window overlaps: 0 (build refuses non-zero)
- protected val-window overlaps measured: 0
- teacher rows in val: 0 (build refuses non-zero)

## Cue provenance

| family | audio_onsets_v2 artifact | sha256 (first 12) |
|---|---|---|
| 73VurrTKCZ8 | runs/lanes/ball_audio_repair2_20260722/raw_audio_onsets/73VurrTKCZ8.audio_onsets_v2.json | abd3ee5be409 |
| Ezz6HDNHlnk | runs/lanes/ball_audio_repair2_20260722/raw_audio_onsets/Ezz6HDNHlnk.audio_onsets_v2.json | 4f42b911efc3 |
| HyUqT7zFiwk | runs/lanes/ball_audio_repair2_20260722/raw_audio_onsets/HyUqT7zFiwk.audio_onsets_v2.json | 72e2fd427a1e |
| _L0HVmAlCQI | runs/ball_lane_20260723/mm_dataset/cue_artifacts/_L0HVmAlCQI.audio_onsets_v2.json | d96a9c2420be |
| wBu8bC4OfUY | runs/lanes/ball_audio_repair2_20260722/raw_audio_onsets/wBu8bC4OfUY.audio_onsets_v2.json | 8d117824d1cf |
| xkadsq9bli3h | runs/lanes/ball_audio_repair2_20260722/raw_audio_onsets/xkadsq9bli3h.audio_onsets_v2.json | f4f6ef0e571a |
| zwCtH_i1_S4 | runs/ball_lane_20260723/mm_dataset/cue_artifacts/zwCtH_i1_S4.audio_onsets_v2.json | bdadd7ef4070 |

- audio media-binding counts per label set: {"owner": {"verified_local_media": 102}, "teacher": {"row_source_video_sha256": 49}}
- ball_inflections artifacts (per clip, built from WASB prelabel ball tracks): 25
- wrist_velocity_peaks: no skeleton3d.json artifact exists locally for any labeled source video; the wrist_velocity_peaks builder requires skeleton3d upstream, so the wrist modality is masked no_artifact for every row

Per-family and per-fps-regime breakdowns live in coverage.json under
`label_sets.*.per_family` and `label_sets.*.per_fps_regime`; regime beyond
family/fps is not determinable from the consumed artifacts.

## Unbuildable rows

| label set | family | reason | rows |
|---|---|---|---|
| teacher | 143sf3gdwxsa | missing_media | 334 |
| teacher | 98z43hspqz13 | missing_media | 256 |
| teacher | st0epgnab7dr | missing_media | 226 |
| teacher | td2szayjwtrj | missing_media | 156 |
| teacher | utasf5hnozwz | missing_media | 168 |

Total unbuildable rows: 1140 (full per-row ledger in coverage.json `unbuildable_rows`).

## Clip-to-source time mapping (ball modality)

| clip | clip_start_s | matched onsets | median residual (s) | verified |
|---|---|---|---|---|
| 73VurrTKCZ8_rally_0002 | 54.0 | 123 | -0.00468 | True |
| 73VurrTKCZ8_rally_0005 | 188.0 | 36 | -0.004963 | True |
| 73VurrTKCZ8_rally_0008 | 248.0 | 129 | 0.000374 | True |
| Ezz6HDNHlnk_rally_0002 | 60.0 | 65 | 0.003063 | True |
| Ezz6HDNHlnk_rally_0003 | 215.0 | 10 | -0.024938 | True |
| Ezz6HDNHlnk_rally_0004 | 239.0 | 150 | 0.001809 | True |
| Ezz6HDNHlnk_rally_0005 | 499.5 | 2 | -0.007006 | False |
| Ezz6HDNHlnk_rally_0006 | 511.0 | 98 | 0.006946 | True |
| Ezz6HDNHlnk_rally_0008 | 757.5 | 16 | -0.015581 | True |
| HyUqT7zFiwk_rally_0001 | 0.0 | 2943 | 0.0 | True |
| _L0HVmAlCQI_rally_0001 | 0.0 | 81 | 0.0 | True |
| _L0HVmAlCQI_rally_0002 | 63.0 | 6 | 0.019585 | True |
| _L0HVmAlCQI_rally_0003 | 83.0 | 6 | -0.000873 | True |
| _L0HVmAlCQI_rally_0006 | 166.0 | 4 | -0.021689 | False |
| _L0HVmAlCQI_rally_0007 | 191.5 | 9 | 0.001187 | True |
| _L0HVmAlCQI_rally_0009 | 261.0 | 6 | -0.006436 | True |
| _L0HVmAlCQI_rally_0010 | 290.0 | 4 | 7.3e-05 | False |
| _L0HVmAlCQI_rally_0011 | 308.5 | 6 | -0.003409 | True |
| _L0HVmAlCQI_rally_0014 | 407.0 | 6 | -0.020904 | True |
| _L0HVmAlCQI_rally_0017 | 490.0 | 6 | 0.033068 | True |
| _L0HVmAlCQI_rally_0019 | 559.0 | 3 | 0.011876 | False |
| wBu8bC4OfUY_rally_0001 | 0.0 | 923 | 0.0 | True |
| wBu8bC4OfUY_rally_0002 | 430.0 | 30 | 0.012304 | True |
| wBu8bC4OfUY_rally_0003 | 513.5 | 7 | 0.010048 | True |
| zwCtH_i1_S4_rally_0001 | 0.0 | 553 | 0.0 | True |

