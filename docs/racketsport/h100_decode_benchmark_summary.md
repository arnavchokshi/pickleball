# H100 Decode Benchmark Summary

Recorded on 2026-06-27 from `body4d-gcp-prod` (`a3-highgpu-1g`, H100 80 GB) using real ignored DATA-1 clips in `/workspace/pickleball/data/testclips`.

This report is evidence for ENV-4 backend triage only. It does not set a global decode default; backend choice remains empirical per real clip set.

## Fastest Backend By Clip

| Clip | Backend | Decode FPS | Realtime Factor |
|---|---|---:|---:|
| `burlington_gold_0300_low_steep_corner` | `cpu` | 2063.870 | 33.524 |
| `ppa_austin_md_qf_1200_long_high_baseline` | `cpu` | 1296.374 | 43.042 |
| `ppa_singles_q4zd_0500_long_high_baseline` | `cpu` | 2068.100 | 34.468 |

## Aggregate By Backend

| Backend | Runs | Clips | Mean Decode FPS | Mean Realtime Factor | Total Frames |
|---|---:|---:|---:|---:|---:|
| `cpu` | 3 | 3 | 1809.448 | 37.012 | 86652 |
| `cuda` | 3 | 3 | 898.987 | 21.953 | 86652 |

## Notes

- Benchmarks: 6 total runs over 3 clips.
- All sampled clips were 1920x1080 H.264.
- CUDA decode was not faster for this sample; the earlier synthetic-short-clip result was directionally consistent for 60 fps clips.
- Keep re-running `scripts/racketsport/benchmark_decode.py` and `scripts/racketsport/summarize_decode_benchmarks.py` as the DATA-1 matrix grows, especially for 4K, 120 fps, and 240 fps captures.
