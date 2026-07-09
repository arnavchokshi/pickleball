# NS-06 pipeline speed and storage efficiency — 2026-07-09

## Outcome

Implemented and GPU-tested a scoped efficiency change set without touching the active
`process_video.py` / orchestrator ownership fence or `ios/**`.

- The selected mesh format reduces Wolverine's indexed replay mesh from **51,921,156 bytes to
  21,538,180 bytes (-58.5%)**.
- The mesh-index stage is **5.765s**, versus **22.068s** in the prior measured baseline (-73.9%).
- A real manifest-closed delivery bundle is **47.8% smaller** than the full Wolverine output and
  **26.9% smaller** than the full Outdoor output.
- The final full-pipeline candidate was 470.492s versus the matched 481.757s control, but this is
  one candidate sample. The attributable claims are the index-stage and byte-size results, not a
  global runtime promotion.
- `VERIFIED=0` remains binding. This work changes transport/representation efficiency, not model
  accuracy.

Machine-readable results: `results.json`. Exact compression sweep: `compression_level_sweep.json`.
Selected run evidence: `vm_evidence/v2_level6/`.

## What the baseline showed

The six-run Wolverine baseline was 483.610–497.460s, mean 489.423s. BODY consumed about 79% of the
wall clock, but steady H100 inference was only about 5.5s. The selected final candidate still spent:

| BODY phase | Seconds |
|---|---:|
| Array-native gate feed | 135.973 |
| Other BODY orchestration | 118.614 |
| Gates | 25.214 |
| Model load | 13.602 |
| Mesh index | 5.765 |
| Steady inference | 5.574 |

The next large runtime target is therefore CPU-side BODY preparation/gate construction, not a larger
GPU. Those surfaces overlap files already modified by other active lanes, so this lane did not race
them.

## Implemented changes

### Mesh representation and decoding

- Vectorized float-to-int16 quantization with scalar-path parity checks.
- Added `gzip_int16_delta_world_vertices_v2`: exact int16 deltas per player and window, with an
  absolute-frame fallback when shapes or int16 ranges do not permit a delta.
- Kept the existing v1 decoder path backward compatible.
- Selected gzip level 6 after a real 23-window sweep. Level 6 took 3.548s and produced 20,261,818
  bytes when recompressing the exact v2 raw bytes; level 9 took 12.957s and produced 20,373,414
  bytes. The level-9 default was rejected.
- Added deterministic player ordering so binary bytes and metadata stay aligned even when input
  players arrive out of order.

### Website / delivery storage

- Replaced the second full output-tree copy with a transitive replay-manifest closure.
- Preserved nested BODY mesh and replay-scene asset paths.
- Stream-compacted delivered JSON with bounded memory while preserving parsed JSON semantics.
- Rejected missing files, traversal, absolute escapes, symlink escapes, and destination collisions.
- Staged into a temporary directory and published the rewritten manifest last.
- Changed worker uploads to immutable per-job generations; incomplete generations reclaim only
  their own uploaded keys, and a lost publish response does not delete a possibly-live generation.
- Kept heartbeat coverage through package/upload work and preserved recursive local artifact copies.

### Diagnostic honesty

- Preserved the underlying BODY import exception in probe/HMR errors. This exposed the real invalid
  benchmark cause: under `Exclusive_Process`, the coordinator's CUDA context held the H100 while the
  self-dispatched BODY process tried to open it. The isolated benchmark VM was switched to compute
  mode `Default`; rejected runs were not counted.

## Matched GPU results

All rows used Wolverine, 300 video frames, 244 scheduled BODY frames, 705 scheduled player-frames,
23 mesh windows, stride 1, and a clean critical-file version stamp.

| Run | Pipeline wall | Shell wall | BODY wall | Index | Mesh chunks | Disposition |
|---|---:|---:|---:|---:|---:|---|
| Vectorized absolute v1 control | 481.757s | 485.991s | 373.276s | 5.557s | 50,662,377 B | Valid control |
| Delta v2, gzip 9 | 491.566s | 495.750s | 383.340s | 15.279s | 20,373,414 B | Valid measurement; rejected default |
| Delta v2, gzip 6 | **470.492s** | **474.655s** | **362.767s** | **5.765s** | **20,261,966 B** | Selected candidate |

The final candidate's index cost is only +0.208s versus the matched v1 control while its chunk bytes
are 60.0% lower. Its total indexed mesh, including index and faces JSON, is 58.5% lower.

The full pipeline remained `partial`: the clip's low-angle input-quality advisory remained below
acceptance and the scratch output path was outside the replay-manifest Vite allow root. BODY itself
ran, and the scoped BODY gates below passed.

## Gate and decoder checks

- Exact workload parity: 244/244 BODY frames and 705/705 player-frames.
- Full-clip BODY gate: pass; coverage 1.0; contact mesh coverage 1.0; no blockers or warnings.
- Grounding gate: pass; 0.019924665m versus the 0.03m threshold.
- Joint quality: `quality_checked_needs_accuracy_gate`.
- Mesh readiness: `mesh_available_needs_accuracy_gate`.
- Promotion blocker remains `missing_world_mpjpe_gate`.
- The largest observed matched joint-summary float change was about 1.44e-7, within the observed GPU
  nondeterminism scale; this is not accuracy proof.
- Validated all 23 final v2 windows, 705 frames, 641 delta frames, and 39,146,535 reconstructed int16
  values. A real gzip-v2 chunk also decoded through the TypeScript web decoder.

## Delivery-bundle measurements

| Real output | Full output | Curated compact bundle | Reduction | Files | Stage wall |
|---|---:|---:|---:|---:|---:|
| Wolverine | 117,268,905 B | 61,266,182 B | 47.8% | 34 | 1.674s |
| Outdoor | 312,183,386 B | 228,278,081 B | 26.9% | 37 | 5.320s |

For Wolverine, the old raw + full-artifact + duplicated-full-bundle layout modeled to 240,997,002
bytes. The new exact-file layout models to 182,841,215 bytes, saving 58,155,787 bytes (24.1%) while
retaining the full diagnostic artifact generation. This is an exact local artifact calculation, not
a claim that production S3 has already been migrated.

## Verification

- `tests/render_service`: 137 passed.
- BODY/index/probe/batch focused suite: 63 passed, 1 intentional skip.
- Web replay: 235 passed.
- TypeScript typecheck: passed.
- Real v2 chunk decoder and all-window reconstruction: passed.
- Wolverine and Outdoor staged JSON: parsed successfully and matched source JSON semantics.
- Dead-code audit: passed, 555 Python sources, zero unknown.
- Scaffold index: exited 0; 280 tools, all with direct/related tests.
- `git diff --check`: passed.
- Web production build remains blocked by an unrelated dangling active-lane fixture symlink at
  `web/replay/public/critique/wolverine_mixed_0200_mid_steep_corner/source.mp4`; this lane did not
  delete or replace another agent's artifact.
- Repo-wide storage-policy audit still fails on concurrent generated caches and missing allowlisted
  source packages. They were preserved because other agents are active.

## Deployment and remaining work

- Deploy the v2-capable web viewer before, or atomically with, the v2 producer. The new viewer reads
  both v1 and v2; an old viewer will reject v2.
- iOS currently does not consume `body_mesh_index`; native USDZ/payload optimization remains a
  separate iOS-owned task.
- Add state-aware retention/lifecycle cleanup for superseded immutable S3 generations. Immediate
  deletion was intentionally avoided because a lost completion response makes publish state
  ambiguous.
- Profile and then optimize the 135.973s array-feed and 118.614s BODY orchestration phases under the
  owning integration lane.

## GPU teardown

`pickleball-h100-ns06eff-bx1` (a3-highgpu-1g H100 Spot, asia-southeast1-b) ran for about 1.71 hours,
had zero preemptions, was deleted at 2026-07-09T23:04:46Z, and was list-confirmed absent. An initial
asia-southeast1-c create attempt was rejected for stockout and created no lingering instance.
