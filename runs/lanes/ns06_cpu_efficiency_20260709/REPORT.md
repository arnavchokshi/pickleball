# NS-06 CPU BODY preparation efficiency — 2026-07-09

## Outcome

Adopted a scoped CPU-preparation candidate after a matched warm H100 run.

- Full pipeline: **502.810s → 366.810s**, saving **136.000s (27.0%)**.
- Measured BODY phase: **384.035s → 241.090s**, saving **142.945s (37.2%)**.
- Array-native gate feed: **147.144s → 78.071s**, saving **69.073s (46.9%)**.
- Residual BODY work: **127.194s → 53.750s**, saving **73.444s (57.7%)**.
- Peak BODY-process RSS: **15.34 GiB → 15.18 GiB**. The vector-only intermediate was faster but
  regressed to **19.98 GiB**; topology interning removed **4.80 GiB (24.0%)** from that intermediate.

The workload, compile time, steady inference, gates, mesh schedule, and version checks matched. This
is a strong scoped efficiency result, not a model-accuracy promotion. `VERIFIED=0` remains binding.

Machine-readable results: `benchmark_results.json`. Selected evidence: `vm_evidence/topology_final/`.

## What CPU BODY preparation meant

The H100 was not the long pole. Before this work, steady inference for all 705 player-frames was about
5.8s, while Python spent minutes converting and copying dense arrays:

- a preserved profile made roughly 453 million Python calls during normalization;
- the same 36,874-triangle topology was parsed and copied into hundreds of frame records;
- 18,439 vertices per mesh were repeatedly converted between arrays and nested Python lists;
- world grounding and gate/readiness construction scanned those large records again.

Live `/proc` sampling confirmed the representation problem. The vector-only process reached **19.98
GiB RSS** while producing an approximately 20.26 MB compressed replay mesh.

## Implemented

### Vectorized validation and translation

- Rectangular finite joint/vertex arrays validate through NumPy C loops.
- Integer topology bounds and shape checks use vectorized reductions.
- Dense translations use NumPy while malformed/ragged/boolean inputs retain the scalar error path.
- CPU Torch tensors requiring gradients fall back through the existing detach/cpu path.

Review caught and fixed three candidate regressions before final benchmarking: mixed booleans,
malformed empty topology rows, and an uncaught `requires_grad` conversion error.

### Exact transform ruling

A trial BLAS camera-to-world transform was rejected. A valid crafted rotation moved one coordinate
from exactly 1.9755m to 1.9754999999999998m, changing replay quantization from 1976mm to 1975mm.
The selected candidate retains the legacy scalar accumulation order and has a regression fixture for
that half-millimetre boundary.

### Static topology interning

- Added an explicit clip-scoped topology interner; no unbounded global cache or object-ID cache.
- Validated topology is immutable, list-compatible, pickle/JSON/Pydantic compatible, and content
  identified with SHA-256.
- World grounding, smoothing, and common-topology selection reuse one canonical topology.
- Vertex-count changes recheck bounds; changed or malformed topology still fails.

After the orchestrator ownership fence released, the integration follow-up wired one interner into
the BODY normalization loop. Every normal pipeline BODY run now creates one scope per clip and passes
it to every player-frame; there is no process-global cache. The automatic world-stage path reuses the
same immutable topology representation downstream.

The follow-up also resolved an older selection-source contradiction: the constructor used shared
array-native BODY compute by default while `best_stack.json` still called it dormant. Best-stack
revision 10 now selects the shared array-native path, and local plus generated remote runners resolve
and pass that setting explicitly. Pipeline summaries record the resolved selection.

## Matched H100 results

All rows used one isolated `a3-highgpu-1g` H100 Spot VM, Wolverine, 300 video frames, 244 BODY
frames, 705 player-frames, 23 mesh windows, clean critical version stamps, and GPU compute mode
`Default`.

| Run | Pipeline | BODY phase | Array feed | Other BODY | Compile | Inference | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Warm control `bdeebfa04` | 502.810s | 384.035s | 147.144s | 127.194s | 31.019s | 5.791s | 15.34 GiB |
| Vector `6eeaead70` | 418.293s | 290.651s | 126.878s | 53.068s | 30.996s | 5.794s | 19.98 GiB |
| Topology final `cf5295182` | **366.810s** | **241.090s** | **78.071s** | **53.750s** | **30.824s** | **5.741s** | **15.18 GiB** |

A cold control took 518.537s with 54.715s compile warm-up and was excluded from the matched claim.
The warm control and both candidates had approximately 31s compile warm-up, so the selected speedup
is not a compile-cache artifact.

Only one matched warm sample per implementation was run. The effect is large and phase-attributed,
but the global pipeline default should still retain `VERIFIED=0` until the normal integration owner
runs the broader frozen gate.

## Output and gate parity

- BODY ran for 244/244 frames and 705/705 player-frames in every counted run.
- Full-clip BODY gate passed; coverage and contact mesh coverage were both 1.0.
- Grounding passed; final max foot slide was 0.0199248566m.
- Joint status remained `quality_checked_needs_accuracy_gate` with no quality blockers.
- Mesh status remained `mesh_available_needs_accuracy_gate`; `missing_world_mpjpe_gate` still blocks
  promotion.
- The static face file SHA-256 was identical across control and both candidates.
- Mesh-index semantics were exact after excluding per-run GPU joint-confidence noise and resulting
  compressed byte sizes. Final versus control summary changes were below 1e-7m; root speed and anchor
  residual were exact.
- The pipeline remained `partial` because input-quality and scratch-path manifest stages were already
  degraded; BODY itself ran and passed its scoped gates.

## Storage outcome

This lane reduces transient CPU memory and keeps the selected evidence compact. The earlier committed
NS-06 delivery work in `runs/lanes/ns06_efficiency_20260709/REPORT.md` remains the website/iPhone
storage result:

- indexed Wolverine replay mesh: **-58.5%**;
- curated Wolverine delivery bundle: **-47.8%**;
- curated Outdoor delivery bundle: **-26.9%**.

For this follow-up, redundant raw control evidence was discarded. The selected final's three larger
generated JSON artifacts were checksumed and summarized instead of duplicated. The lane evidence was
reduced from 7.5 MB to about 0.3 MB before commit.

## What to do next

1. Keep vertices as NumPy/CUDA arrays through grounding and shared reductions; do not convert 705
   dense meshes into Python lists until the final quantized transport boundary.
2. Batch GPU world transforms, reductions, int32 deltas, range checks, and int16 quantization only if
   downstream int16 bytes pass the half-millimetre regression and same-input replay gate.
3. Benchmark `--body-schedule overlap`; the selected run still spent about 79s in BALL arc solving
   before BODY, so overlap may hide useful wall time. Do not change the default without CPU/RAM
   contention and output-parity proof.
4. Prototype a persistent preemption-safe BODY worker. It could remove about 14s model load plus 31s
   compile warm-up, but needs code/model/config invalidation and GPU-context lifecycle proof.
5. Profile/fuse gates that scan the same compact summaries. Do not remove full-clip, grounding,
   quality, or readiness gates to hit latency.

L2 may intentionally omit BODY as a product-tier choice. L3 should not skip required mesh frames,
interpolate them, or drop gates merely for speed.

## Verification and teardown

- Always-on integration follow-up: exact staged snapshot 16 passed; wider BODY/best-stack 143
  passed, process-pipeline 142 passed, and remote-dispatch 76 passed.
- Local topology/body/schema suite: 131 passed, one expected runtime warning.
- Broader candidate suite: 169 passed, 1 skipped, 2 warnings.
- Final commit on VM Python 3.10: 62 passed, one upstream warning.
- Dead-code audit: pass, 555 Python sources, zero unknown.
- Scaffold index: pass, 280 tools, zero missing direct or related tests.
- Targeted Ruff still reports the pre-existing `worldhmr.py` F402 loop-variable shadow outside this
  diff; no unrelated cleanup was folded into this lane.
- Repo-wide storage audit still fails on concurrent generated caches and missing allowlisted source
  packages, with no unknown large files. Other agents' active artifacts were preserved.

`pickleball-h100-ns06cpu-bx1` was created at 2026-07-09T23:38:34Z, had no observed preemption,
was deleted after the final run, and was list-confirmed absent by 2026-07-10T00:34:37Z.
