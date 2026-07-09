# NS-06 CPU/GPU efficiency research — 2026-07-09

Status: implementation candidate; `VERIFIED=0`.

## Measured cause

The selected Wolverine run spent 118.614s in residual BODY work and 135.973s in
array-native gate preparation while steady H100 inference took 5.574s.

A read-only `cProfile` replay on 359 preserved real SAM3D player-frames found:

| Work | Profiled time |
|---|---:|
| Output normalization | 131.26s |
| Reparse identical 36,874-triangle topology | 75.35s |
| Convert dense vertices through Python lists | 46.76s |
| Apply camera translation | 7.89s |
| World/joint/mesh postchain | 150.95s |
| Ground each dense frame | 112.56s |
| Dense camera-to-world rotation | 40.56s |
| Second dense-vector validation/copy | 37.15s |
| Second topology parse | 31.26s |

The profile contains 453 million Python calls during normalization. This is
data conversion and repeated validation, not useful model compute.

## Implemented first candidate

- Rectangular finite joint/vertex arrays validate and convert through NumPy C
  loops, with the old scalar validator retained for malformed/ragged inputs.
- Integer face topology validates with vectorized bounds checks, retaining the
  old scalar error path for non-integer/ragged inputs.
- Dense translation uses float64 NumPy operations. A trial NumPy/BLAS
  camera-to-world rotation was rejected after review found a valid crafted
  half-millimetre boundary where it changed an int16 replay coordinate by
  1 mm; the candidate retains the legacy scalar accumulation order.
- Common topology comparisons use NumPy equality instead of rebuilding three
  Python integers for every triangle.

The initial direct legacy/candidate replay in `local_replay_parity.json`
preserved exact normalization, face topology, int16 mesh bytes, and BODY
metrics on its real artifact. Review then found a crafted quantization-boundary
counterexample outside that artifact, so the unsafe matrix multiply was
reverted and covered by a regression test before the H100 candidate run.

## Research-backed next order

1. Eliminate CPU work before adding cores. Keep large arrays in NumPy or CUDA
   tensors and avoid `.cpu()`, `.numpy()`, `.item()`, and Python branches inside
   GPU loops because they can synchronize the device. See the
   [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
   and [CUDA data-transfer guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#data-transfer-between-host-and-device).
2. If this NumPy candidate passes the full pipeline, move world transforms,
   reductions, int32 deltas, range checks, and final int16 quantization into one
   batched GPU postprocessor. Transfer only compact deltas and small summaries.
3. Pipeline GPU batch N+1 with CPU compression/write of batch N using bounded
   reusable pinned buffers and explicit synchronization. The
   [PyTorch transfer guide](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)
   warns that pinning inside the hot loop is itself blocking.
4. Fuse gates that scan the same arrays. Parallel full-array scans can lose to
   memory-bandwidth contention; parallelize only independent small-summary
   consumers after one shared reduction.
5. Use threads only for NumPy operations that release the GIL. Use long-lived
   processes and shared memory for genuinely Python-heavy work; do not pickle
   dense meshes through queues. See [NumPy thread safety](https://numpy.org/doc/2.3/reference/thread_safety.html)
   and [Python shared memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html).
6. A persistent warm BODY worker can remove the measured ~13.6s model load and
   ~23.9s compile warmup, but requires code/model/config invalidation and
   preemption-safe lifecycle proof.

## Skips and parallel work

Safe now:

- Keep the already-selected default that omits `smpl_motion.json` and
  `body_mesh.json` monoliths.
- Keep tier-2 dense mesh vertices omitted.
- Validate static face topology once rather than once per frame.
- Keep diagnostics outside delivery bundles.

Not safe for L3 merely to hit latency:

- Dropping BODY quality/full-clip/grounding/readiness gates.
- Reducing required computed mesh frames without the same frozen accuracy and
  replay gate.
- Replacing exact BODY outputs with interpolation without an independent gate.

L2 may omit BODY while L3 continues, but that is tiered product delivery rather
than an L3 accuracy-preserving optimization.
