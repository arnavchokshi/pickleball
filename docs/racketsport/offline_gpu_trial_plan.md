# Offline GPU Trial Plan

Last updated: 2026-07-01

This note records the planned GPU trial set for the server-side offline/deep
pipeline. It is a planning document, not a promotion claim. Current production
truth still lives in `CAPABILITIES.md`, `BUILD_CHECKLIST.md`, and the actual
run artifacts.

The target path is the offline job shape where phone and CPU work narrow the
work before the paid GPU starts:

- phone/capture priors and CPU materialization happen first;
- GPU work is billed only while model setup and inference are running;
- BODY uses Fast SAM-3D-Body in tracked-bbox mode;
- Fast SAM is invoked once per scheduled tracked-player crop/person-frame;
- full-frame BALL/TrackNet is treated separately because it scales by video
  frame and can dominate runtime if run naively.

## Current Cost Model

Use `runs/gpu_job_cost_model_20260701/` as the current model snapshot.

Key model assumptions:

- 10-minute planning job.
- 24 scheduled BODY person-frames per video second.
- 14,400 BODY person-frames per 10-minute job.
- A100 measured-family anchor: about 310 ms per BODY person-frame after warmup.
- Model setup: about 16.3 seconds per warm job.
- Runtime formula:

```text
gpu_seconds = setup_seconds + body_person_frames * ms_per_person_frame * 1.10 + packaging_seconds
gpu_cost = gpu_seconds / 3600 * hourly_gpu_price
```

Cold pod capacity wait is excluded because the intended deployment stores
models on a pod/volume and starts the expensive GPU only for execution.

## GPUs To Try

| GPU | Role | Planning runtime/job | Planning cost/job | Current confidence | Decision rule |
|---|---|---:|---:|---|---|
| H100 | Fast deadline runner and acceptance reference | 55-58 min | $2.91-$3.17 | repo-target estimate plus prior H100 smoke evidence | Use when turnaround or H100-named gates matter. |
| A100 | Safe measured-family baseline | 82-84 min | $1.94-$2.04 | measured-family A100 BODY logs | Use as baseline for comparing every other GPU. |
| RTX 5090 | Consumer Blackwell speed/cost candidate | 65 min | $1.07 | estimate only; needs smoke | Try after baseline. Keep only if no OOM/retry and outputs match A100/H100 structure. |
| RTX 4090 | Likely best routine cost/job if runtime fits | 88 min at 330 ms/person-frame | $1.01 at $0.69/hr | estimate only; needs smoke | Use for routine BODY jobs only if actual Fast SAM BODY is <=350 ms/person-frame and stable. |

RTX 4090 sensitivity at $0.69/hr:

| Actual BODY speed | Runtime/job | Cost/job |
|---:|---:|---:|
| 260 ms/person-frame | 69 min | $0.79 |
| 310 ms/person-frame | 82 min | $0.95 |
| 330 ms/person-frame | 88 min | $1.01 |
| 420 ms/person-frame | 111 min | $1.28 |
| 480 ms/person-frame | 127 min | $1.46 |
| 600 ms/person-frame | 159 min | $1.83 |

Interpretation: RTX 4090 is worth trying because the repo's H100 single-image
Fast SAM smoke recorded about 4.7 GB peak allocation, so 24 GB VRAM should be
enough for sequential BODY crops. It is not yet trusted for batching, full
offline serving, or promotion evidence because it has less VRAM headroom, no
datacenter ECC profile, and no local BODY smoke result yet.

## Trial Order

1. **A100 baseline rerun**
   - Re-run the current Fast SAM BODY smoke on the same clip/window shape used
     for the cost model.
   - Capture setup time, warmup image time, steady-state ms/person-frame,
     p50/p95, peak allocated/reserved VRAM, output schema, and full-clip BODY
     structural gate status.

2. **H100 rerun**
   - Re-run the same BODY smoke on H100.
   - Treat this as the deadline and H100-gate reference path.
   - Record the exact H100 SKU because H100 PCIe, SXM, and NVL have different
     price/runtime tradeoffs.

3. **RTX 4090 smoke**
   - Run the exact same BODY smoke sequentially, not batched.
   - Pass condition: no OOM/retry, output artifacts structurally match A100,
     steady BODY speed is <=350 ms/person-frame, and p95 is not wildly above
     the mean.
   - If actual speed is 350-450 ms/person-frame, keep it as a cheap dev/backlog
     option only.
   - If actual speed is >450 ms/person-frame or unstable, do not use it for the
     routine offline worker.

4. **RTX 5090 smoke**
   - Run after RTX 4090 because it should answer whether the extra 8 GB VRAM
     and Blackwell speed make the higher hourly price worthwhile.
   - Pass condition: no OOM/retry, output artifacts structurally match A100,
     and cost/job stays near or below RTX 4090 while improving turnaround.

## Metrics To Record For Every GPU

Each GPU trial should write a compact JSON summary next to the run artifacts:

- provider, GPU name, SKU, VRAM, hourly price, availability tier;
- git commit, container image, CUDA, driver, PyTorch, model manifest sha256s;
- model setup seconds;
- first warmup person-frame seconds;
- steady-state per-person-frame mean, median, p95, min, max;
- peak allocated and reserved VRAM;
- scheduled frames and scheduled person-frames;
- total billed GPU seconds for the run;
- cost per run and normalized cost per 10,000 BODY person-frames;
- BODY output schema status and structural gate status;
- failures, retries, OOMs, or runtime warnings.

## What Not To Conclude Yet

- Do not promote BODY from RTX 4090 or RTX 5090 evidence until the same
  structural and accuracy gates used for A100/H100 pass.
- Do not compare GPUs using naive full-frame TrackNet unless the BALL path is
  explicitly part of that benchmark. TrackNet currently scales by video frame,
  while BODY scales by scheduled person-frame.
- Do not use hourly price alone. The decision metric is cost per completed job
  at the required turnaround time.
- Do not assume all A100 or H100 listings are equivalent. Record PCIe/SXM/NVL
  and VRAM size for each run.

## Current Decision

The likely future worker split is:

- **RTX 4090:** first candidate for cheap routine BODY-only offline jobs if the
  smoke lands near A100 speed.
- **RTX 5090:** candidate when sub-75-minute turnaround matters and the smoke
  proves stable.
- **A100:** measured-family baseline and safe fallback when consumer cards are
  unstable or too tight on VRAM.
- **H100:** deadline/acceptance runner when H100-class evidence is needed or
  queue latency matters more than cost/job.

Related artifacts:

- `runs/gpu_job_cost_model_20260701/gpu_cost_per_job_report_v2.md`
- `runs/gpu_job_cost_model_20260701/job_cost_model_v2.csv`
- `runs/gpu_job_cost_model_20260701/job_cost_shortlist.csv`
- `runs/gpu_job_cost_model_20260701/03_rtx4090_sensitivity.png`
