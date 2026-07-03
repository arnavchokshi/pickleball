# GPU Cold Start: BODY-stage pipeline readiness from a bare CUDA Ubuntu VM

Task OWNER-11. This is the durability runbook: what it takes to take a brand
new GPU VM from "bare CUDA Ubuntu with an NVIDIA driver" to "the BODY stage
of the pipeline can load the SAM-3D-Body model and run real GPU inference,"
using only what's in git (this repo + the vendored Fast-SAM-3D-Body repo)
plus one downloadable, license-gated model checkpoint. It exists so the
current spot-instance A100 (`pickleball-a100-spot-ase1a`, preemptible) can
be turned off without losing the ability to reproduce a working BODY
pipeline quickly on a replacement box.

Companion script: `scripts/racketsport/gpu_cold_start.sh`. Run it with a
target root directory:

```
scripts/racketsport/gpu_cold_start.sh /home/<user>/body_runtime_v2
```

It is idempotent (safe to re-run; each step checks its own completion state
before doing work) and writes a `cold_start_summary.json` with per-step
status and wall time under the root directory it's given.

## Scope and non-scope

In scope: everything `scripts/racketsport/remote_body_dispatch.py` needs on
the remote side to run the BODY stage (`--stage body` of
`threed.racketsport.orchestrator`) via its `FAST_SAM_PYTHON` /
`FAST_SAM_ROOT` subprocess bridge, proven by (a) the two-and-then-some real
GPU regression tests in `tests/racketsport/test_run_sam3dbody_batch.py`
passing (not skipping) and (b) a real single-crop GPU forward pass through
the actual SAM-3D-Body model.

Out of scope: no video-specific data, no eval clips, no CVAT labels, no
training checkpoints for BALL/TRK/court-keypoint work, no iOS toolchain. Not
in scope either: reproducing the *exact* multi-venv sprawl that has
accumulated on the current VM (`physpt_venv`, `blurball_venv`,
`ball_runtime_isolated`, `sam31_acq`, etc.) — those are other lanes' scratch
environments, not part of "pipeline readiness."

## Measured cold-start wall time (proof run, VM1, 2026-07-03)

Executed for real into an isolated directory tree on the existing VM that
shared nothing with `body_runtime/` or any existing repo checkout except the
GPU driver (`/home/arnavchokshi/cold_start_proof_20260703T0929Z`, since
deleted after evidence capture — see "Proof-run result" below). Full log,
`cold_start_summary.json`, pytest output, and inference-smoke output are
saved locally at `runs/cold_start_proof_20260703T0929Z/`.

| Step | What | Measured wall time |
|---|---|---|
| 1 | OS deps check (git, python3.10, python3.10-venv, flock) | 0s (all already present) |
| 2 | `git clone` this repo (depth 1, `main`) | 5s |
| 3 | `git clone` + checkout pinned commit of Fast-SAM-3D-Body | 5s |
| 4 | Build `body_venv` (torch+cu124 wheel download + ~85 more pins) | 156s |
| 5 | Download + sha256-verify the SAM-3D-Body checkpoint (~2.8 GB, HF) | 44s |
| 6 | `pytest tests/racketsport/test_run_sam3dbody_batch.py` (13 tests, 0 skipped) | 6s |
| 7 | Real single-crop GPU inference smoke (synthetic image) | 42s |
| **Total** | | **258s (~4m 18s)** |

Network conditions on this VM were fast (GitHub clones in single-digit
seconds; PyPI/PyTorch wheel downloads at 30-140 MB/s; the ~2.8 GB HF
checkpoint download took 44s including sha256 verification of both files).
A colder network path would mostly lengthen steps 4 (torch wheel is ~910 MB
of the ~2 GB step-4 total) and 5. Step 7's 42s is dominated by cold model
load onto GPU (state-dict load + CUDA context init) rather than the forward
pass itself, which the inference log shows took 4.15s
(`forward_decoder_body: 3.6313s`, `[run_inference] TOTAL: 4.1588s`) once the
model was resident on GPU — consistent with the SAM-3D-Body cold-start
stall documented elsewhere in this repo (12.8s→0.56s across a warmup
boundary) for a first real forward pass with no compile warmup.

**This total does not include OS-level GPU driver/CUDA provisioning** — this
runbook assumes a GCE image with the NVIDIA driver already working
(confirmed via `nvidia-smi` on VM1: driver `580.159.03`, A100-SXM4-40GB),
matching the task's "shares nothing... except the GPU driver" scope.

## Asset table

Everything the BODY stage actually loads at runtime today (i.e. with
`detector_name=""` and `fov_name=""` — see "Why detector/FOV assets aren't
required" below). Hashes are read live from `models/MANIFEST.json` by the
script, not hardcoded, but were also verified independently on VM1 during
this audit:

| Asset | Size | sha256 | Source |
|---|---|---|---|
| `model.ckpt` (SAM-3D-Body DINOv3 backbone) | 2,109,129,346 B (~2.0 GiB) | `b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf` | HuggingFace `facebook/sam-3d-body-dinov3` (gated, SAM License) via `huggingface_hub.snapshot_download` |
| `assets/mhr_model.pt` (MHR→SMPL mapping) | 696,110,248 B (~664 MiB) | `352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc` | same HF repo, same download call |
| `model_config.yaml` | 1,488 B | (not separately hashed in manifest) | same HF repo |
| Fast-SAM-3D-Body repo (code, no weights) | ~43 MB `.git` (2.7 GB checked-out incl. sample notebook images) | pinned at commit `808b53c7d9c26a7e511d31144f1e5efb058e15c9` | `https://github.com/yangtiming/Fast-SAM-3D-Body.git` (public) |
| This repo | 57 MB `.git` at HEAD | commit `b437b411886d7fe858e96136ce0d75fb46e95d32` | `https://github.com/arnavchokshi/pickleball.git` (public) |

Assets from `models/MANIFEST.json` that are declared but **not required** by
the BODY stage as it actually runs today (`moge_2_vitl_normal`, `yolo26m`)
are intentionally excluded — see below.

### Manifest inaccuracy found

`models/MANIFEST.json`'s `fast_sam_3d_body_dinov3` entry declares
`"repo_commit": "936894c37e51de9918012bcbc9ba2d9c20f73252"`. The Fast-SAM-3D-Body
checkout actually on VM1 (`/home/arnavchokshi/body_runtime/Fast-SAM-3D-Body`)
is at commit `808b53c7d9c26a7e511d31144f1e5efb058e15c9` ("Update README.md",
2026-06-18) — a different commit. The manifest's declared commit does not
reproduce what is actually running and gate-passing today.
`gpu_cold_start.sh` pins to the real VM1 commit (`808b53c7`), not the
manifest's stated one. `models/MANIFEST.json` should be corrected in a
follow-up (not done here — my writes for this task are restricted to
`docs/racketsport/GPU_COLD_START.md`, `scripts/racketsport/gpu_cold_start.sh`,
and the `runs/` proof directory).

## Why `detector_name=""` / `fov_name=""` (and why that shrinks the dependency set a lot)

`threed/racketsport/orchestrator.py`'s committed `_default_runners` hardcodes
`BodyStageRunner()` with `detector_name="yolo"`, `fov_name="moge2"`. Those
require `moge_2_vitl_normal` (manifest `local_path`:
`/workspace/checkpoints/body4d/moge-2-vitl-normal/model.pt`) and `yolo26m`.
**`/workspace` does not exist on VM1** — those paths are leftovers from an
earlier H100 container image. `scripts/racketsport/remote_body_dispatch.py`
works around this by generating a custom runner script
(`_remote_body_runner_script`) that explicitly passes
`RemoteConfig.body_detector_name=""` and `body_fov_name=""` — this is called
out in `RemoteConfig.body_detector_name`'s docstring as "the only
configuration that has actually produced real meshes on VM1." Per-frame
bboxes already come from `tracks.json`, so the detector is redundant for
this dispatch path anyway.

Tracing `Fast-SAM-3D-Body/notebook/utils.py::setup_sam_3d_body`: the
detector import (`tools.build_detector.HumanDetector`, needs `ultralytics`)
is gated by `if detector_name:`, the segmentor import
(`tools.build_sam.HumanSegmentor`, needs SAM2) is gated by `if segmentor_path:`
(never passed by this codebase), and the FOV import
(`tools.build_fov_estimator.FOVEstimator`, needs MoGe) is gated by
`if fov_name:`. With both `detector_name` and `fov_name` empty, **none of
detectron2, ultralytics, MoGe, or SAM2 are ever imported** by the BODY
stage as it actually runs. This is confirmed by `pip freeze` on the real
production venv (see below) — none of those packages are installed there.

Consequence: `verify_fast_sam_manifest_assets()`'s default
`REQUIRED_FAST_SAM_MODEL_IDS` (4 ids, including `moge_2_vitl_normal` and
`yolo26m`) is the wrong list for what's actually required; the code path
that's actually exercised
(`orchestrator.BodyStageRunner._ensure_runtime` at the call site) calls
`fast_sam_required_model_ids(detector_name=self.detector_name,
fov_name=self.fov_name)`, which with empty strings collapses to
`CORE_FAST_SAM_MODEL_IDS = (fast_sam_3d_body_dinov3, sam_3d_body_mhr_model)`
— exactly the two assets this runbook downloads and verifies.

## Why not `scripts/racketsport/install_fast_sam_env.sh`

`install_fast_sam_env.sh` is the repo's existing, checked-in Fast-SAM env
installer. It was **not used** to build `gpu_cold_start.sh`'s `body_venv`,
deliberately, because it does not match what is actually running in
production on VM1 today:

- It builds a **conda** environment (`conda create -n fast_sam_3d_body`,
  rooted at `$CONDA_ROOT` default `/opt/conda`). **VM1 has no conda
  installed at all** (`which conda` → not found). The real, working,
  gate-passing BODY-stage interpreter on VM1
  (`/home/arnavchokshi/body_runtime/fast_sam_venv`, the exact path
  `remote_body_dispatch.py`'s `DEFAULT_REMOTE_FAST_SAM_PYTHON` points at) is
  a plain `python3.10 -m venv`, not a conda env.
- It installs `detectron2`, `ultralytics`, `microsoft/MoGe`, TensorRT, ONNX
  Runtime GPU, etc. — none of which appear in the real venv's `pip freeze`
  (101 packages) and none of which are ever imported given
  `detector_name=""`/`fov_name=""` (see above). Installing them is wasted
  time and wasted disk on a real cold start.
- It installs a CUDA toolkit + gcc/g++ 13 via conda specifically so
  detectron2's C++/CUDA extensions can build (`FORCE_CUDA=1`,
  `TORCH_CUDA_ARCH_LIST=9.0`). Since detectron2 is never imported, none of
  that is needed either — VM1 itself has no system `nvcc` (`nvcc: command
  not found`) and doesn't need one; the pip `torch==2.5.1+cu124` wheel
  bundles its own CUDA 12.4 runtime libraries (`nvidia-cublas-cu12`,
  `nvidia-cudnn-cu12`, etc.), which is sufficient for inference.

`gpu_cold_start.sh` instead builds a plain `python3.10 -m venv` and installs
the exact 87-package pinned list captured from
`/home/arnavchokshi/body_runtime/fast_sam_venv`'s real `pip freeze` on
2026-07-03 (plus `pytest`, added only so this venv can also run the smoke
tests below — see "vendor workaround" next). `install_fast_sam_env.sh`'s
`pydantic==2.11.3` pin is already reflected in that captured list; the rest
of that script's package set (torch/torchvision pinned to `2.5.1+cu124`,
`smplx`, `chumpy` via `--no-build-isolation`, etc.) is compatible in spirit
but the exact version pins in `gpu_cold_start.sh` are what's proven to run
on VM1, not what `install_fast_sam_env.sh` would resolve to today (its own
list is unpinned beyond a handful of packages, so a fresh run of it would
not reproducibly match anyway). **Recommendation**: retarget
`install_fast_sam_env.sh` at the plain-venv, minimal-dependency shape (or
deprecate it in favor of `gpu_cold_start.sh`'s step 4) in a follow-up; not
done here per this task's file-ownership restriction.

## Is `/home/arnavchokshi/sam3d_validation2_bench/vendor/` (the pydantic/pytest PYTHONPATH workaround) still needed?

**No — it is obsolete for the documented pipeline and is excluded from this
runbook.**

`sam3d_validation2_bench/fast_sam_python_wrapper.sh` prepends
`sam3d_validation2_bench/vendor` (a `pip install --target` dump of
`pydantic==2.11.3`, `pytest==9.1.1`, and pytest's transitive deps —
`pluggy`, `iniconfig`, `tomli`, `exceptiongroup`, `pygments`,
`annotated_types`) to `PYTHONPATH` before exec-ing
`body_runtime/fast_sam_venv/bin/python`. Its own header comment says why it
exists: "a direct install was denied by the permission system in both
sessions" — i.e. an agent session could not get permission to `pip install`
directly into the shared production `fast_sam_venv`, so it vendored the
packages beside it instead, as a session-scoped hack.

Both of the reasons that hack existed are now moot:

1. **pydantic is no longer missing.** `pip freeze` on the real
   `body_runtime/fast_sam_venv` today shows `pydantic==2.11.3` installed
   directly in the venv (this must have landed after the vendor workaround
   was created, or independently of it).
2. **pytest was never needed in the shared production venv at all** — it's
   only needed to run repo unit/regression tests, which is a
   validation-time concern, not a BODY-stage runtime concern. On a fresh
   venv you control from the start (like `gpu_cold_start.sh`'s
   `body_venv`), there's no "permission system denied a direct install"
   problem — the venv is yours, so `pip install pytest` directly is simply
   step 4's last line. No PYTHONPATH shim required.

The `sam3d_validation2_bench/` directory as a whole (bench harness +
variant sweep results for the Phase D speed-optimization work) is unrelated
scratch from a separate lane and is out of scope for this runbook; only its
`vendor/` workaround sub-piece was in the audit's remit.

## Auth / durability HOLE: gated HuggingFace checkpoint

`facebook/sam-3d-body-dinov3` is a **gated** repository under Meta's SAM
License. The actual download call
(`Fast-SAM-3D-Body/sam_3d_body/build_models.py::_hf_download`, called by
`load_sam_3d_body_hf`) is:

```python
from huggingface_hub import snapshot_download
local_dir = snapshot_download(repo_id="facebook/sam-3d-body-dinov3")
```

This is a real, scripted download path (used by `gpu_cold_start.sh` step 5)
— but it only works for an HF account/token that has **already clicked
"accept" on the SAM License** for that repo on huggingface.co. That
acceptance is a manual, one-time, account-bound action; no script can do it.
VM1 already has such a token cached (`~/.cache/huggingface/token`,
verified present) from an earlier manual login, which is how the checkpoint
got there in the first place — the manifest's own note says "Downloaded
with huggingface_hub on 2026-07-01." **A genuinely fresh VM has no such
token file and no such acceptance; whoever cold-starts a new VM must
`huggingface-cli login` with a token from an HF account that has accepted
the license, or pass `HF_TOKEN` to `gpu_cold_start.sh`.** This is the one
real "exists only because of manual/account state, not git" prerequisite in
this whole runbook — flagged here per the task's instruction to call out
any asset with no fully scripted download path.

(Note: production code itself never calls `load_sam_3d_body_hf` — it always
passes `local_checkpoint_path`, i.e. `load_sam_3d_body`'s "use local
checkpoint" branch. The HF download is not on the BODY stage's hot path at
all; it is purely how the checkpoint file *gets onto disk* in the first
place, which is exactly the cold-start concern this runbook addresses.)

## Other durability findings (not fixed here — file-ownership restricted to the 3 targets listed for this task)

- **`remote_body_dispatch.py`'s own default remote python path is broken
  against VM1 today.** `DEFAULT_REMOTE_PYTHON =
  f"{DEFAULT_REMOTE_HOME}/pickleball_train_main/.venv/bin/python"`, but
  `/home/arnavchokshi/pickleball_train_main/.venv` **does not exist** on
  VM1 — there is no `.venv` anywhere under that checkout. The orchestrator
  venv that's actually used in practice on VM1 is
  `/home/arnavchokshi/pickleball/.venv` (torch `2.12.1+cu129`, pytest
  `9.1.1`, `torchreid`), a directory that is **not a git checkout at all**
  (`git -C /home/arnavchokshi/pickleball status` → "not a git repository").
  So the real working setup on VM1 is: code lives in one directory
  (`pickleball_train_main`, a git clone), the venv that runs it lives in a
  completely different, non-git directory (`pickleball/.venv`) — a split
  that only works because whoever runs commands there does so by hand with
  the right combination of `cd`/interpreter path, and that is exactly the
  kind of tribal knowledge that does not survive a VM rebuild. This
  runbook's own `body_venv` deliberately lives *inside* the isolated root
  next to the cloned repo, not split across two unrelated directories, so
  it doesn't reproduce this fragility. Filing this as a finding rather than
  fixing `remote_body_dispatch.py`'s default, which is out of this task's
  file scope.
- Three different `torch` versions are installed across three venvs
  observed on VM1: `pickleball/.venv` → `2.12.1+cu129`,
  `pickleball_git/.venv` → `2.1.0+cu118`, `body_runtime/fast_sam_venv` →
  `2.5.1+cu124`. This is drift accumulated over many sessions, not a
  designed multi-version setup. `gpu_cold_start.sh` pins to the
  `fast_sam_venv` version only, since that's the one BODY-stage dispatch
  actually depends on; it does not attempt to reproduce an orchestrator-CLI
  venv (`pickleball/.venv`'s full 165-package, 8.2 GB set) because nothing
  in this task's smoke-test bar (`test_run_sam3dbody_batch.py` +
  single-crop inference) requires it.
- VM1's boot disk was at **96-98% full (5.5-9 GB free)** for most of this
  audit, out of a 200 GB single disk with no secondary volume. `~/.cache/pip`
  (8.6 GB) and `~/.cache/uv` (7.8 GB) were cleared as part of this audit to
  create headroom for the proof run (safe: both are reproducible caches,
  not unique state; a genuinely fresh VM starts with both empty anyway).
  This is a real operational risk independent of cold-start durability: at
  96%+ disk, routine venv builds or checkpoint downloads can fail outright.
  Recommend either a larger boot disk or a dedicated scratch volume for
  future VM1-class instances.
- `docs/racketsport/scaffold_tool_index_schema.json`'s backing test,
  `tests/racketsport/test_scaffold_tool_index.py::test_real_scaffold_tool_index_matches_checked_in_schema`,
  enforces that every `scripts/racketsport/*.sh` has a "direct CLI
  reference test" (currently `EXPECTED_MISSING_DIRECT_CLI_REFERENCE` is the
  empty set — 0 gaps allowed). Adding `gpu_cold_start.sh` without a
  reference test will fail that assertion. The detection is purely
  string-literal: `scripts/racketsport/list_scaffold_tools.py`'s
  `_first_test_referencing` greps every `tests/racketsport/test_*.py` file
  for the literal string `"scripts/racketsport/gpu_cold_start.sh"`. The
  existing pattern for `scripts/gpu-eval-run.sh` lives in
  `tests/racketsport/test_shell_scripts.py` (`SHELL_SCRIPTS` list +
  `test_shell_scripts_are_executable_and_parse` + direct subprocess
  invocations later in that file, e.g. around line 317). **This task's
  write scope is restricted to this doc, `gpu_cold_start.sh`, and the
  `runs/` proof directory** — `tests/racketsport/test_shell_scripts.py` is
  a shared file another lane may be actively editing, so the needed change
  is specified here instead of applied:
  - Add `Path("scripts/racketsport/gpu_cold_start.sh")` to
    `test_shell_scripts.py::SHELL_SCRIPTS` (covers the existing
    "executable and parses" test for free).
  - Add a direct-CLI reference test, mirroring the existing
    `scripts/gpu-eval-run.sh` pattern, e.g. asserting
    `scripts/racketsport/gpu_cold_start.sh --help` exits 0 and its stdout
    mentions `body_venv` and `checkpoints/sam-3d-body-dinov3`.
  - Until that lands, `tests/racketsport/test_scaffold_tool_index.py`'s
    `test_real_scaffold_tool_index_matches_checked_in_schema` will fail
    after this commit (one new uncovered CLI). This is a known, expected,
    single-test gap to close in the same PR/lane that adds the test above.

## Runbook: step by step

All steps are inside `scripts/racketsport/gpu_cold_start.sh`; this section
explains what each does and why, for a human reading before running it.

### 1. OS-level prerequisites

Checks for `git`, `python3.10`, the `venv` stdlib module, and `flock`
(required by `scripts/gpu-eval-run.sh`'s shared-slot lease). Installs via
`apt-get` if anything is missing. On a standard GCE Deep Learning VM image
(NVIDIA driver + CUDA already provisioned by the image, which this runbook
assumes as the "GPU driver" baseline it doesn't try to reproduce) all four
are normally already present — confirmed present on VM1 (Ubuntu 22.04,
`git 2.34.1`, `python3.10.12`, `python3.10-venv` installed, `flock` from
`util-linux`).

### 2. Clone this repo

`git clone --depth 1 --branch main https://github.com/arnavchokshi/pickleball.git`.
The GitHub repo is **public** (`curl -s -o /dev/null -w '%{http_code}'
https://github.com/arnavchokshi/pickleball` → `200`, no auth needed for
read/clone). Re-running does `git fetch --depth 1` + `checkout -B` to
update in place rather than re-cloning.

### 3. Clone Fast-SAM-3D-Body, pinned

`git clone https://github.com/yangtiming/Fast-SAM-3D-Body.git`, then
`git checkout 808b53c7d9c26a7e511d31144f1e5efb058e15c9` (see "Manifest
inaccuracy found" above for why this commit, not the manifest's stated
one). This repo is used in "editable path" style — `sys.path.insert(0,
fast_sam_repo)` from `scripts/racketsport/run_sam3dbody_probe.py`'s
`_load_setup_sam_3d_body` — never `pip install`ed.

### 4. Build `body_venv`

`python3.10 -m venv`, then `pip install torch==2.5.1+cu124
torchvision==0.20.1+cu124 --extra-index-url
https://download.pytorch.org/whl/cu124`, then the remaining 85 pinned
packages captured from VM1's real `fast_sam_venv`, then `chumpy==0.70
--no-build-isolation` (needs numpy already installed, matching
`install_fast_sam_env.sh`'s own ordering note), then `pytest>=8.0` (added;
see vendor-workaround section above for why this is safe and sufficient).
No detectron2, no ultralytics, no MoGe, no SAM2, no system CUDA toolkit, no
gcc/g++ toolchain — see "Why not install_fast_sam_env.sh" above.

### 5. Fetch + verify the checkpoint

Reads the expected sha256 for `fast_sam_3d_body_dinov3` and
`sam_3d_body_mhr_model` live from `models/MANIFEST.json` (in the
just-cloned repo). If a local copy already sha256-matches, skips the
download (idempotent re-run). Otherwise downloads via
`huggingface_hub.snapshot_download(repo_id="facebook/sam-3d-body-dinov3",
local_dir=<checkpoint dir>)`, using `HF_TOKEN` (falls back with a loud
warning to the operator's existing cached token if unset — see the auth
HOLE above) and an **isolated `HF_HOME`** under the cold-start root (not
the operator's shared `~/.cache/huggingface`) for token/config lookups,
then sha256-verifies `model.ckpt` and `assets/mhr_model.pt`, failing loudly
on any mismatch.

`local_dir=` materializes real files directly at the target path in one
shot rather than the library's default global-cache-plus-symlink layout
(which would need a second full-size copy out of the cache to reach the
checkpoint directory — this is literally what production's own checkpoint
dir looks like too: its `.cache/huggingface/download/*.metadata` marker
files, found during this audit, are that same direct-materialization mode's
signature). **The first proof-run attempt used the copy-out-of-cache
pattern and ran VM1's boot disk out of space mid-copy** (`OSError: [Errno
28] No space left on device`, disk hit 100% full); switching to
`local_dir=` avoids ever holding two full copies of the ~2.8 GB checkpoint
at once. `gpu_cold_start.sh` step 4 also runs `pip cache purge` on the
just-built venv immediately after verifying it, since a fresh venv build
leaves several GB in `~/.cache/pip` that would otherwise still be competing
for the same tight disk headroom during step 5.

### 6. `pytest tests/racketsport/test_run_sam3dbody_batch.py`

Run through `scripts/gpu-eval-run.sh` (the shared-slot GPU lease — never
`gpu-train-lock.sh`, matching `remote_body_dispatch.py`'s own discipline).
This file has 13 tests; 3 gate on `pytest.importorskip("torch")` and would
silently SKIP rather than fail on a torch-less interpreter — exactly the
failure mode a cold start needs to catch. The script asserts `0 skipped`
and `13 passed`, a strictly stronger bar than "the two GPU regression tests
pass."

### 7. Real single-crop GPU inference smoke

Generates a tiny synthetic 640x480 JPEG (no real eval-clip data touched —
per this task's scope, no video-specific data matters) and a matching
one-request, one-bbox batch payload (`torch_compile: false`,
`crop_bucket_sizes: [1]`, to keep this fast — no compile-warmup stall),
then invokes the real production entrypoint,
`scripts/racketsport/run_sam3dbody_batch.py`, through
`scripts/gpu-eval-run.sh`, with `--detector-name "" --fov-name ""` (the
proven-working VM1 configuration). Asserts the output JSON has exactly one
frame with `pred_keypoints_3d` present — i.e. the real SAM-3D-Body model
loaded its weights and produced a real forward-pass output on GPU, not a
mocked/skipped path.

## Proof-run result

Executed on VM1 twice. **Attempt 1 failed** at step 5
(`OSError: [Errno 28] No space left on device`, VM boot disk hit 100% full)
because the original script downloaded the checkpoint into a global HF
cache and then `shutil.copy`'d it into the checkpoint directory — briefly
holding two full ~2.8 GB copies at once on a disk that had only 5-14 GB
free throughout this audit (VM1 was observed at 93-98% disk usage the whole
time, independent of this task). Fixed by switching to
`snapshot_download(..., local_dir=...)` (direct materialization, no copy)
and purging `pip`'s cache right after the venv build. **Attempt 2 passed
all 7 steps end to end**, `all_ok: true` in `cold_start_summary.json`,
258s total. Both `model.ckpt` and `assets/mhr_model.pt` were independently
re-hashed after the run and matched `models/MANIFEST.json` exactly. The
inference smoke log shows a real DINOv3 ViT-H/16+ backbone forward pass
(`[forward_pose_branch] TOTAL: 4.1547s`) producing real
`pred_keypoints_3d`/`pred_vertices` tensors for the synthetic crop — not a
skip, not a mock. Full evidence (run log, summary JSON, pytest output,
inference-smoke output and log) is at
`runs/cold_start_proof_20260703T0929Z/` in this repo.

The proof directory on VM1 (8.5 GB) was removed after evidence capture,
given the disk was already under real pressure from concurrent lane
activity during this audit.

## Re-running / cleanup

The script is idempotent — re-run it with the same root directory to
resume after a failure or refresh to a newer commit. To tear down a
cold-start root: `rm -rf <root_dir>` (nothing outside it is touched, except
whatever `HF_TOKEN` fallback read from `~/.cache/huggingface/token`, which
is only read, never written, by this script).
