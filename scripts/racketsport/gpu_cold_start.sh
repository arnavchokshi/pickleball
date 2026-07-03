#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/racketsport/gpu_cold_start.sh <root_dir> [options]

Brings a bare CUDA Ubuntu GPU VM to "BODY-stage pipeline ready" using only
git (this repo + the vendored Fast-SAM-3D-Body repo), a downloadable model
checkpoint (HuggingFace, gated -- see AUTH below), and pip. Every artifact
this script creates lives under <root_dir>; nothing outside <root_dir> is
read except the GPU driver/CUDA runtime already on the box and (for
HF_TOKEN auto-discovery) the operator's own ~/.cache/huggingface/token.

Idempotent: safe to re-run. Each step checks its own completion state
before doing work (git pull instead of re-clone, pip install is naturally
idempotent, checkpoint download is skipped once sha256 matches the
manifest, smoke tests always re-run because they are cheap and their
output is the evidence).

Layout created under <root_dir>:
  repo/                                   git clone of PICKLEBALL_REPO_URL
  body_runtime/body_venv/                 python3.10 venv (mirrors the
                                           real remote FAST_SAM_PYTHON venv
                                           that scripts/racketsport/
                                           remote_body_dispatch.py invokes
                                           on VM1 today -- NOT the conda env
                                           install_fast_sam_env.sh builds;
                                           see GPU_COLD_START.md "why not
                                           install_fast_sam_env.sh")
  body_runtime/Fast-SAM-3D-Body/          git clone of FAST_SAM_REPO_URL
                                           pinned at FAST_SAM_REPO_COMMIT
  body_runtime/checkpoints/sam-3d-body-dinov3/
                                           model.ckpt, model_config.yaml,
                                           assets/mhr_model.pt
  hf_home/                                isolated HF_HOME for token/config
                                           lookups only (the checkpoint
                                           itself downloads straight into
                                           body_runtime/checkpoints/ via
                                           local_dir=, not through here) so
                                           acquisition proves independent of
                                           any existing ~/.cache/huggingface
                                           on this box
  smoke/                                  synthetic smoke-test input/output
  cold_start_summary.json                 per-step status + wall time

Options:
  --skip-smoke        Skip both GPU smoke tests (steps 6-7). Useful for
                       CPU-only dry runs of steps 1-5.
  -h, --help           Show this help and exit.

Env vars:
  PICKLEBALL_REPO_URL      default: https://github.com/arnavchokshi/pickleball.git
  PICKLEBALL_REPO_REF      default: main
  FAST_SAM_REPO_URL        default: https://github.com/yangtiming/Fast-SAM-3D-Body.git
  FAST_SAM_REPO_COMMIT     default: 808b53c7d9c26a7e511d31144f1e5efb058e15c9
                           (the commit actually checked out in
                           /home/arnavchokshi/body_runtime/Fast-SAM-3D-Body
                           on VM1 as of 2026-07-03 -- NOT the repo_commit
                           models/MANIFEST.json's fast_sam_3d_body_dinov3
                           entry declares, 936894c37e51de9918012bcbc9ba2d9c20f73252,
                           which does not match; see GPU_COLD_START.md)
  HF_TOKEN                 HuggingFace token with read access to the
                           gated facebook/sam-3d-body-dinov3 repo (SAM
                           License must already be accepted by this
                           token's account on huggingface.co -- this is a
                           manual, account-bound prerequisite that no
                           script can automate). If unset, the script
                           falls back to copying the operator's existing
                           ~/.cache/huggingface/token, which only works on
                           a box where that login already happened -- a
                           genuinely fresh VM MUST supply HF_TOKEN.
  MANIFEST_PATH             default: <root_dir>/repo/models/MANIFEST.json
  GPU_LOCK_TIMEOUT_S        default: 120 (passed through to gpu-eval-run.sh)
EOF
}

ROOT_DIR=""
SKIP_SMOKE=0
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --skip-smoke)
      SKIP_SMOKE=1
      ;;
    -*)
      echo "gpu_cold_start: unknown option $arg" >&2
      usage >&2
      exit 64
      ;;
    *)
      if [ -n "$ROOT_DIR" ]; then
        echo "gpu_cold_start: unexpected extra argument $arg" >&2
        exit 64
      fi
      ROOT_DIR="$arg"
      ;;
  esac
done
if [ -z "$ROOT_DIR" ]; then
  usage >&2
  exit 64
fi

PICKLEBALL_REPO_URL="${PICKLEBALL_REPO_URL:-https://github.com/arnavchokshi/pickleball.git}"
PICKLEBALL_REPO_REF="${PICKLEBALL_REPO_REF:-main}"
FAST_SAM_REPO_URL="${FAST_SAM_REPO_URL:-https://github.com/yangtiming/Fast-SAM-3D-Body.git}"
FAST_SAM_REPO_COMMIT="${FAST_SAM_REPO_COMMIT:-808b53c7d9c26a7e511d31144f1e5efb058e15c9}"
GPU_LOCK_TIMEOUT_S="${GPU_LOCK_TIMEOUT_S:-120}"

ROOT_DIR="$(mkdir -p "$ROOT_DIR" && cd "$ROOT_DIR" && pwd)"
REPO_DIR="$ROOT_DIR/repo"
BODY_RUNTIME_DIR="$ROOT_DIR/body_runtime"
BODY_VENV_DIR="$BODY_RUNTIME_DIR/body_venv"
FAST_SAM_DIR="$BODY_RUNTIME_DIR/Fast-SAM-3D-Body"
CHECKPOINT_DIR="$BODY_RUNTIME_DIR/checkpoints/sam-3d-body-dinov3"
HF_HOME_DIR="$ROOT_DIR/hf_home"
SMOKE_DIR="$ROOT_DIR/smoke"
SUMMARY_FILE="$ROOT_DIR/cold_start_summary.json"
MANIFEST_PATH="${MANIFEST_PATH:-$REPO_DIR/models/MANIFEST.json}"

mkdir -p "$BODY_RUNTIME_DIR" "$HF_HOME_DIR" "$SMOKE_DIR"

STEP_NAMES=()
STEP_STATUSES=()
STEP_SECONDS=()
STEP_NOTES=()

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

record_step() {
  local name="$1" status="$2" seconds="$3" note="$4"
  STEP_NAMES+=("$name")
  STEP_STATUSES+=("$status")
  STEP_SECONDS+=("$seconds")
  STEP_NOTES+=("$note")
}

run_step() {
  local name="$1"
  shift
  log "=== step $name: start ==="
  local start_s end_s
  start_s=$(date +%s)
  if "$@"; then
    end_s=$(date +%s)
    record_step "$name" "ok" "$((end_s - start_s))" ""
    log "=== step $name: ok (${#STEP_SECONDS[@]}th step, $((end_s - start_s))s) ==="
  else
    end_s=$(date +%s)
    record_step "$name" "failed" "$((end_s - start_s))" "see log above"
    log "=== step $name: FAILED after $((end_s - start_s))s ==="
    write_summary
    exit 1
  fi
}

write_summary() {
  python3 - "$SUMMARY_FILE" "${#STEP_NAMES[@]}" "$ROOT_DIR" <<'PY' "${STEP_NAMES[@]}" -- "${STEP_STATUSES[@]}" -- "${STEP_SECONDS[@]}" -- "${STEP_NOTES[@]}"
import json
import sys

out_path = sys.argv[1]
n = int(sys.argv[2])
root_dir = sys.argv[3]
rest = sys.argv[4:]
sep_indices = [i for i, v in enumerate(rest) if v == "--"]
names = rest[: sep_indices[0]]
statuses = rest[sep_indices[0] + 1 : sep_indices[1]]
seconds = rest[sep_indices[1] + 1 : sep_indices[2]]
notes = rest[sep_indices[2] + 1 :]

steps = []
for i in range(n):
    steps.append(
        {
            "step": names[i],
            "status": statuses[i],
            "wall_seconds": int(seconds[i]),
            "note": notes[i],
        }
    )

payload = {
    "schema_version": 1,
    "artifact_type": "racketsport_gpu_cold_start_summary",
    "root_dir": root_dir,
    "steps": steps,
    "total_wall_seconds": sum(s["wall_seconds"] for s in steps),
    "all_ok": all(s["status"] == "ok" for s in steps),
}
with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
print(out_path)
PY
}

# --- step 1: OS-level prerequisites -----------------------------------------
step_os_deps() {
  local missing=()
  command -v git >/dev/null 2>&1 || missing+=(git)
  command -v python3.10 >/dev/null 2>&1 || missing+=(python3.10)
  python3.10 -c "import venv" >/dev/null 2>&1 || missing+=(python3.10-venv)
  command -v flock >/dev/null 2>&1 || missing+=(util-linux)
  if [ "${#missing[@]}" -eq 0 ]; then
    log "OS deps already present: git, python3.10, python3.10-venv, flock"
    return 0
  fi
  log "installing missing OS packages: ${missing[*]}"
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y git python3.10 python3.10-venv util-linux
  else
    apt-get update -y
    apt-get install -y git python3.10 python3.10-venv util-linux
  fi
}

# --- step 2: clone this repo -------------------------------------------------
step_clone_repo() {
  if [ -d "$REPO_DIR/.git" ]; then
    log "repo already present at $REPO_DIR, fetching latest $PICKLEBALL_REPO_REF"
    git -C "$REPO_DIR" fetch --depth 1 origin "$PICKLEBALL_REPO_REF"
    git -C "$REPO_DIR" checkout -B "$PICKLEBALL_REPO_REF" "origin/$PICKLEBALL_REPO_REF"
  else
    log "cloning $PICKLEBALL_REPO_URL@$PICKLEBALL_REPO_REF into $REPO_DIR"
    git clone --depth 1 --branch "$PICKLEBALL_REPO_REF" "$PICKLEBALL_REPO_URL" "$REPO_DIR"
  fi
  git -C "$REPO_DIR" rev-parse HEAD
}

# --- step 3: clone vendored Fast-SAM-3D-Body ---------------------------------
step_clone_fast_sam_repo() {
  if [ -d "$FAST_SAM_DIR/.git" ]; then
    log "Fast-SAM-3D-Body already present at $FAST_SAM_DIR, fetching pinned commit"
    git -C "$FAST_SAM_DIR" fetch --depth 1 origin "$FAST_SAM_REPO_COMMIT"
  else
    log "cloning $FAST_SAM_REPO_URL into $FAST_SAM_DIR"
    git clone "$FAST_SAM_REPO_URL" "$FAST_SAM_DIR"
    git -C "$FAST_SAM_DIR" fetch --depth 1 origin "$FAST_SAM_REPO_COMMIT" || true
  fi
  git -C "$FAST_SAM_DIR" checkout "$FAST_SAM_REPO_COMMIT"
}

# --- step 4: build body_venv (mirrors the real remote FAST_SAM_PYTHON venv) -
FAST_SAM_VENV_REQUIREMENTS=$(cat <<'REQS'
absl-py==2.4.0
aiohappyeyeballs==2.6.2
aiohttp==3.14.1
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
antlr4-python3-runtime==4.9.3
anyio==4.14.1
appdirs==1.4.4
async-timeout==5.0.1
attrs==26.1.0
braceexpand==0.1.7
certifi==2026.6.17
click==8.4.2
cloudpickle==3.1.2
colorlog==6.10.1
contourpy==1.3.2
cycler==0.12.1
dill==0.4.1
einops==0.8.2
exceptiongroup==1.3.1
filelock==3.29.4
fonttools==4.63.0
frozenlist==1.8.0
fsspec==2026.6.0
grpcio==1.81.1
h11==0.16.0
hf-xet==1.5.1
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.21.0
hydra-colorlog==1.2.0
hydra-core==1.3.3
hydra-submitit-launcher==1.2.0
idna==3.18
Jinja2==3.1.6
jsonlines==4.0.0
kiwisolver==1.5.0
lightning-utilities==0.15.3
Markdown==3.10.2
markdown-it-py==4.2.0
MarkupSafe==3.0.3
matplotlib==3.10.9
mdurl==0.1.2
mpmath==1.3.0
multidict==6.7.1
networkx==3.4.2
numpy==2.2.6
omegaconf==2.3.1
opencv-python-headless==4.13.0.92
packaging==26.2
pillow==12.3.0
propcache==0.5.2
protobuf==7.35.1
pydantic==2.11.3
pydantic_core==2.33.1
Pygments==2.20.0
pyparsing==3.3.2
pyrootutils==1.0.4
python-dateutil==2.9.0.post0
python-dotenv==1.2.2
pytorch-lightning==2.6.5
PyYAML==6.0.3
rich==15.0.0
roma==1.5.6
safetensors==0.8.0
scipy==1.15.3
shellingham==1.5.4
six==1.17.0
smplx==0.1.28
submitit==1.5.4
sympy==1.13.1
tensorboard==2.21.0
tensorboard-data-server==0.7.2
termcolor==3.3.0
timm==1.0.27
torchmetrics==1.9.0
tqdm==4.68.3
typer==0.25.1
typing-inspection==0.4.2
typing_extensions==4.15.0
webdataset==1.0.2
Werkzeug==3.1.8
yacs==0.1.8
yarl==1.24.2
REQS
)

step_build_body_venv() {
  if [ ! -x "$BODY_VENV_DIR/bin/python" ]; then
    log "creating venv at $BODY_VENV_DIR"
    python3.10 -m venv "$BODY_VENV_DIR"
  else
    log "venv already exists at $BODY_VENV_DIR, reusing (pip install below is idempotent)"
  fi
  "$BODY_VENV_DIR/bin/python" -m pip install --upgrade pip
  # torch/torchvision pinned exactly to what VM1's real remote FAST_SAM_PYTHON
  # venv (/home/arnavchokshi/body_runtime/fast_sam_venv) has today, verified
  # via `pip freeze` on 2026-07-03. detector_name=""/fov_name="" (the only
  # config that has ever produced real meshes on VM1 -- see
  # scripts/racketsport/remote_body_dispatch.py's RemoteConfig docstring)
  # means detectron2/ultralytics/MoGe/SAM2 are never imported at runtime, so
  # they are deliberately NOT installed here (install_fast_sam_env.sh installs
  # them; that script targets a different, unused conda layout -- see
  # GPU_COLD_START.md).
  "$BODY_VENV_DIR/bin/python" -m pip install \
    torch==2.5.1+cu124 torchvision==0.20.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124
  printf '%s\n' "$FAST_SAM_VENV_REQUIREMENTS" | "$BODY_VENV_DIR/bin/python" -m pip install -r /dev/stdin
  # chumpy is a legacy setup.py package; --no-build-isolation (matching
  # install_fast_sam_env.sh's own approach, needed because chumpy's setup.py
  # unconditionally imports numpy at build time) requires setuptools+wheel to
  # already be present in the venv, which a plain `python3.10 -m venv` does
  # not guarantee on this pip/Python combination (pip 22.0.2 as shipped by
  # Ubuntu 22.04's python3.10-venv does not vendor `wheel`) -- without this,
  # the build fails with "invalid command 'bdist_wheel'".
  "$BODY_VENV_DIR/bin/python" -m pip install wheel setuptools
  "$BODY_VENV_DIR/bin/python" -m pip install chumpy==0.70 --no-build-isolation
  # pytest is NOT part of the real production fast_sam_venv (it is added here
  # only so this same venv can run the smoke tests below). On the real VM1
  # venv, someone previously worked around a missing pytest by PYTHONPATH-
  # vendoring it into a side directory
  # (/home/arnavchokshi/sam3d_validation2_bench/vendor) rather than installing
  # it directly -- see GPU_COLD_START.md "vendor workaround" section for why
  # that workaround is unnecessary on a fresh venv like this one.
  "$BODY_VENV_DIR/bin/python" -m pip install 'pytest>=8.0'
  "$BODY_VENV_DIR/bin/python" - <<'PY'
import importlib.util
for mod in ("torch", "torchvision", "cv2", "pydantic", "pytest", "huggingface_hub", "smplx"):
    if importlib.util.find_spec(mod) is None:
        raise SystemExit(f"missing {mod}")
print("body_venv ready")
PY
  # A fresh venv build downloads several GB into pip's HTTP cache
  # (~/.cache/pip) on top of the venv's own installed-package bytes. On a
  # tight boot disk (VM1 was observed at 93-98% full / single-digit GB free
  # throughout this audit) that transient cache is exactly what pushes the
  # checkpoint download in the next step over the edge -- purge it now that
  # the venv itself is built and verified importable.
  "$BODY_VENV_DIR/bin/python" -m pip cache purge || true
}

# --- step 5: fetch + verify the SAM-3D-Body checkpoint -----------------------
step_fetch_checkpoint() {
  mkdir -p "$CHECKPOINT_DIR/assets"
  local expected_ckpt_sha expected_mhr_sha
  expected_ckpt_sha=$(python3 -c "
import json
manifest = json.load(open('$MANIFEST_PATH'))
for m in manifest['models']:
    if m['id'] == 'fast_sam_3d_body_dinov3':
        print(m['sha256'])
        break
")
  expected_mhr_sha=$(python3 -c "
import json
manifest = json.load(open('$MANIFEST_PATH'))
for m in manifest['models']:
    if m['id'] == 'sam_3d_body_mhr_model':
        print(m['sha256'])
        break
")
  if [ -z "$expected_ckpt_sha" ] || [ -z "$expected_mhr_sha" ]; then
    log "FATAL: models/MANIFEST.json is missing fast_sam_3d_body_dinov3 or sam_3d_body_mhr_model sha256"
    return 1
  fi

  if [ -f "$CHECKPOINT_DIR/model.ckpt" ] && [ -f "$CHECKPOINT_DIR/assets/mhr_model.pt" ] \
     && [ "$(sha256sum "$CHECKPOINT_DIR/model.ckpt" | cut -d' ' -f1)" = "$expected_ckpt_sha" ] \
     && [ "$(sha256sum "$CHECKPOINT_DIR/assets/mhr_model.pt" | cut -d' ' -f1)" = "$expected_mhr_sha" ]; then
    log "checkpoint + mhr asset already present and sha256-verified, skipping download"
    return 0
  fi

  # AUTH HOLE (see GPU_COLD_START.md): facebook/sam-3d-body-dinov3 is a
  # gated HF repo under the SAM License. HF_TOKEN's account must have
  # already accepted that license on huggingface.co -- this cannot be
  # scripted, it is a one-time manual step per HF account.
  local hf_token="${HF_TOKEN:-}"
  if [ -z "$hf_token" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    hf_token="$(cat "$HOME/.cache/huggingface/token")"
    log "WARNING: HF_TOKEN not set, falling back to this operator's existing" \
        "~/.cache/huggingface/token. A genuinely fresh VM has no such file and" \
        "MUST be given HF_TOKEN explicitly (see GPU_COLD_START.md AUTH HOLE)."
  fi
  if [ -z "$hf_token" ]; then
    log "FATAL: no HF_TOKEN available and no cached token to fall back to." \
        "facebook/sam-3d-body-dinov3 is gated; export HF_TOKEN before re-running."
    return 1
  fi

  # local_dir=<checkpoint dir> makes huggingface_hub materialize real files
  # directly at that path (this is what production's own checkpoint dir
  # layout looks like too -- its .cache/huggingface/download/*.metadata
  # marker files, found during this audit, are this exact mode's signature)
  # instead of the default global-cache-plus-symlink layout, which would
  # need a second full-size copy out of HF_HOME to reach CHECKPOINT_DIR and
  # doubles peak disk usage for no benefit on a one-shot cold start (this is
  # exactly what caused the first proof attempt to run the VM's boot disk
  # out of space). HF_HOME is still set to the isolated dir so token/config
  # lookups never touch the operator's shared ~/.cache/huggingface.
  log "downloading facebook/sam-3d-body-dinov3 via huggingface_hub directly into $CHECKPOINT_DIR"
  HF_HOME="$HF_HOME_DIR" HF_TOKEN="$hf_token" "$BODY_VENV_DIR/bin/python" - <<PY
import os
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="facebook/sam-3d-body-dinov3",
    token=os.environ["HF_TOKEN"],
    local_dir="$CHECKPOINT_DIR",
)
print("downloaded directly into", local_dir)
PY

  local actual_ckpt_sha actual_mhr_sha
  actual_ckpt_sha=$(sha256sum "$CHECKPOINT_DIR/model.ckpt" | cut -d' ' -f1)
  actual_mhr_sha=$(sha256sum "$CHECKPOINT_DIR/assets/mhr_model.pt" | cut -d' ' -f1)
  if [ "$actual_ckpt_sha" != "$expected_ckpt_sha" ]; then
    log "FATAL: model.ckpt sha256 mismatch: expected $expected_ckpt_sha got $actual_ckpt_sha"
    return 1
  fi
  if [ "$actual_mhr_sha" != "$expected_mhr_sha" ]; then
    log "FATAL: mhr_model.pt sha256 mismatch: expected $expected_mhr_sha got $actual_mhr_sha"
    return 1
  fi
  log "checkpoint + mhr asset downloaded and sha256-verified against models/MANIFEST.json"
}

# --- step 6: pytest GPU regression smoke -------------------------------------
# test_run_sam3dbody_batch.py has 13 tests; 3 of them
# (test_direct_bucket_model_calls_and_numpy_conversion_run_under_inference_mode,
# test_warmup_and_real_synthetic_batches_have_matching_guard_signatures,
# test_static_clip_intrinsics_warmup_runs_each_bucket_shape_configured_passes)
# gate on `pytest.importorskip("torch")` and SKIP silently on a torch-less
# interpreter instead of failing loudly -- exactly the failure mode a cold
# start must catch. The bar here is strictly stronger than "2 GPU tests
# pass": 0 skipped, all 13 passed, on this venv's real CUDA torch build.
step_pytest_smoke() {
  ( cd "$REPO_DIR" && \
    GPU_LOCK_TIMEOUT_S="$GPU_LOCK_TIMEOUT_S" "$REPO_DIR/scripts/gpu-eval-run.sh" \
    "$BODY_VENV_DIR/bin/python" -m pytest tests/racketsport/test_run_sam3dbody_batch.py -v \
    2>&1 | tee "$SMOKE_DIR/pytest_full_file.log" )
  if grep -qE "[1-9][0-9]* skipped" "$SMOKE_DIR/pytest_full_file.log"; then
    log "FATAL: one or more tests in test_run_sam3dbody_batch.py SKIPPED (torch not importable in body_venv?)"
    return 1
  fi
  grep -q "13 passed" "$SMOKE_DIR/pytest_full_file.log" || {
    log "FATAL: expected all 13 tests in test_run_sam3dbody_batch.py to pass"
    return 1
  }
}

# --- step 7: minimal single-bucket GPU inference smoke ------------------------
step_inference_smoke() {
  "$BODY_VENV_DIR/bin/python" - <<PY
import numpy as np
import cv2

img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (160, 72), (480, 456), (90, 140, 200), thickness=-1)
cv2.imwrite("$SMOKE_DIR/synthetic_frame.jpg", img)
print("wrote synthetic frame")
PY

  cat > "$SMOKE_DIR/requests.json" <<JSON
{
  "schema_version": 1,
  "clip_intrinsics": {
    "fx": 1000.0,
    "fy": 1000.0,
    "cx": 320.0,
    "cy": 240.0,
    "dist": [],
    "source": "gpu_cold_start_synthetic",
    "static_per_clip": true
  },
  "optimization": {
    "sam3d_body_input_size_px": 384,
    "crop_bucket_sizes": [1],
    "torch_compile": false,
    "compile_warmup_buckets": [],
    "steady_state_empty_cache": true,
    "inner_bucket_sync": true,
    "upstream_env": {},
    "tier2_output_lite": false
  },
  "requests": [
    {
      "request_id": "cold-start-smoke-1",
      "image": "$SMOKE_DIR/synthetic_frame.jpg",
      "bboxes": [[160.0, 72.0, 480.0, 456.0]]
    }
  ]
}
JSON

  ( cd "$REPO_DIR" && \
    GPU_LOCK_TIMEOUT_S="$GPU_LOCK_TIMEOUT_S" "$REPO_DIR/scripts/gpu-eval-run.sh" \
    "$BODY_VENV_DIR/bin/python" "$REPO_DIR/scripts/racketsport/run_sam3dbody_batch.py" \
    --requests "$SMOKE_DIR/requests.json" \
    --out "$SMOKE_DIR/out.json" \
    --fast-sam-repo "$FAST_SAM_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --detector-name "" \
    --fov-name "" \
    2>&1 | tee "$SMOKE_DIR/inference_smoke.log" )

  python3 -c "
import json
out = json.load(open('$SMOKE_DIR/out.json'))
assert out['request_count'] == 1, out
frame = out['frames'][0]
assert frame['request_id'] == 'cold-start-smoke-1', frame
records = frame['records']
assert len(records) == 1 and 'pred_keypoints_3d' in records[0], records
print('inference smoke OK: real GPU forward pass produced pred_keypoints_3d')
"
}

main() {
  run_step "01_os_deps" step_os_deps
  run_step "02_clone_pickleball_repo" step_clone_repo
  run_step "03_clone_fast_sam_3d_body_repo" step_clone_fast_sam_repo
  run_step "04_build_body_venv" step_build_body_venv
  run_step "05_fetch_and_verify_checkpoint" step_fetch_checkpoint
  if [ "$SKIP_SMOKE" -eq 0 ]; then
    run_step "06_pytest_gpu_regression_smoke" step_pytest_smoke
    run_step "07_gpu_inference_smoke" step_inference_smoke
  else
    log "--skip-smoke set: skipping steps 06-07"
  fi
  write_summary
  log "cold start complete: $SUMMARY_FILE"
}

main
