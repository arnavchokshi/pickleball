#!/usr/bin/env bash
# Fleet VM startup (FABLE_OPERATING_MANUAL §12) — runs on every boot of a fable-lane VM.
# STATUS: SCAFFOLD — flesh out in the P0-1 GPU cold-start lane; keep steps idempotent.
# CUDA COMPUTE MODE POLICY:
#   - Pipeline VMs default to DEFAULT. process_video.py and its BODY-local child
#     process need separate CUDA contexts on the same single-GPU VM.
#   - EXCLUSIVE_PROCESS is opt-in only for an explicitly declared training lane
#     guaranteed to use one CUDA context. Set FABLE_ROLE=training (or GCE
#     fable-role=training) together with FABLE_CUDA_COMPUTE_MODE=EXCLUSIVE_PROCESS
#     (or GCE fable-cuda-compute-mode=EXCLUSIVE_PROCESS).
#   - Do not opt a pipeline/BODY-local VM into EXCLUSIVE_PROCESS.
set -euo pipefail

# 1. Arm the preemption watcher before any bounded metadata lookup.
( while sleep 5; do
    if curl -s -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/instance/preempted | grep -q TRUE; then
      touch /tmp/PREEMPTED; break
    fi
  done ) & disown

metadata_attribute() {
  curl -s -f --connect-timeout 1 --max-time 2 -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null
}

# 2. Select CUDA context policy for this lane. Environment wins over metadata;
#    absent configuration means the pipeline-safe DEFAULT mode.
CUDA_COMPUTE_MODE_METADATA=""
if [ -z "${FABLE_CUDA_COMPUTE_MODE:-}" ]; then
  CUDA_COMPUTE_MODE_METADATA="$(metadata_attribute fable-cuda-compute-mode || true)"
fi
CUDA_COMPUTE_MODE="${FABLE_CUDA_COMPUTE_MODE:-${CUDA_COMPUTE_MODE_METADATA:-DEFAULT}}"
FABLE_ROLE_METADATA=""
if [ -z "${FABLE_ROLE:-}" ]; then
  FABLE_ROLE_METADATA="$(metadata_attribute fable-role || true)"
fi
LANE_FABLE_ROLE="${FABLE_ROLE:-${FABLE_ROLE_METADATA:-}}"
case "$CUDA_COMPUTE_MODE" in
  DEFAULT|EXCLUSIVE_PROCESS)
    ;;
  *)
    echo "lane_vm_startup: unsupported CUDA compute mode: $CUDA_COMPUTE_MODE" >&2
    exit 64
    ;;
esac
if [ "$CUDA_COMPUTE_MODE" = "EXCLUSIVE_PROCESS" ] && [ "$LANE_FABLE_ROLE" != "training" ]; then
  echo "lane_vm_startup: EXCLUSIVE_PROCESS requires explicit fable-role=training" >&2
  exit 64
fi
echo "lane_vm_startup: CUDA compute mode $CUDA_COMPUTE_MODE"
if ! nvidia-smi -c "$CUDA_COMPUTE_MODE"; then
  echo "lane_vm_startup: failed to set CUDA compute mode $CUDA_COMPUTE_MODE" >&2
  exit 1
fi

# 3. Training-data integrity gate. A training VM is not startup-complete until
#    the exact checked-out verifier writes a passing proof. The intended-input
#    manifest must enumerate every trainer-visible data path and ledger asset.
#    When any path is under /cache, FABLE_CACHE_MANIFEST must point at the
#    mounted CACHE_MANIFEST.json. Trainers receive the resulting path through
#    --gate-proof and must validate it again immediately before input reads.
if [ "$LANE_FABLE_ROLE" = "training" ]; then
  if [ -z "${FABLE_TRAINING_INPUT_MANIFEST:-}" ]; then
    echo "lane_vm_startup: TRAINING_INPUT_MANIFEST_REQUIRED: set FABLE_TRAINING_INPUT_MANIFEST" >&2
    exit 65
  fi
  if [ -z "${FABLE_REPO_DIR:-}" ]; then
    echo "lane_vm_startup: TRAINING_REPO_REQUIRED: set FABLE_REPO_DIR to the exact checked-out revision" >&2
    exit 65
  fi
  if [ -z "${FABLE_GATE_PROOF:-}" ]; then
    echo "lane_vm_startup: GATE_PROOF_PATH_REQUIRED: set FABLE_GATE_PROOF" >&2
    exit 65
  fi

  TRAINING_DATA_LEDGER="${FABLE_DATA_LEDGER:-$FABLE_REPO_DIR/runs/manager/data_ledger.json}"
  TRAINING_PYTHON="${FABLE_TRAINING_PYTHON:-$FABLE_REPO_DIR/.venv/bin/python}"
  TRAINING_VERIFIER="$FABLE_REPO_DIR/scripts/racketsport/verify_training_inputs.py"
  if [ ! -x "$TRAINING_PYTHON" ]; then
    echo "lane_vm_startup: TRAINING_PYTHON_UNAVAILABLE: $TRAINING_PYTHON" >&2
    exit 65
  fi
  if [ ! -f "$TRAINING_VERIFIER" ]; then
    echo "lane_vm_startup: TRAINING_VERIFIER_UNAVAILABLE: $TRAINING_VERIFIER" >&2
    exit 65
  fi

  TRAINING_GATE_ARGS=(
    --inputs "$FABLE_TRAINING_INPUT_MANIFEST"
    --ledger "$TRAINING_DATA_LEDGER"
    --repo-root "$FABLE_REPO_DIR"
    --gate-proof "$FABLE_GATE_PROOF"
  )
  if [ -n "${FABLE_CACHE_MANIFEST:-}" ]; then
    TRAINING_GATE_ARGS+=(--cache-manifest "$FABLE_CACHE_MANIFEST")
  fi
  if ! "$TRAINING_PYTHON" "$TRAINING_VERIFIER" "${TRAINING_GATE_ARGS[@]}"; then
    echo "lane_vm_startup: TRAINING_INPUT_GATE_FAILED: refusing training VM startup" >&2
    exit 65
  fi
  if [ ! -s "$FABLE_GATE_PROOF" ]; then
    echo "lane_vm_startup: GATE_PROOF_MISSING: verifier returned without a proof artifact" >&2
    exit 65
  fi
  echo "lane_vm_startup: training input gate PASS: $FABLE_GATE_PROOF"
fi

# 4. Code + weights: git clone/pull the repo (it is pushed) + restore vendor pins per
#    third_party/VENDOR_PINS.md + pull weights per models/MANIFEST.json (see RESET_HANDOFF §7 /
#    scripts/racketsport/gpu_cold_start.sh — proven 258s).
echo "lane_vm_startup: scaffold complete (extend in P0-1 lane)"

# ---------------------------------------------------------------------------
# 5. INFRA-2 pull-worker install hook (product-infra plan §INFRA-2 / §3).
#    INERT on every VM except one booted with GCE metadata
#    `fable-role=pickleball-worker` — training/eval fleet VMs are untouched
#    by construction. The bounded role lookup above leaves a missing role empty,
#    so a normal lane VM never trips `set -e` here.
#    Idempotent: safe to re-run on every boot.
# ---------------------------------------------------------------------------
WORKER_FABLE_ROLE="$LANE_FABLE_ROLE"
if [ "$WORKER_FABLE_ROLE" = "pickleball-worker" ]; then
  echo "lane_vm_startup: fable-role=pickleball-worker — installing pull-worker"

  WORKER_ROOT=/opt/pickleball-worker
  WORKER_ENV_DIR=/etc/pickleball-worker
  PIPELINE_ROOT="$WORKER_ROOT/pipeline"
  WORKER_VENV="$WORKER_ROOT/worker_venv"
  REPO_DIR="$PIPELINE_ROOT/repo"

  mkdir -p "$WORKER_ROOT" "$WORKER_ENV_DIR"

  # 4a. Repo + pipeline runtime (checkpoints, model manifest): REUSE the
  #     proven idempotent cold-start (scripts/racketsport/gpu_cold_start.sh)
  #     rather than re-implementing clone/venv/checkpoint logic here.
  #     Bootstrapped via raw-fetch since the repo doesn't exist yet on a
  #     bare VM (gpu_cold_start.sh itself performs the clone as its step 1
  #     once invoked).
  curl -fsSL \
    "https://raw.githubusercontent.com/arnavchokshi/pickleball/main/scripts/racketsport/gpu_cold_start.sh" \
    -o /tmp/gpu_cold_start.sh
  bash /tmp/gpu_cold_start.sh "$PIPELINE_ROOT" --skip-smoke

  # KNOWN GAP (tracked outside this lane, confirm before flipping
  # PICKLEBALL_QUEUE_ENABLED=1 against a genuinely fresh VM):
  # gpu_cold_start.sh provisions body_runtime/body_venv (the narrower
  # Fast-SAM-3D-Body venv) and the repo clone, but process_video.py itself
  # runs under the FULL pipeline `.venv` (torch/cv2/mmdet/etc.) that today
  # only exists pre-built on VM1 (/home/arnavchokshi/pickleball_git/.venv).
  # Provisioning that full venv on a bare worker VM is not yet automated by
  # gpu_cold_start.sh.

  # 4b. Worker daemon's own tiny venv (httpx + boto3 only — see
  #     requirements-worker.txt; the daemon shells out to the pipeline venv
  #     above for actual GPU work, it never imports torch/cv2 itself).
  WORKER_PYTHON_BIN="$(command -v python3.11 || command -v python3)"
  "$WORKER_PYTHON_BIN" -m venv "$WORKER_VENV"
  "$WORKER_VENV/bin/pip" install --upgrade pip
  "$WORKER_VENV/bin/pip" install -r "$REPO_DIR/requirements-worker.txt"

  # 4c. Worker env file from GCE metadata (chmod 600 — never in git; see
  #     product-infra plan "Worker VM env" / INFRA-0 provisioning runbook).
  curl -s -f -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/attributes/pickleball-worker-env' \
    -o "$WORKER_ENV_DIR/worker.env"
  chmod 600 "$WORKER_ENV_DIR/worker.env"

  # 4d. systemd unit + enable --now.
  cp "$REPO_DIR/scripts/worker/pickleball-worker.service" /etc/systemd/system/pickleball-worker.service
  systemctl daemon-reload
  systemctl enable --now pickleball-worker.service

  echo "lane_vm_startup: pickleball-worker install hook complete"
else
  echo "lane_vm_startup: fable-role != pickleball-worker — worker hook skipped"
fi
