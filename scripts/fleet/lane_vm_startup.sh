#!/usr/bin/env bash
# Fleet VM startup (FABLE_OPERATING_MANUAL §12) — runs on every boot of a fable-lane VM.
# STATUS: SCAFFOLD — flesh out in the P0-1 GPU cold-start lane; keep steps idempotent.
set -euo pipefail
# 1. One lane per GPU, fail-loud on contention:
nvidia-smi -c EXCLUSIVE_PROCESS || true
# 2. Preemption watcher (belt-and-suspenders alongside the GCE shutdown-script hook):
( while sleep 5; do
    if curl -s -H 'Metadata-Flavor: Google' \
      http://metadata.google.internal/computeMetadata/v1/instance/preempted | grep -q TRUE; then
      touch /tmp/PREEMPTED; break
    fi
  done ) & disown
# 3. Code + weights: git clone/pull the repo (it is pushed) + restore vendor pins per
#    third_party/VENDOR_PINS.md + pull weights per models/MANIFEST.json (see RESET_HANDOFF §7 /
#    scripts/racketsport/gpu_cold_start.sh — proven 258s).
echo "lane_vm_startup: scaffold complete (extend in P0-1 lane)"

# ---------------------------------------------------------------------------
# 4. INFRA-2 pull-worker install hook (product-infra plan §INFRA-2 / §3).
#    INERT on every VM except one booted with GCE metadata
#    `fable-role=pickleball-worker` — training/eval fleet VMs are untouched
#    by construction (the curl below is guarded with `|| true` so a missing
#    metadata attribute on a normal lane VM never trips `set -e` here).
#    Idempotent: safe to re-run on every boot.
# ---------------------------------------------------------------------------
WORKER_FABLE_ROLE="$(curl -s -f -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/attributes/fable-role' 2>/dev/null || true)"
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
