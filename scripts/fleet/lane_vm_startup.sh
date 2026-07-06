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
