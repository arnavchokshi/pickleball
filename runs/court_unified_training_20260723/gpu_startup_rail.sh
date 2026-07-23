#!/usr/bin/env bash
set -euo pipefail

# Fail closed on spend even if the controlling laptop sleeps or disconnects.
shutdown -P +240
echo "court23 rail armed: automatic poweroff in 240 minutes" | tee /var/log/court23-rail.log

# One training process owns the GPU. Retry briefly while the baked NVIDIA service settles.
for attempt in $(seq 1 60); do
  if nvidia-smi -c EXCLUSIVE_PROCESS; then
    nvidia-smi --query-gpu=name,memory.total,compute_mode --format=csv,noheader \
      | tee -a /var/log/court23-rail.log
    exit 0
  fi
  sleep 5
done

echo "court23 rail failed to configure the GPU" | tee -a /var/log/court23-rail.log >&2
shutdown -P now
exit 1
