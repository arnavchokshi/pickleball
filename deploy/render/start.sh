#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PICKLEBALL_GPU_SSH_KEY_PATH:-}" && -f "${PICKLEBALL_GPU_SSH_KEY_PATH}" ]]; then
  chmod 600 "${PICKLEBALL_GPU_SSH_KEY_PATH}" || true
fi

if [[ -n "${PICKLEBALL_GPU_KNOWN_HOSTS_PATH:-}" && -f "${PICKLEBALL_GPU_KNOWN_HOSTS_PATH}" ]]; then
  chmod 644 "${PICKLEBALL_GPU_KNOWN_HOSTS_PATH}" || true
fi

exec uvicorn server.render_app:app \
  --host 0.0.0.0 \
  --port "${PORT:-10000}" \
  --proxy-headers
