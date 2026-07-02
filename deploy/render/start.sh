#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PICKLEBALL_GPU_SSH_KEY_PATH:-}" && -f "${PICKLEBALL_GPU_SSH_KEY_PATH}" ]]; then
  runtime_key="/tmp/pickleball_gcp_ssh_key"
  cp "${PICKLEBALL_GPU_SSH_KEY_PATH}" "${runtime_key}"
  chmod 600 "${runtime_key}"
  export PICKLEBALL_GPU_SSH_KEY_PATH="${runtime_key}"
fi

if [[ -n "${PICKLEBALL_GPU_KNOWN_HOSTS_PATH:-}" && -f "${PICKLEBALL_GPU_KNOWN_HOSTS_PATH}" ]]; then
  chmod 644 "${PICKLEBALL_GPU_KNOWN_HOSTS_PATH}" || true
fi

exec uvicorn server.render_app:app \
  --host 0.0.0.0 \
  --port "${PORT:-10000}" \
  --proxy-headers
